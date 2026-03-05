import os
from collections.abc import Callable
from typing import Dict

import timm
import torch
from huggingface_hub import login

from models.resnet50 import resnet50

try:
    from timm.layers.helpers import to_2tuple
except:
    from timm.models.layers.helpers import to_2tuple

INPUT_FEATURE_SIZE = {
    "univ2": 1536,
    "virchow2": 1280,
    "hoptimus1": 1536,
    "ctranspath": 768,
    "retccl": 2048,
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
HOPTIMUS_MEAN = [0.707223, 0.578729, 0.703617]
HOPTIMUS_STD = [0.211883, 0.230117, 0.177517]

ENCODER_NORMALIZATIONS = {
    "univ2": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "hoptimus1": {"mean": HOPTIMUS_MEAN, "std": HOPTIMUS_STD},
    "virchow2": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "ctranspath": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "retccl": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
}


def load_encoder(encoder: str, device: torch.device, pretrained_path: str = None) -> torch.nn.Module:
    if pretrained_path is not None:
        model = ENCODER_LOADERS[encoder](pretrained_path=pretrained_path)
    else:
        model = ENCODER_LOADERS[encoder]()
    model.to(device)
    model.eval()
    return model


def hugging_face_login() -> None:
    token_id = ""  ### <------- Add your token
    login(token_id)


def UNIv2(
    pretrained_path: str = "../models/UNIv2",
) -> torch.nn.Module:
    # timm kwargs taken from https://github.com/mahmoodlab/uni?tab=readme-ov-file#2-downloading-weights--creating-model
    model = timm.create_model(
        "vit_giant_patch14_224",
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    # Load the pretrained weights
    e = model.load_state_dict(
        torch.load(
            os.path.join(pretrained_path, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    print(e)
    return model


def HOptimus1(
    pretrained_path: str = "../models/HOPTIMUS1",
) -> torch.nn.Module:
    hugging_face_login()
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1",
        init_values=1e-5,
        dynamic_img_size=True,
    )
    # Load the pretrained weights
    e = model.load_state_dict(
        torch.load(
            os.path.join(pretrained_path, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    print(e)
    return model


def Virchow2(
    pretrained_path: str = "../models/VIRCHOW2",
) -> torch.nn.Module:
    class VirchowWithPostprocessing(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            output = self.base_model(x)  # [B, 261, 1280]
            class_token = output[:, 0]  # [B, 1280]
            class_token_plus_mean = torch.cat([class_token, output[:, 5:].mean(dim=1)], dim=-1)  # [B, 2560]
            return class_token  # we take CLS-ONLY as it was shown to outperform CLS+MEAN for Virchow2 https://arxiv.org/html/2408.00738v3

    hugging_face_login()
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=False,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        dynamic_img_size=True,
    )

    # Load the pretrained weights
    e = model.load_state_dict(
        torch.load(
            os.path.join(pretrained_path, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )

    print(e)
    model = VirchowWithPostprocessing(model)
    return model


def CTransPath(
    pretrained_path: str = "../models/CTRANSPATH",
) -> torch.nn.Module:
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        embed_layer=ConvStem,
        pretrained=False,
        # dynamic_img_size=True,
    )
    model.head = torch.nn.Identity()
    print(model)
    e = model.load_state_dict(
        torch.load(
            os.path.join(pretrained_path, "ctranspath.pth"),
            map_location="cpu",
        )["model"],
        strict=True,
    )
    print(e)
    return model


def RetCCL(
    pretrained_path: str = "../models/RetCCL/best_ckpt.pth",
):
    model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
    model.fc = torch.nn.Identity()
    e = model.load_state_dict(torch.load(pretrained_path), strict=True)
    print(e)
    return model


ENCODER_LOADERS: Dict[str, Callable] = {
    "univ2": UNIv2,  # Uni2-h
    "hoptimus1": HOptimus1,
    "virchow2": Virchow2,
    "ctranspath": CTransPath,
    "retccl": RetCCL,
}


class ConvStem(torch.nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(
                torch.nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(torch.nn.BatchNorm2d(output_dim))
            stem.append(torch.nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(torch.nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = torch.nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
