from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NatureNet(nn.Module):
    def __init__(
        self, num_classes: int = 2, pretrained_path: str = "../models/TISSUE_SGM/tissue_sgm_pretrained.pt"
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 1024, kernel_size=11, padding=0),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1, padding=0),
        )
        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path))

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = F.softmax(x, dim=1 if len(x.shape) == 4 else 0)
        return x


def compute_upsampling_and_padding(
    model: nn.Module, input_shape: Tuple[int, int, int] = (3, 224, 224)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute total border pixel loss, upampling factor, and output shape for a CNN model."""
    h, w = input_shape[1], input_shape[2]
    current_shape = np.array([h, w], dtype=np.float32)
    total_padding = np.zeros(4)  # [left, right, top, bottom]
    total_upsample = np.ones(2)  # [height_factor, width_factor]

    for idx, layer in enumerate(model.model):
        if isinstance(layer, nn.Conv2d):
            kh, kw = layer.kernel_size
            sh, sw = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
            out_h = (current_shape[0] - kh) // sh + 1
            out_w = (current_shape[1] - kw) // sw + 1
            # print(f"[{idx}] Conv2d: kernel=({kh},{kw}), stride=({sh},{sw}), in shape: {current_shape}, out shape: {(int(out_h), int(out_w))}")
            current_shape = np.array([out_h, out_w])

        elif isinstance(layer, nn.MaxPool2d):
            kh, kw = (
                layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
            )
            sh, sw = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
            out_h = (current_shape[0] - kh) // sh + 1
            out_w = (current_shape[1] - kw) // sw + 1
            # print(f"[{idx}] MaxPool2d: kernel=({kh},{kw}), stride=({sh},{sw}), in shape: {current_shape}, out shape: {(int(out_h), int(out_w))}")
            current_shape = np.array([out_h, out_w])
            total_upsample *= [sh, sw]

    h_loss = h - current_shape[0] * total_upsample[0]
    w_loss = w - current_shape[1] * total_upsample[1]
    total_padding = np.array([np.ceil(w_loss / 2), np.floor(w_loss / 2), np.ceil(h_loss / 2), np.floor(h_loss / 2)])
    return total_padding.astype(int), total_upsample.astype(int), current_shape.astype(int)


def load_npz_weights_to_model(npz_path: str, model: nn.Module) -> nn.Module:
    weights = np.load(npz_path)
    conv_layers = [layer for layer in model.model if isinstance(layer, nn.Conv2d)]

    for idx, layer in enumerate(conv_layers):
        weight_key = f"conv2d_{idx + 1}_weight_0"
        bias_key = f"conv2d_{idx + 1}_weight_1"

        w = torch.from_numpy(weights[weight_key]).permute(3, 2, 0, 1)  # Keras to PyTorch
        b = torch.from_numpy(weights[bias_key])

        layer.weight.data.copy_(w)
        layer.bias.data.copy_(b)

    return model


def verify_saved_model(npz_path: str, pt_path: str) -> None:
    npz = np.load(npz_path)
    model = NatureNet(num_classes=2)
    state_dict = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    conv_layers = [layer for layer in model.model if isinstance(layer, torch.nn.Conv2d)]

    for idx, layer in enumerate(conv_layers):
        weight_key = f"conv2d_{idx + 1}_weight_0"
        bias_key = f"conv2d_{idx + 1}_weight_1"

        expected_weight = torch.from_numpy(npz[weight_key]).permute(3, 2, 0, 1)
        expected_bias = torch.from_numpy(npz[bias_key])

        model_weight = layer.weight.data
        model_bias = layer.bias.data

        assert model_weight.shape == expected_weight.shape, f"Mismatch in weight shape at layer {idx + 1}"
        assert model_bias.shape == expected_bias.shape, f"Mismatch in bias shape at layer {idx + 1}"
        assert torch.allclose(model_weight, expected_weight, atol=1e-5), f"Mismatch in weights at layer {idx + 1}"
        assert torch.allclose(model_bias, expected_bias, atol=1e-5), f"Mismatch in bias at layer {idx + 1}"

    print("All parameters match correctly.")