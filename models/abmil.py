import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnNet(nn.Module):
    # Adapted from https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
    # Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

    def __init__(self, L=1024, D=256, dropout=False, p_dropout_atn=0.25, n_classes=1):
        super(AttnNet, self).__init__()

        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(p_dropout_atn))
            self.attention_b.append(nn.Dropout(p_dropout_atn))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A


class FC_block(nn.Module):
    def __init__(self, dim_in, dim_out, act_layer=nn.ReLU, dropout=True, p_dropout_fc=0.25):
        super(FC_block, self).__init__()

        self.fc = nn.Linear(dim_in, dim_out)
        self.act = act_layer()
        self.drop = nn.Dropout(p_dropout_fc) if dropout else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResBlockFC(nn.Module):
    def __init__(self, dim: int) -> None:
        super(ResBlockFC, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # (N, D)
            out = self.block(x)
            return self.relu(x + out)
        elif x.dim() == 3:  # (B, N, D)
            B, N, D = x.shape
            x_flat = x.view(B * N, D)
            out = self.block(x_flat)
            out = out.view(B, N, D)
            return self.relu(x + out)
        else:
            raise ValueError(f"Expected shape (N, D) or (B, N, D), got {x.shape}")


class ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out


class ClassificationAttentionNet(nn.Module):
    def __init__(
        self,
        input_feature_size=1024,
        precompression_layer=False,
        feature_size_comp=512,
        feature_size_attn=256,
        feature_size_comp_post=128,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
        n_classes=2,
    ):
        super(ClassificationAttentionNet, self).__init__()
        self.n_classes = n_classes

        if precompression_layer:
            self.compression_layer = nn.Sequential(
                FC_block(input_feature_size, feature_size_comp, dropout=False),
                # ResBlockFC(feature_size_comp)
            )
            dim_post_compression = feature_size_comp
        else:
            self.compression_layer = nn.Identity()
            dim_post_compression = input_feature_size

        # From other survival tasks the attention scores are binary (set to class=1).
        self.attention_net = AttnNet(
            L=dim_post_compression,
            D=feature_size_attn,
            dropout=dropout,
            p_dropout_atn=p_dropout_atn,
            n_classes=self.n_classes,
        )

        self.post_compression_layer = nn.Sequential(
            *[
                FC_block(
                    dim_post_compression,
                    feature_size_comp_post,
                    p_dropout_fc=p_dropout_fc,
                )
            ]
        )

        # Classification head.
        self.classifiers = nn.ModuleList([nn.Linear(feature_size_comp_post, 1) for i in range(self.n_classes)])

        # Init weights.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward_attention(self, h):
        A_ = self.attention_net(h)  # h shape is N_tilesxdim
        A_raw = torch.transpose(A_, 1, 0)  # K_attention_classesxN_tiles
        A = F.softmax(A_raw, dim=-1)  # #normalize attentions scores over tiles
        return A_raw, A

    def forward_classification(self, m):

        logits = torch.empty(1, self.n_classes).float().to(m.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](m[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat

    def forward(self, h):

        # H&E embedding
        h = self.compression_layer(h)

        ### Attention MIL
        A_raw, A = self.forward_attention(h)  # 1xN tiles

        ### First-order pooling
        m = A @ h  # torch.Size([1, dim_embedding])  # 1x512 [Sum over N(aihi,1), ..., Sum over N(aihi,dim_embedding)]

        # post compression of he embedding
        m = self.post_compression_layer(m)

        logits, Y_prob, Y_hat = self.forward_classification(m)

        return logits, Y_prob, Y_hat, A_raw, m


class NormModel(nn.Module):
    def __init__(self, model: nn.Module, data_dir: str, eps: float = 1e-6):
        super().__init__()
        logging.info("Initalize normalization model wrapper")
        self.model = model
        self.eps = eps

        norm_whitening_params = np.load(f"{data_dir}/norm_whitening.npz")
        self.register_buffer("min", torch.tensor(norm_whitening_params["min"], dtype=torch.float32))
        self.register_buffer("max", torch.tensor(norm_whitening_params["max"], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            B, N, D = x.shape
            x = x.view(-1, D)  # (B*N,D)

        # Min-max normalization to [-1, 1]
        x_norm = 2 * (self.max - x) / (self.max - self.min + self.eps) - 1
        if x_norm.max() > 1.5 or x_norm.min() < -1.5:
            logging.warning(f"Feature embeddings contain outlier values x_max={x_norm.max()}, x_min={x_norm.min()}!")
        x_norm = torch.clamp(x_norm, -1.0, 1.0)

        if len(x.shape) == 3:
            x_norm = x_norm.view(B, N, D)
        return self.model(x_norm)


class WhitenedModel(nn.Module):
    def __init__(self, model: nn.Module, data_dir: str, eps: float = 1e-6):
        super().__init__()
        logging.info("Initalize normalization and whitening model wrapper")
        self.model = model
        self.eps = eps

        # Load whitening parameters
        norm_whitening_params = np.load(f"{data_dir}/norm_whitening.npz")
        self.register_buffer("mean", torch.tensor(norm_whitening_params["mean"], dtype=torch.float32))
        self.register_buffer(
            "whitening_matrix",
            torch.tensor(norm_whitening_params["whitening_matrix"], dtype=torch.float32),
        )

        # Load min-max normalization values
        # self.register_buffer('feature_min', torch.tensor(norm_whitening_params["whiten_min"], dtype=torch.float32))
        # self.register_buffer('feature_max', torch.tensor(norm_whitening_params["whiten_max"], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            B, N, D = x.shape
            x = x.view(-1, D)  # (B*N,D)

        # Whitening
        x_white = (x - self.mean) @ self.whitening_matrix

        # Min-max normalization to [-1, 1]
        # x_norm = 2 * (self.feature_max - x_white) / (self.feature_max - self.feature_min + self.eps) - 1
        # x_norm = torch.clamp(x_norm, -1.0, 1.0)

        if len(x.shape) == 3:
            x_white = x_white.view(B, N, D)
        return self.model(x_white)
