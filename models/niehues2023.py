from typing import Optional

import torch
from torch import nn

__all__ = ["MILModel", "Attention"]


class MILModel(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_out: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        """Create a new attention MIL model.

        Args:
            n_feats:  The nuber of features each bag instance has.
            n_out:  The number of output layers of the model.
            encoder:  A network transforming bag instances into feature vectors.
        """
        super().__init__()
        self.encoder = encoder or nn.Sequential(nn.Linear(n_feats, 256), nn.ReLU())
        self.attention = attention or Attention(256)
        self.head = head or nn.Sequential(nn.Flatten(), nn.BatchNorm1d(256), nn.Dropout(), nn.Linear(256, n_out))

    def forward(self, bag):
        """
        Wrapper around forward_masked for batch-size-1, unpadded sequences.
        Automatically sets lens = seq_len.
        bag: [1, seq_len, feature_dim]
        """
        assert bag.ndim == 3
        lens = torch.tensor([bag.shape[1]], device=bag.device)
        return self.forward_masked(bag, lens)

    def forward_masked(self, bags, lens):
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]

        embeddings = self.encoder(bags)

        masked_attention_scores = self._masked_attention_scores(embeddings, lens)
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)

        return scores

    def _masked_attention_scores(self, embeddings, lens):
        """Calculates attention scores for all bags.

        Returns:
            A tensor containingtorch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device)

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        masked_attention = torch.where(attention_mask, attention_scores, torch.full_like(attention_scores, -1e10))
        return torch.softmax(masked_attention, dim=1)


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight."""
    n_latent = n_latent or (n_in + 1) // 2

    return nn.Sequential(nn.Linear(n_in, n_latent), nn.Tanh(), nn.Linear(n_latent, 1))


class BinaryLogitWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        out = self.model(*args, **kwargs)
        return (out[:, 0] - out[:, 1]).view(-1)


def Niehues2023(
    pretrained_path: str = "../models/NIEHEUS2023/export-0.pth",
) -> nn.Module:
    mil = MILModel(n_feats=2048, n_out=2)
    ckpt = torch.load(pretrained_path, map_location="cpu")
    e = mil.load_state_dict(ckpt)
    print(e)
    return BinaryLogitWrapper(mil)


if __name__ == "__main__":
    n_feats = 2048
    n_classes = 2
    mil = MILModel(n_feats=n_feats, n_out=n_classes)
    mil.eval()
    print(mil)
    input = torch.rand(1, 1, n_feats)
    output = mil(input)
    print(output)
    assert torch.equal(torch.tensor(output.size()), torch.tensor([1, n_classes]))

    ckpt = torch.load(
        "/home/lydia.schoenpflug/airmec-basec/MSI_stress_testing/existing_models/niehues2023/export-0.pth",
        map_location="cpu",
    )
    print(ckpt)
    e = mil.load_state_dict(ckpt)
    print(e)
