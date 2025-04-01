"""
Positional Embeddings for Transformer Models
"""

import torch
import torch.nn as nn


class AbsPositionalEmbedding(nn.Module):
    """
    Adds absolute positional embeddings to input tensors.
    Works on B, T, C, H, W tensors.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor
    max_seq_len : int
        Maximum sequence length
    """

    def __init__(
        self,
        num_channels: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_seq_len = max_seq_len

        # Learned positional embeddings
        self.pe = nn.Parameter(torch.randn(1, max_seq_len, num_channels) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape B, T, C, H, W

        Returns
        -------
        torch.Tensor
            Tensor with positional embeddings added
        """
        # For 2D data: [batch_size, t, c, h, w]
        b, t, c, h, w = x.shape
        # Reshape to [batch_size, t, c, h*w]
        x = x.reshape(b, t, c, h * w)
        # Transpose to [batch_size, t, h*w, c]
        x = x.permute(0, 3, 1, 2)

        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]

        # Reshape back to original format
        x = x.permute(0, 2, 1)
        x = x.reshape(b, c, h, w)

        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached



def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)