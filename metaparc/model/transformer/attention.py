"""
Attention module
By: Florian Wiesner
Date: 2025-03-31
"""

import torch
import torch.nn as nn

from einops import rearrange


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads

        self.to_qkv = nn.Conv2d(
            hidden_dim, 3 * hidden_dim, kernel_size=1, bias=False
        )  # no bias for qkv projections

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            batch_first=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)  # B, 3C, H, W
        qkv = rearrange(qkv, "b c h w -> b (h w) c")
        q, k, v = qkv.chunk(3, dim=-1)  # B, H*W, C

        # NOTE: potentially also norm qk again?
        x = self.attention(q, k, v)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x


class AttentionBlock(nn.Module):
    """
    Attention block with axial attention and MLP.
    Input is normalized pre-attention and pre-MLP.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)

    Parameters
    ----------
    hidden_dim: int
        Hidden dimension of the input.
    num_heads: int
        Number of attention heads.
    dropout: float
        Dropout rate.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = Attention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.mlp = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x.clone()
        x = self.norm1(x)
        x = self.attention(x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x + skip
