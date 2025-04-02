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

        self.to_qkv = nn.Conv3d(
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
        B, T, H, W, C = x.shape
        x = rearrange(x, "b t h w c -> b c t h w")
        qkv = self.to_qkv(x)  # B, 3C, T, H, W
        qkv = rearrange(qkv, "b c t h w -> b (t h w) c")  # B, num_patches, 3C
        q, k, v = qkv.chunk(3, dim=-1)  # B, num_patches, C

        # NOTE: potentially also norm qk again?
        x, att_weights = self.attention(q, k, v)
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=H, w=W)

        return x


class MLP(nn.Module):
    """
    MLP with 1x1 convolutions.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, padding="valid"),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, padding="valid"),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.mlp(x)
        x = rearrange(x, "b c t h w -> b t h w c")
        return x


class AttentionBlock(nn.Module):
    """
    Attention block with axial attention and MLP.
    Input is normalized pre-attention and pre-MLP.

    Input shape: (B, T, H, W, C)
    Output shape: (B, T, H, W, C)

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
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.norm1(input)
        att = self.attention(x) + input  # skip connection across attention and norm
        x = self.norm2(att)
        x = self.mlp(x) + att  # skip connection across MLP and norm
        return x
