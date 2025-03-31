"""
Axial Attention module
By: Florian Wiesner
Date: 2025-03-31

Inspired by: https://github.com/PolymathicAI/multiple_physics_pretraining/tree/main
"""

import torch
import torch.nn as nn

from einops import rearrange


class AxialAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads

        self.to_qkv = nn.Conv2d(
            hidden_dim, 3 * hidden_dim, kernel_size=1, bias=False
        )  # no bias for qkv projections

        self.attention_x = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            batch_first=True,
        )
        self.attention_y = nn.MultiheadAttention(
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
        q, k, v = qkv.chunk(3, dim=1)  # B, C, H, W

        # NOTE: potentially also norm qk again?

        qx = rearrange(q, "b c h w -> (b h) w c")
        kx = rearrange(k, "b c h w -> (b h) w c")
        vx = rearrange(v, "b c h w -> (b h) w c")
        xx, xx_weights = self.attention_x(qx, kx, vx)
        xx = rearrange(xx, "(b h) w c -> b c h w", h=H)

        qy = rearrange(q, "b c h w -> (b w) h c")
        ky = rearrange(k, "b c h w -> (b w) h c")
        vy = rearrange(v, "b c h w -> (b w) h c")
        yy, yy_weights = self.attention_y(qy, ky, vy)
        yy = rearrange(yy, "(b w) h c -> b c h w", w=W)

        x = (xx + yy) / 2

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
        self.attention = AxialAttention(hidden_dim, num_heads, dropout)
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
