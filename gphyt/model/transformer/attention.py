"""
Attention module
By: Florian Wiesner
Date: 2025-03-31
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
from einops import rearrange

from gphyt.model.transformer.pos_encodings import RotaryPositionalEmbedding


class AbstractAttention(nn.Module):
    """
    Abstract attention class.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.pe = pe

        self.to_qkv = nn.Linear(
            hidden_dim, 3 * hidden_dim, bias=False
        )  # no bias for qkv projections

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            batch_first=True,
        )

    def _add_pos_embeddings(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pe is not None:
            q, k = self.pe(q, k)
        return q, k

    def _get_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.to_qkv(x)  # B, T, H, W, 3C
        q, k, v = qkv.chunk(3, dim=-1)  # B, T, H, W, C
        q, k = self._add_pos_embeddings(q, k)

        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SpatioTemporalAttention(AbstractAttention):
    """
    Full attention over time, height, and width.
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
    pe: Optional[RotaryPositionalEmbedding]
        Rotary positional embedding.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__(hidden_dim, num_heads, dropout, pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape

        q, k, v = self._get_qkv(x)  # B, T, H, W, C
        q = rearrange(q, "b t h w c -> b (t h w) c")
        k = rearrange(k, "b t h w c -> b (t h w) c")
        v = rearrange(v, "b t h w c -> b (t h w) c")

        # NOTE: potentially also norm qk again?
        x, att_weights = self.attention(q, k, v)
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=H, w=W)

        return x


class CausalSpatioTemporalAttention(AbstractAttention):
    """
    Full attention over time, height, and width.
    Use a causal mask to prevent attending to future timesteps.
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
    pe: Optional[RotaryPositionalEmbedding]
        Rotary positional embedding.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        time: int,
        height: int,
        width: int,
        dropout: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
        return_att: bool = False,
    ):
        super().__init__(hidden_dim, num_heads, dropout, pe)
        self.return_att = return_att
        # Calculate total number of patches (tokens) and patches per timestep
        num_patches = time * height * width
        patches_per_timestep = height * width

        # Create indices for each patch (token) from 0 to N-1
        indices = torch.arange(num_patches)

        # Calculate the timestep 't' for each flattened patch index 'n'
        # using the formula: t = n // (H * W)
        token_timesteps = indices // patches_per_timestep  # Shape: (N,)

        # --- Create the Causal Mask ---
        # Expand timesteps to compare each query timestep with each key timestep
        query_timesteps = token_timesteps.unsqueeze(
            1
        )  # Shape: (N, 1) -> Represents query token timesteps
        key_timesteps = token_timesteps.unsqueeze(
            0
        )  # Shape: (1, N) -> Represents key token timesteps

        # Generate the boolean mask. mask[i, j] is True if the key token j
        # is from a future timestep relative to the query token i.
        # This means query i cannot attend to key j.
        causal_mask = key_timesteps > query_timesteps  # Shape: (N, N)
        self.register_buffer("mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape

        q, k, v = self._get_qkv(x)  # B, T, H, W, C
        q = rearrange(q, "b t h w c -> b (t h w) c")
        k = rearrange(k, "b t h w c -> b (t h w) c")
        v = rearrange(v, "b t h w c -> b (t h w) c")

        # NOTE: potentially also norm qk again?
        x, att_weights = self.attention(
            q, k, v, attn_mask=self.mask, need_weights=self.return_att
        )
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=H, w=W)

        if self.return_att:
            return x, att_weights
        else:
            return x


class SpatialAttention(AbstractAttention):
    """
    Spatial attention over height and width.

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

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__(hidden_dim, num_heads, dropout, pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        q, k, v = self._get_qkv(x)  # B, T, H, W, C

        # NOTE: potentially also norm qk again?
        q = rearrange(q, "b t h w c -> (b t) (h w) c")
        k = rearrange(k, "b t h w c -> (b t) (h w) c")
        v = rearrange(v, "b t h w c -> (b t) (h w) c")

        x, att_weights = self.attention(q, k, v)
        x = rearrange(x, "(b t) (h w) c -> b t h w c", t=T, h=H, w=W)

        return x


class TemporalAttention(AbstractAttention):
    """
    Temporal attention over time.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__(hidden_dim, num_heads, dropout, pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the temporal attention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, H, W, C)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape

        q, k, v = self._get_qkv(x)  # B, T, H, W, C

        # NOTE: potentially also norm qk again?
        q = rearrange(q, "b t h w c -> (b h w) t c")
        k = rearrange(k, "b t h w c -> (b h w) t c")
        v = rearrange(v, "b t h w c -> (b h w) t c")

        x, att_weights = self.attention(q, k, v)
        x = rearrange(x, "(b h w) t c -> b t h w c", h=H, w=W, t=T)

        return x


class MLP(nn.Module):
    """
    MLP with linear layers.
    """

    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = self.mlp(x)
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
    mlp_dim: int
        Hidden dimension of the MLP.
    num_heads: int
        Number of attention heads.
    dropout: float
        Dropout rate.
    """

    def __init__(
        self,
        att_type: str,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        time: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__()
        if att_type == "full":
            self.attention = SpatioTemporalAttention(hidden_dim, num_heads, dropout, pe)
        elif att_type == "spatial":
            self.attention = SpatialAttention(hidden_dim, num_heads, dropout, pe)
        elif att_type == "temporal":
            self.attention = TemporalAttention(hidden_dim, num_heads, dropout, pe)
        elif att_type == "full_causal":
            if time is None or height is None or width is None:
                raise ValueError(
                    "time, height, and width must be provided for causal attention"
                )
            self.attention = CausalSpatioTemporalAttention(
                hidden_dim, num_heads, time, height, width, dropout, pe
            )
        else:
            raise ValueError(f"Invalid attention type: {att_type}")
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, dropout)
        self.sd = StochasticDepth(stochastic_depth_rate, mode="row")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # skip connection across attention and norm (with stochastic depth)
        x = self.norm1(input)
        att = self.sd(self.attention(x)) + input

        # skip connection across MLP and norm (with stochastic depth)
        x = self.norm2(att)
        x = self.sd(self.mlp(x)) + att
        return x
