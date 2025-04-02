"""
Positional Embeddings for Transformer Models
"""

import torch
import torch.nn as nn


class AbsPositionalEmbedding(nn.Module):
    """
    Adds absolute positional embeddings to input tensors.
    Works on B, T, H, W, C tensors.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor
    time : int
        Number of time steps in the input tensor
    height : int
        Height of the input tensor
    width : int
        Width of the input tensor
    """

    def __init__(
        self,
        num_channels: int,
        time: int,
        height: int,
        width: int,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.time = time
        self.height = height
        self.width = width

        # Learned positional embeddings
        self.pe = nn.Parameter(torch.randn(1, time, height, width, num_channels) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape B, T, H, W, C

        Returns
        -------
        torch.Tensor
            Tensor with positional embeddings added
        """
        x = x + self.pe
        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embeddings for 5D tensors (B, T, H, W, C).

    Implements separate rotary embeddings for time dimension and spatial dimensions (x, y).

    Parameters
    ----------
    dim : int
        Dimension of the embeddings. Should be divisible by 6 to allocate evenly across time and spatial dimensions.
    base : int, optional
        Base for the frequency calculation, by default 10000
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        # Ensure dim is divisible by 6 (2 for each of time, x, y dimensions)
        assert dim % 6 == 0, (
            "Dimension must be divisible by 6 for time and spatial (x,y) embeddings"
        )
        dim_per_component = dim // 3

        # Create separate frequency bands for time and spatial dimensions
        time_inv_freq = 1.0 / (
            base ** (torch.arange(0, dim_per_component, 2).float() / dim_per_component)
        )
        spatial_inv_freq = 1.0 / (
            base ** (torch.arange(0, dim_per_component, 2).float() / dim_per_component)
        )

        self.register_buffer("time_inv_freq", time_inv_freq)
        self.register_buffer("spatial_inv_freq", spatial_inv_freq)

        # Cache for computed values
        self.time_len_cached = None
        self.height_cached = None
        self.width_cached = None

        self.time_cos_cached = None
        self.time_sin_cached = None
        self.x_cos_cached = None
        self.x_sin_cached = None
        self.y_cos_cached = None
        self.y_sin_cached = None

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for query and key tensors.

        Parameters
        ----------
        q : torch.Tensor
            Input tensor of shape (B, T, H, W, C)

        k : torch.Tensor
            Input tensor of shape (B, T, H, W, C)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing query and key tensors with rotary embeddings applied
        """

        self._compute_time_embeddings(q)
        self._compute_spatial_embeddings(q)

        q, k = self._apply_rotary_pos_emb(q, k)

        return q, k

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the dimensions of x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (..., channels)

        Returns
        -------
        torch.Tensor
            Tensor with half of its dimensions rotated
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=x1.ndim - 1)

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embeddings to query and key tensors.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (B, T, H, W, C)

        k : torch.Tensor
            Key tensor of shape (B, T, H, W, C)

        Returns
        -------
        tuple
            Tuple containing query and key tensors with rotary embeddings applied
        """

        time_cos = self.time_cos_cached
        time_sin = self.time_sin_cached
        x_cos = self.x_cos_cached
        x_sin = self.x_sin_cached
        y_cos = self.y_cos_cached
        y_sin = self.y_sin_cached

        # Split channels into three equal parts for time, x, and y dimensions
        dim_per_component = q.shape[-1] // 3

        # Split query and key tensors
        q_time, q_x, q_y = torch.split(q, dim_per_component, dim=-1)
        k_time, k_x, k_y = torch.split(k, dim_per_component, dim=-1)

        # Apply rotary embeddings to each dimension
        # Expand to match the shape of the input tensor
        # (1, time, 1, 1, dim_per_component)
        time_cos = time_cos.view(1, time_cos.shape[0], 1, 1, time_cos.shape[1])
        time_sin = time_sin.view(1, time_sin.shape[0], 1, 1, time_sin.shape[1])
        q_time_out = (q_time * time_cos) + (self._rotate_half(q_time) * time_sin)
        k_time_out = (k_time * time_cos) + (self._rotate_half(k_time) * time_sin)

        # Expand to match the shape of the input tensor
        # (1, 1, height, 1, dim_per_component)
        x_cos = x_cos.view(1, 1, x_cos.shape[0], 1, x_cos.shape[1])
        x_sin = x_sin.view(1, 1, x_sin.shape[0], 1, x_sin.shape[1])
        q_x_out = (q_x * x_cos) + (self._rotate_half(q_x) * x_sin)
        k_x_out = (k_x * x_cos) + (self._rotate_half(k_x) * x_sin)

        # Expand to match the shape of the input tensor
        # (1, 1, 1, width, dim_per_component)
        y_cos = y_cos.view(1, 1, 1, y_cos.shape[0], y_cos.shape[1])
        y_sin = y_sin.view(1, 1, 1, y_sin.shape[0], y_sin.shape[1])
        q_y_out = (q_y * y_cos) + (self._rotate_half(q_y) * y_sin)
        k_y_out = (k_y * y_cos) + (self._rotate_half(k_y) * y_sin)

        # Concatenate the results
        q_out = torch.cat([q_time_out, q_x_out, q_y_out], dim=-1)
        k_out = torch.cat([k_time_out, k_x_out, k_y_out], dim=-1)

        return q_out, k_out

    def _compute_time_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape

        # Recompute time embeddings if needed
        if T != self.time_len_cached:
            self.time_len_cached = T
            t = torch.arange(T, device=x.device).type_as(self.time_inv_freq)
            time_freqs = torch.einsum("i,j->ij", t, self.time_inv_freq)
            time_emb = torch.cat((time_freqs, time_freqs), dim=-1).to(x.device)
            # Shape: (T, dim_per_component)
            self.time_cos_cached = time_emb.cos()
            self.time_sin_cached = time_emb.sin()

    def _compute_spatial_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        # Recompute spatial embeddings if needed
        if H != self.height_cached or W != self.width_cached:
            self.height_cached = H
            self.width_cached = W

            # X-dimension (width)
            x_pos = torch.arange(W, device=x.device).type_as(self.spatial_inv_freq)
            x_freqs = torch.einsum("i,j->ij", x_pos, self.spatial_inv_freq)
            x_emb = torch.cat((x_freqs, x_freqs), dim=-1).to(x.device)
            # Shape: (W, dim_per_component)
            self.x_cos_cached = x_emb.cos()
            self.x_sin_cached = x_emb.sin()

            # Y-dimension (height)
            y_pos = torch.arange(H, device=x.device).type_as(self.spatial_inv_freq)
            y_freqs = torch.einsum("i,j->ij", y_pos, self.spatial_inv_freq)
            y_emb = torch.cat((y_freqs, y_freqs), dim=-1).to(x.device)
            # Shape: (H, dim_per_component)
            self.y_cos_cached = y_emb.cos()
            self.y_sin_cached = y_emb.sin()
