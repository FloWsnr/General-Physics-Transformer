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
        self.pe = nn.Parameter(torch.randn(1, time, num_channels, height, width) * 0.02)

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
        x = x + self.pe
        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embeddings for 5D tensors (B, T, C, H, W).

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

    def forward(self, x):
        """
        Compute rotary embeddings for a 5D tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C, H, W)

        Returns
        -------
        tuple
            Tuple containing cos and sin embeddings for time, x, and y dimensions
        """
        B, T, C, H, W = x.shape

        # Recompute time embeddings if needed
        if T != self.time_len_cached:
            self.time_len_cached = T
            t = torch.arange(T, device=x.device).type_as(self.time_inv_freq)
            time_freqs = torch.einsum("i,j->ij", t, self.time_inv_freq)
            time_emb = torch.cat((time_freqs, time_freqs), dim=-1).to(x.device)
            # Shape: (T, dim_per_component)
            self.time_cos_cached = time_emb.cos()
            self.time_sin_cached = time_emb.sin()

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

        return (
            self.time_cos_cached,
            self.time_sin_cached,
            self.x_cos_cached,
            self.x_sin_cached,
            self.y_cos_cached,
            self.y_sin_cached,
        )


def rotate_half(x):
    """
    Rotate half of the dimensions of x.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (..., embedding_dim)

    Returns
    -------
    torch.Tensor
        Tensor with half of its dimensions rotated
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


def apply_rotary_pos_emb(q, k, time_cos, time_sin, x_cos, x_sin, y_cos, y_sin):
    """
    Apply rotary positional embeddings to query and key tensors.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor (..., embedding_dim)

    k : torch.Tensor
        Key tensor (..., embedding_dim)

    time_cos, time_sin : torch.Tensor
        Cosine and sine embeddings for time dimension

    x_cos, x_sin : torch.Tensor
        Cosine and sine embeddings for x spatial dimension

    y_cos, y_sin : torch.Tensor
        Cosine and sine embeddings for y spatial dimension

    Returns
    -------
    tuple
        Tuple containing query and key tensors with rotary embeddings applied
    """
    # Split channels into three equal parts for time, x, and y dimensions
    dim_per_component = q.shape[-1] // 3

    # Split query and key tensors
    q_time, q_x, q_y = torch.split(q, dim_per_component, dim=-1)
    k_time, k_x, k_y = torch.split(k, dim_per_component, dim=-1)

    # Apply rotary embeddings to each dimension
    q_time_out = (q_time * time_cos) + (rotate_half(q_time) * time_sin)
    k_time_out = (k_time * time_cos) + (rotate_half(k_time) * time_sin)

    q_x_out = (q_x * x_cos) + (rotate_half(q_x) * x_sin)
    k_x_out = (k_x * x_cos) + (rotate_half(k_x) * x_sin)

    q_y_out = (q_y * y_cos) + (rotate_half(q_y) * y_sin)
    k_y_out = (k_y * y_cos) + (rotate_half(k_y) * y_sin)

    # Concatenate the results
    q_out = torch.cat([q_time_out, q_x_out, q_y_out], dim=-1)
    k_out = torch.cat([k_time_out, k_x_out, k_y_out], dim=-1)

    return q_out, k_out
