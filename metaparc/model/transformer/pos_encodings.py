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
