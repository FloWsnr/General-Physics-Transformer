import pytest
import torch
import torch.nn as nn

from gphyt.models.transformer.ax_attention import AxialAttention


def test_axial_attention():
    """
    Test the AxialAttention module.

    Tests the initialization and forward pass of the AxialAttention module
    with various input configurations.
    """
    # Test parameters
    batch_size = 2
    hidden_dim = 64
    height = 16
    width = 16
    num_heads = 8

    # Create dummy input
    x = torch.randn(batch_size, hidden_dim, height, width)

    # Initialize module
    ax_attention = AxialAttention(
        hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1
    )
    # Forward pass
    output = ax_attention(x)

    # Check output shape
    assert output.shape == (batch_size, hidden_dim, height, width), (
        f"Expected shape {(batch_size, hidden_dim, height, width)}, got {output.shape}"
    )

    # Check output type
    assert output.dtype == x.dtype, f"Expected dtype {x.dtype}, got {output.dtype}"
