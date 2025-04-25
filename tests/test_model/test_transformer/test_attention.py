"""
Tests for the attention module.
"""

import pytest
import torch
import torch.nn as nn

from lpfm.model.transformer.attention import (
    SpatioTemporalAttention,
    CausalSpatioTemporalAttention,
    MLP,
    AttentionBlock,
    SpatialAttention,
    TemporalAttention,
)
from lpfm.model.transformer.pos_encodings import RotaryPositionalEmbedding


class TestSpatioTemporalAttention:
    """Test suite for the SpatioTemporalAttention class."""

    def test_forward(self):
        """Test forward pass of SpatioTemporalAttention module."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        hidden_dim = 64
        num_heads = 4

        x = torch.randn(batch_size, time, height, width, hidden_dim)
        attention = SpatioTemporalAttention(hidden_dim, num_heads)

        output = attention(x)

        # Check output shape
        assert output.shape == (batch_size, time, height, width, hidden_dim)

    def test_forward_with_positional_encoding(self):
        """Test forward pass of SpatioTemporalAttention module with positional encoding."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        hidden_dim = 96
        num_heads = 4

        x = torch.randn(batch_size, time, height, width, hidden_dim)
        attention = SpatioTemporalAttention(
            hidden_dim, num_heads, pe=RotaryPositionalEmbedding(hidden_dim)
        )

        output = attention(x)

        # Check output shape
        assert output.shape == (batch_size, time, height, width, hidden_dim)


class TestCausalSpatioTemporalAttention:
    """Test suite for the CausalSpatioTemporalAttention class."""

    def test_forward(self):
        """Test forward pass of CausalSpatioTemporalAttention module."""
        batch_size = 1
        time = 2
        height = 2
        width = 2
        hidden_dim = 64
        num_heads = 4

        x = torch.randn(batch_size, time, height, width, hidden_dim)
        attention = CausalSpatioTemporalAttention(
            hidden_dim, num_heads, time, height, width, return_att=True
        )

        output, att_weights = attention(x)
        num_patches = time * height * width

        # Check output shape
        assert output.shape == (batch_size, time, height, width, hidden_dim)
        assert att_weights.shape == (batch_size, num_patches, num_patches)


class TestMLP:
    """Test suite for the MLP class."""

    def test_init(self):
        """Test initialization of MLP module."""
        hidden_dim = 64
        mlp_dim = 256
        mlp = MLP(hidden_dim, mlp_dim)

        assert isinstance(mlp.mlp, nn.Sequential)
        assert len(mlp.mlp) == 4

    def test_forward(self):
        """Test forward pass of MLP module."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        hidden_dim = 64
        mlp_dim = 256

        x = torch.randn(batch_size, time, height, width, hidden_dim)
        mlp = MLP(hidden_dim, mlp_dim)

        output = mlp(x)

        # Check output shape
        assert output.shape == (batch_size, time, height, width, hidden_dim)


class TestSpatialAttention:
    """Test suite for the SpatialAttention class."""

    def test_forward(self):
        """Test forward pass of SpatialAttention module."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        hidden_dim = 96
        num_heads = 4

        x = torch.randn(batch_size, time, height, width, hidden_dim)
        attention = SpatialAttention(
            hidden_dim, num_heads, pe=RotaryPositionalEmbedding(hidden_dim)
        )

        output = attention(x)

        # Check output shape
        assert output.shape == (batch_size, time, height, width, hidden_dim)


class TestTemporalAttention:
    """Test suite for the TemporalAttention class."""

    def test_forward(self):
        """Test forward pass of TemporalAttention module."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        hidden_dim = 96
        num_heads = 4

        x = torch.randn(batch_size, time, height, width, hidden_dim)
        attention = TemporalAttention(
            hidden_dim, num_heads, pe=RotaryPositionalEmbedding(hidden_dim)
        )

        output = attention(x)

        # Check output shape
        assert output.shape == (batch_size, time, height, width, hidden_dim)


class TestAttentionBlock:
    """Test suite for the AttentionBlock class."""

    def test_forward(self):
        """Test forward pass of AttentionBlock module."""
        batch_size = 2
        channels = 96
        mlp_dim = 256
        time = 3
        height = 8
        width = 8
        num_heads = 4
        dropout = 0.1

        x = torch.randn(batch_size, time, height, width, channels)
        block = AttentionBlock(
            att_type="full",
            hidden_dim=channels,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        output = block(x)

        # Check output shape
        assert output.shape == (batch_size, time, height, width, channels)


if __name__ == "__main__":
    test = TestCausalSpatioTemporalAttention()
    test.test_forward()
