"""
Tests for the attention module.
"""

import pytest
import torch
import torch.nn as nn

from metaparc.model.transformer.attention import Attention, MLP, AttentionBlock


class TestAttention:
    """Test suite for the Attention class."""

    def test_init(self):
        """Test initialization of Attention module."""
        hidden_dim = 64
        num_heads = 4
        attention = Attention(hidden_dim, num_heads)

        assert attention.num_heads == num_heads
        assert isinstance(attention.to_qkv, nn.Conv3d)
        assert isinstance(attention.attention, nn.MultiheadAttention)

    def test_forward(self):
        """Test forward pass of Attention module."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        hidden_dim = 64
        num_heads = 4

        x = torch.randn(batch_size, time, height, width, hidden_dim)
        attention = Attention(hidden_dim, num_heads)

        output = attention(x)

        # Check output shape
        assert output.shape == (batch_size, time, height, width, hidden_dim)


class TestMLP:
    """Test suite for the MLP class."""

    def test_init(self):
        """Test initialization of MLP module."""
        hidden_dim = 64
        mlp = MLP(hidden_dim)

        assert isinstance(mlp.mlp, nn.Sequential)
        assert len(mlp.mlp) == 3

    def test_forward(self):
        """Test forward pass of MLP module."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        hidden_dim = 64

        x = torch.randn(batch_size, time, height, width, hidden_dim)
        mlp = MLP(hidden_dim)

        output = mlp(x)

        # Check output shape
        assert output.shape == (batch_size, time, height, width, hidden_dim)


class TestAttentionBlock:
    """Test suite for the AttentionBlock class."""

    def test_init(self):
        """Test initialization of AttentionBlock module."""
        hidden_dim = 64
        num_heads = 4
        dropout = 0.1

        block = AttentionBlock(hidden_dim, num_heads, dropout)

        assert isinstance(block.attention, Attention)
        assert isinstance(block.norm1, nn.InstanceNorm2d)
        assert isinstance(block.norm2, nn.InstanceNorm2d)
        assert isinstance(block.mlp, MLP)

    def test_forward(self):
        """Test forward pass of AttentionBlock module."""
        batch_size = 2
        channels = 64
        time = 3
        height = 8
        width = 8
        num_heads = 4

        x = torch.randn(batch_size, time, height, width, channels)
        block = AttentionBlock(channels, num_heads)

        output = block(x)

        # Check output shape
        assert output.shape == (batch_size, time, height, width, channels)
