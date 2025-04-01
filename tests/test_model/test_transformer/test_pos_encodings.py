"""
Tests for positional embedding implementations.
"""

import torch
import pytest
from metaparc.model.transformer.pos_encodings import (
    AbsPositionalEmbedding,
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
)


def test_abs_positional_embedding_initialization():
    """Test initialization of AbsPositionalEmbedding."""
    num_channels = 64
    time = 8
    height = 16
    width = 16

    pe = AbsPositionalEmbedding(num_channels, time, height, width)

    assert pe.num_channels == num_channels
    assert pe.time == time
    assert pe.height == height
    assert pe.width == width
    assert pe.pe.shape == (1, time, num_channels, height, width)


def test_abs_positional_embedding_forward():
    """Test forward pass of AbsPositionalEmbedding."""
    batch_size = 2
    num_channels = 64
    time = 8
    height = 16
    width = 16

    pe = AbsPositionalEmbedding(num_channels, time, height, width)
    x = torch.randn(batch_size, time, num_channels, height, width)

    output = pe(x)

    assert output.shape == x.shape
    # Check that positional embeddings were added
    assert not torch.allclose(output, x)


def test_rotary_positional_embedding_initialization():
    """Test initialization of RotaryPositionalEmbedding."""
    dim = 96  # divisible by 6
    pe = RotaryPositionalEmbedding(dim)

    assert pe.time_inv_freq.shape[0] == dim // 6
    assert pe.spatial_inv_freq.shape[0] == dim // 6

    # Test invalid dimension
    with pytest.raises(AssertionError):
        RotaryPositionalEmbedding(95)  # not divisible by 6


def test_rotary_positional_embedding_forward():
    """Test forward pass of RotaryPositionalEmbedding."""
    batch_size = 2
    time = 8
    channels = 96  # divisible by 6
    height = 16
    width = 16

    pe = RotaryPositionalEmbedding(channels)
    x = torch.randn(batch_size, time, channels, height, width)

    time_cos, time_sin, x_cos, x_sin, y_cos, y_sin = pe(x)

    # Check shapes
    dim_per_component = channels // 3
    assert time_cos.shape == (time, dim_per_component)
    assert time_sin.shape == (time, dim_per_component)
    assert x_cos.shape == (width, dim_per_component)
    assert x_sin.shape == (width, dim_per_component)
    assert y_cos.shape == (height, dim_per_component)
    assert y_sin.shape == (height, dim_per_component)


def test_apply_rotary_pos_emb():
    """Test the application of rotary positional embeddings."""
    batch_size = 2
    time = 8
    channels = 96  # divisible by 6
    height = 16
    width = 16

    # Create query and key tensors with channel dimension last
    q = torch.randn(batch_size, time, height, width, channels)
    k = torch.randn(batch_size, time, height, width, channels)

    # Create rotary embeddings
    pe = RotaryPositionalEmbedding(channels)
    time_cos, time_sin, x_cos, x_sin, y_cos, y_sin = pe(q)

    # Apply rotary embeddings
    q_out, k_out = apply_rotary_pos_emb(
        q, k, time_cos, time_sin, x_cos, x_sin, y_cos, y_sin
    )

    # Check shapes
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

    # Check that embeddings were actually applied
    assert not torch.allclose(q_out, q)
    assert not torch.allclose(k_out, k)


def test_rotary_positional_embedding_caching():
    """Test that rotary positional embeddings are properly cached."""
    dim = 96
    pe = RotaryPositionalEmbedding(dim)

    # First forward pass
    x1 = torch.randn(2, 8, dim, 16, 16)
    time_cos1, time_sin1, x_cos1, x_sin1, y_cos1, y_sin1 = pe(x1)

    # Second forward pass with same dimensions
    x2 = torch.randn(2, 8, dim, 16, 16)
    time_cos2, time_sin2, x_cos2, x_sin2, y_cos2, y_sin2 = pe(x2)

    # Check that cached values are reused
    assert torch.equal(time_cos1, time_cos2)
    assert torch.equal(time_sin1, time_sin2)
    assert torch.equal(x_cos1, x_cos2)
    assert torch.equal(x_sin1, x_sin2)
    assert torch.equal(y_cos1, y_cos2)
    assert torch.equal(y_sin1, y_sin2)

    # Different dimensions should trigger recomputation
    x3 = torch.randn(2, 12, dim, 24, 24)
    time_cos3, time_sin3, x_cos3, x_sin3, y_cos3, y_sin3 = pe(x3)

    assert not torch.equal(time_cos1, time_cos3)
    assert not torch.equal(x_cos1, x_cos3)
    assert not torch.equal(y_cos1, y_cos3)
