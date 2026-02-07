"""
Tests for positional embedding implementations.
"""

import torch
import pytest
from gphyt.models.transformer.pos_encodings import (
    AbsPositionalEmbedding,
    RotaryPositionalEmbedding,
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
    assert pe.pe.shape == (1, time, height, width, num_channels)


def test_abs_positional_embedding_forward():
    """Test forward pass of AbsPositionalEmbedding."""
    batch_size = 2
    num_channels = 64
    time = 8
    height = 16
    width = 16

    pe = AbsPositionalEmbedding(num_channels, time, height, width)
    x = torch.randn(batch_size, time, height, width, num_channels)

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


def test_rotary_positional_embedding_forward_time():
    """Test forward pass of RotaryPositionalEmbedding."""
    batch_size = 2
    time = 8
    channels = 96  # divisible by 6
    height = 16
    width = 16

    pe = RotaryPositionalEmbedding(channels)
    q = torch.randn(batch_size, time, height, width, channels)
    k = torch.randn(batch_size, time, height, width, channels)

    q_out, k_out = pe(q, k)

    # Check shapes
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

def test_rope_embeddings_are_correct():
    """Test that positional embeddings are correct."""
    batch_size = 2
    time = 8
    height = 16
    width = 16
    channels = 96
    dim_per_component = channels //3

    pe = RotaryPositionalEmbedding(channels)
    q = torch.ones(batch_size, time, height, width, channels)
    k = torch.ones(batch_size, time, height, width, channels)

    q_out, k_out = pe(q, k)

    q_out_time = q_out[:, :, :, :, :dim_per_component]
    k_out_time = k_out[:, :, :, :, :dim_per_component]

    q_out_time_0 = q_out_time[:, 0, :, :, :]
    k_out_time_0 = k_out_time[:, 0, :, :, :]
    q_out_time_1 = q_out_time[:, 1, :, :, :]
    k_out_time_1 = k_out_time[:, 1, :, :, :]

    assert torch.allclose(q_out_time_0, k_out_time_0)
    assert torch.allclose(q_out_time_1, k_out_time_1)
    assert not torch.allclose(q_out_time_0, q_out_time_1)

    q_out_x = q_out[:, :, :, :, dim_per_component:2*dim_per_component]
    k_out_x = k_out[:, :, :, :, dim_per_component:2*dim_per_component]

    q_out_x_0 = q_out_x[:, :, :, 0, :]
    k_out_x_0 = k_out_x[:, :, :, 0, :]
    q_out_x_1 = q_out_x[:, :, :, 1, :]
    k_out_x_1 = k_out_x[:, :, :, 1, :]

    assert torch.allclose(q_out_x_0, k_out_x_0)
    assert torch.allclose(q_out_x_1, k_out_x_1)
    assert not torch.allclose(q_out_x_0, q_out_x_1)


    q_out_y = q_out[:, :, :, :, 2*dim_per_component:]
    k_out_y = k_out[:, :, :, :, 2*dim_per_component:]

    q_out_y_0 = q_out_y[:, :, 0, :, :]
    k_out_y_0 = k_out_y[:, :, 0, :, :]
    q_out_y_1 = q_out_y[:, :, 1, :, :]
    k_out_y_1 = k_out_y[:, :, 1, :, :]

    assert torch.allclose(q_out_y_0, k_out_y_0)
    assert torch.allclose(q_out_y_1, k_out_y_1)
    assert not torch.allclose(q_out_y_0, q_out_y_1)
