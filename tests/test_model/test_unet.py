"""
Tests for the UNet model.
"""

import pytest
import torch

from gphyt.model.unet import UNet, get_model, DownBlock, UpBlock
from gphyt.model.model_specs import UNet_M, UNet_S


def test_down_block():
    """Test DownBlock forward pass."""
    down_block = DownBlock(in_channels=64, out_channels=128)
    x = torch.randn(2, 64, 32, 32)
    skip, down = down_block(x)
    assert skip.shape == (2, 128, 32, 32)
    assert down.shape == (2, 128, 16, 16)


def test_up_block():
    """Test UpBlock forward pass."""
    up_block = UpBlock(in_channels=256, skip_channels=128, out_channels=64)
    x = torch.randn(2, 256, 8, 8)
    skip = torch.randn(2, 128, 16, 16)
    up = up_block(x, skip)
    assert up.shape == (2, 64, 16, 16)


def test_unet_forward():
    """Test basic UNet forward pass."""
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 32, 32, 3)
    unet = UNet(
        in_channels=3,  # channels only, time is merged in model
        out_channels=3,
        starting_hidden_dim=64,
        n_down_blocks=3,
        n_time_steps=8,
    )
    output = unet(data)
    assert output.shape == (10, 1, 32, 32, 3)


def test_unet_forward_single_timestep():
    """Test UNet forward pass with only one timestep in input."""
    # (batch_size, time=1, height, width, channels)
    data = torch.randn(10, 1, 32, 32, 3)
    unet = UNet(
        in_channels=3,
        out_channels=3,
        starting_hidden_dim=64,
        n_down_blocks=2,
        n_time_steps=1,
    )
    output = unet(data)
    assert output.shape == (10, 1, 32, 32, 3)


def test_unet_different_input_output_channels():
    """Test UNet with different input and output channels."""
    # (batch_size, time, height, width, channels)
    data = torch.randn(5, 4, 64, 64, 5)
    unet = UNet(
        in_channels=5,
        out_channels=2,
        starting_hidden_dim=32,
        n_down_blocks=2,
        n_time_steps=4,
    )
    output = unet(data)
    assert output.shape == (5, 1, 64, 64, 2)


def test_unet_forward_cuda():
    """Test UNet forward pass on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    data = torch.randn(10, 8, 32, 32, 3).cuda()
    unet = UNet(
        in_channels=3,
        out_channels=3,
        starting_hidden_dim=64,
        n_down_blocks=3,
        n_time_steps=8,
    )
    unet.cuda()
    output = unet(data)
    assert output.shape == (10, 1, 32, 32, 3)


def test_get_model():
    """Test the get_model function."""
    unet_m = get_model(UNet_M(), n_time_steps=8)
    assert isinstance(unet_m, UNet)
    unet_s = get_model(UNet_S(), n_time_steps=8)
    assert isinstance(unet_s, UNet)


def test_unet_s_config():
    """Test UNet with UNet_S configuration."""
    data = torch.randn(2, 4, 32, 32, 5)
    unet = get_model(UNet_S(), n_time_steps=4)
    output = unet(data)
    assert output.shape == (2, 1, 32, 32, 5)


def test_unet_m_config():
    """Test UNet with UNet_M configuration."""
    data = torch.randn(2, 4, 32, 32, 5)
    unet = get_model(UNet_M(), n_time_steps=4)
    output = unet(data)
    assert output.shape == (2, 1, 32, 32, 5)


def test_unet_gradient_flow():
    """Test that gradients flow through the UNet properly."""
    data = torch.randn(2, 4, 32, 32, 3, requires_grad=True)
    unet = UNet(
        in_channels=3,
        out_channels=3,
        starting_hidden_dim=32,
        n_down_blocks=2,
        n_time_steps=4,
    )

    output = unet(data)
    loss = output.sum()
    loss.backward()

    # Check that input gradients exist
    assert data.grad is not None
    assert data.grad.shape == data.shape

    # Check that model parameters have gradients
    for param in unet.parameters():
        assert param.grad is not None
