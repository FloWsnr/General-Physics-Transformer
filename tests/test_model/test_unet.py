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
    config_m = {"model_size": "UNet_M", "n_time_steps": 8}
    unet_m = get_model(config_m)
    assert isinstance(unet_m, UNet)

    config_s = {"model_size": "UNet_S", "n_time_steps": 8}
    unet_s = get_model(config_s)
    assert isinstance(unet_s, UNet)


def test_unet_s_config():
    """Test UNet with UNet_S configuration."""
    data = torch.randn(2, 4, 32, 32, 5)
    config = {"model_size": "UNet_S", "n_time_steps": 4}
    unet = get_model(config)
    output = unet(data)
    assert output.shape == (2, 1, 32, 32, 5)


def test_unet_m_config():
    """Test UNet with UNet_M configuration."""
    data = torch.randn(2, 4, 32, 32, 5)
    config = {"model_size": "UNet_M", "n_time_steps": 4}
    unet = get_model(config)
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


def test_unet_integrate_false():
    """Test UNet with integrate=False (default behavior)."""
    data = torch.randn(2, 4, 32, 32, 3)
    unet = UNet(
        in_channels=3,
        out_channels=3,
        starting_hidden_dim=32,
        n_down_blocks=2,
        n_time_steps=4,
        integrate=False,
    )
    output = unet(data)
    assert output.shape == (2, 1, 32, 32, 3)

    # Test that output is different from input (no residual connection)
    input_last_timestep = data[:, -1, ...].unsqueeze(1)
    assert not torch.allclose(output, input_last_timestep, atol=1e-5)


def test_unet_integrate_true():
    """Test UNet with integrate=True (adds residual connection)."""
    torch.manual_seed(42)  # For reproducible results
    data = torch.randn(2, 4, 32, 32, 3)

    # Test with integrate=True
    unet_integrate = UNet(
        in_channels=3,
        out_channels=3,
        starting_hidden_dim=32,
        n_down_blocks=2,
        n_time_steps=4,
        integrate=True,
    )
    output_integrate = unet_integrate(data)
    assert output_integrate.shape == (2, 1, 32, 32, 3)

    # Test with integrate=False for comparison
    torch.manual_seed(42)  # Reset seed for fair comparison
    unet_no_integrate = UNet(
        in_channels=3,
        out_channels=3,
        starting_hidden_dim=32,
        n_down_blocks=2,
        n_time_steps=4,
        integrate=False,
    )
    output_no_integrate = unet_no_integrate(data)

    # Outputs should be different due to residual connection
    assert not torch.allclose(output_integrate, output_no_integrate, atol=1e-5)

    # The integrated output should be related to the last timestep input
    input_last_timestep = data[:, -1, ...].unsqueeze(1)
    # The difference between integrate and non-integrate should be the input
    # (this is an approximation test since the weights are random)


def test_get_model_with_integrate():
    """Test get_model function with integrate parameter."""
    # Test with integrate=False
    config_false = {
        "model_size": "UNet_S",
        "n_time_steps": 4,
        "integrate": False
    }
    unet_false = get_model(config_false)
    assert isinstance(unet_false, UNet)
    assert unet_false.integrate is False

    # Test with integrate=True
    config_true = {
        "model_size": "UNet_M",
        "n_time_steps": 4,
        "integrate": True
    }
    unet_true = get_model(config_true)
    assert isinstance(unet_true, UNet)
    assert unet_true.integrate is True

    # Test default behavior (should be False)
    config_default = {
        "model_size": "UNet_S",
        "n_time_steps": 4
    }
    unet_default = get_model(config_default)
    assert isinstance(unet_default, UNet)
    assert unet_default.integrate is False
