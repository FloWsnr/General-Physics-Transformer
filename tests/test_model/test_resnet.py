"""
Tests for the ResNet model.
"""

import pytest
import torch

from gphyt.model.resnet import ResNet, get_model
from gphyt.model.model_specs import ResNet_M, ResNet_S


def test_forward():
    """Test basic forward pass."""
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 32, 32, 3)

    resnet = ResNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=16,
        n_layers=2,
    )
    output = resnet(data)
    assert output.shape == (10, 1, 32, 32, 3)


def test_forward_single_timestep():
    """Test forward pass with only one timestep in input."""
    # (batch_size, time=1, height, width, channels)
    data = torch.randn(10, 1, 32, 32, 3)

    resnet = ResNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=16,
        n_layers=2,
    )
    output = resnet(data)
    assert output.shape == (10, 1, 32, 32, 3)


def test_forward_cuda():
    """Test forward pass on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = torch.randn(10, 8, 32, 32, 3).cuda()
    resnet = ResNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=16,
        n_layers=2,
    )
    resnet.cuda()
    output = resnet(data)
    assert output.shape == (10, 1, 32, 32, 3)


def test_get_model():
    """Test the get_model function."""
    resnet_m = get_model(ResNet_M())
    assert isinstance(resnet_m, ResNet)

    resnet_s = get_model(ResNet_S())
    assert isinstance(resnet_s, ResNet)
