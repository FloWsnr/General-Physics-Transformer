"""
Unit tests for the ConvModel class.
"""

import pytest
import torch
import torch.nn as nn
from metaparc.model.base_models.conv_model import ConvModel


@pytest.fixture
def model_params():
    """Fixture for default model parameters."""
    return {
        "input_channels": 3,
        "output_dim": 4,
        "hidden_channels": 32,
        "dropout_rate": 0.1,
    }


@pytest.fixture
def model(model_params):
    """Fixture for creating a ConvModel instance."""
    return ConvModel(**model_params)


def test_model_initialization(model, model_params):
    """Test if the model initializes with correct parameters."""
    assert isinstance(model, nn.Module)
    assert isinstance(model.conv1, nn.Conv2d)
    assert isinstance(model.conv2, nn.Conv2d)
    assert isinstance(model.bn1, nn.BatchNorm2d)
    assert isinstance(model.bn2, nn.BatchNorm2d)
    assert isinstance(model.pool1, nn.MaxPool2d)
    assert isinstance(model.pool2, nn.MaxPool2d)
    assert isinstance(model.dropout, nn.Dropout)
    assert isinstance(model.fc1, nn.Linear)

    # Check conv1 parameters
    assert model.conv1.in_channels == model_params["input_channels"]
    assert model.conv1.out_channels == model_params["hidden_channels"]

    # Check conv2 parameters
    assert model.conv2.in_channels == model_params["hidden_channels"]
    assert model.conv2.out_channels == model_params["hidden_channels"] * 2

    # Check dropout rate
    assert model.dropout.p == model_params["dropout_rate"]


def test_forward_pass_shape(model, model_params):
    """Test if the forward pass produces correct output shapes."""
    batch_size = 4
    height, width = 32, 32

    # Create random input tensor
    x = torch.randn(batch_size, model_params["input_channels"], height, width)

    # Run forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, model_params["output_dim"])
    assert output.shape == expected_shape


def test_forward_pass_different_input_sizes(model, model_params):
    """Test if the model handles different input sizes correctly."""
    batch_size = 2
    input_sizes = [(16, 16), (32, 32), (64, 64)]

    for height, width in input_sizes:
        x = torch.randn(batch_size, model_params["input_channels"], height, width)
        with torch.no_grad():
            output = model(x)

        expected_shape = (batch_size, model_params["output_dim"])
        assert output.shape == expected_shape
