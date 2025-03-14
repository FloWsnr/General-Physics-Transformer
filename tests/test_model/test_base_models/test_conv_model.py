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
        "output_channels": 4,
        "hidden_channels": 32,
    }


@pytest.fixture
def model(model_params):
    """Fixture for creating a ConvModel instance."""
    return ConvModel(**model_params)


def test_model_initialization(model: ConvModel):
    """Test if the model initializes with correct parameters."""
    assert isinstance(model, nn.Module)


def test_forward_pass_shape(model: ConvModel, model_params: dict):
    """Test if the forward pass produces correct output shapes."""
    batch_size = 4
    height, width = 32, 32

    # Create random input tensor
    x = torch.randn(batch_size, model_params["input_channels"], height, width)

    # Run forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, model_params["output_channels"], height, width)
    assert output.shape == expected_shape


def test_forward_pass_different_input_sizes(model: ConvModel, model_params: dict):
    """Test if the model handles different input sizes correctly."""
    batch_size = 2
    input_sizes = [(16, 16), (32, 32), (64, 64)]

    for height, width in input_sizes:
        x = torch.randn(batch_size, model_params["input_channels"], height, width)
        with torch.no_grad():
            output = model(x)

        expected_shape = (batch_size, model_params["output_channels"], height, width)
        assert output.shape == expected_shape


def test_model_with_gpu():
    """Test if the model can be moved to GPU."""
    model = ConvModel(input_channels=3, output_channels=4, hidden_channels=32)
    model = model.to("cuda")

    x = torch.randn(1, 3, 32, 32)
    x = x.to("cuda")
    output = model(x)
    assert output.shape == (1, 4, 32, 32)
