import torch
import pytest
from metaparc.model.base_models.autoencoder import Encoder


def test_encoder_forward():
    """Test the forward function of the Encoder class."""
    # Define test parameters
    batch_size = 2
    time = 8
    height = 16
    width = 16
    in_channels = 3
    out_channels = 12
    num_layers = 2
    patch_size = (2, 4, 4)  # (time, height, width)
    
    # Create input tensor
    x = torch.randn(batch_size, time, height, width, in_channels)
    
    # Initialize encoder
    encoder = Encoder(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        patch_size=patch_size,
    )
    
    # Forward pass
    output = encoder(x)
    
    # Calculate expected output shape
    expected_time = time // patch_size[0]
    expected_height = height // patch_size[1]
    expected_width = width // patch_size[2]
    
    # Check output shape
    assert output.shape == (batch_size, expected_time, expected_height, expected_width, out_channels)
    
    # Check output is not all zeros or NaNs
    assert not torch.isnan(output).any()
    assert not (output == 0).all()
