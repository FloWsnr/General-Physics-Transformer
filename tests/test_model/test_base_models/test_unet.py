import pytest
import torch
import torch.nn as nn
from metaparc.model.base_models.unet import UNetDownBlock, UNetUpBlock, Unet


def test_unet_down_block():
    """
    Test the UNetDownBlock for correct shape transformation and downsampling.
    """
    batch_size = 4
    height = 32
    width = 32
    in_channels = 64
    out_channels = 128

    # Create block and input tensor
    block = UNetDownBlock(in_channels=in_channels, out_channels=out_channels)
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward pass
    output = block(x)

    # Check output shape (should be halved spatially due to maxpool)
    assert output.shape == (batch_size, out_channels, height // 2, width // 2)


def test_unet_up_block_with_skip():
    """
    Test the UNetUpBlock with skip connections.
    """
    batch_size = 4
    height = 16
    width = 16
    in_channels = 128
    out_channels = 64
    skip_channels = 64

    # Create block
    block = UNetUpBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        skip_channels=skip_channels,
        use_concat=True,
    )

    # Create input and skip connection tensors
    x = torch.randn(batch_size, in_channels, height, width)
    skip = torch.randn(batch_size, skip_channels, height * 2, width * 2)

    # Forward pass
    output = block(x, skip)

    # Check output shape (should be doubled spatially due to upsampling)
    assert output.shape == (batch_size, out_channels, height * 2, width * 2)


def test_unet_up_block_without_skip():
    """
    Test the UNetUpBlock without skip connections.
    """
    batch_size = 4
    height = 16
    width = 16
    in_channels = 128
    out_channels = 64

    # Create block without skip connection
    block = UNetUpBlock(
        in_channels=in_channels, out_channels=out_channels, use_concat=False
    )

    # Forward pass without skip connection
    x = torch.randn(batch_size, in_channels, height, width)
    output = block(x, None)

    # Check output shape
    assert output.shape == (batch_size, out_channels, height * 2, width * 2)


def test_reactionnet_shape():
    """
    Test if ReactionNet maintains expected shapes throughout the network.
    """
    batch_size = 4
    height = 64
    width = 64
    input_channels = 3
    output_channels = 2
    block_dimensions = [64, 128, 256]

    # Create model with all skip connections
    model = Unet(
        block_dimensions=block_dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        up_block_use_concat=[True, True],
        skip_connection_indices=[0, 1],
    )

    x = torch.randn(batch_size, input_channels, height, width)
    output = model(x)

    # Check if output shape matches input spatial dimensions
    assert output.shape == (batch_size, output_channels, height, width)


def test_reactionnet_no_skip_connections():
    """
    Test ReactionNet without skip connections.
    """
    batch_size = 4
    height = 64
    width = 64
    input_channels = 3
    output_channels = 2
    block_dimensions = [64, 128, 256]

    # Create model without skip connections
    model = Unet(
        block_dimensions=block_dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        up_block_use_concat=[False, False],
        skip_connection_indices=[],
    )

    x = torch.randn(batch_size, input_channels, height, width)
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, output_channels, height, width)


def test_reactionnet_selective_skip_connections():
    """
    Test ReactionNet with selective skip connections.
    """
    batch_size = 4
    height = 64
    width = 64
    input_channels = 3
    output_channels = 2
    block_dimensions = [64, 128, 256]

    # Create model with selective skip connections
    model = Unet(
        block_dimensions=block_dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        up_block_use_concat=[True, False],
        skip_connection_indices=[1],  # Only use skip connection from middle layer
    )

    x = torch.randn(batch_size, input_channels, height, width)
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, output_channels, height, width)


def test_reactionnet_gradient_flow():
    """
    Test if gradients flow properly through the ReactionNet.
    """
    batch_size = 4
    height = 64
    width = 64
    input_channels = 3
    output_channels = 2
    block_dimensions = [64, 128, 256]

    model = Unet(
        block_dimensions=block_dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        up_block_use_concat=[True, True],
        skip_connection_indices=[0, 1],
    )

    x = torch.randn(batch_size, input_channels, height, width, requires_grad=True)
    output = model(x)

    # Create a dummy loss and check if gradients flow
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_reactionnet_padding_modes():
    """
    Test if ReactionNet works with different padding modes.
    """
    batch_size = 4
    height = 64
    width = 64
    input_channels = 3
    output_channels = 2
    block_dimensions = [64, 128]

    padding_modes = ["zeros", "reflect", "replicate"]

    for mode in padding_modes:
        model = Unet(
            block_dimensions=block_dimensions,
            input_channels=input_channels,
            output_channels=output_channels,
            padding_mode=mode,
            up_block_use_concat=[True],
            skip_connection_indices=[0],
        )

        x = torch.randn(batch_size, input_channels, height, width)
        output = model(x)

        assert output.shape == (batch_size, output_channels, height, width)
