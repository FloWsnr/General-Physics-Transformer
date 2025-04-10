import pytest
import torch
import torch.nn as nn
from metaparc.model.base_models.resnet import ResNet, ResNetBlock


def test_resnet_block_shape():
    """
    Test if ResNetBlock maintains the expected output shape.

    Tests both cases:
    1. When input and output channels are the same
    2. When output channels differ from input channels
    """
    batch_size = 4
    height = 32
    width = 32

    # Test with same channels
    in_channels = 64
    out_channels = 64
    block = ResNetBlock(
        in_channels=in_channels, out_channels=out_channels, kernel_size=3
    )
    x = torch.randn(batch_size, in_channels, height, width)
    output = block(x)
    assert output.shape == (batch_size, out_channels, height, width)

    # Test with different channels
    out_channels = 128
    block = ResNetBlock(
        in_channels=in_channels, out_channels=out_channels, kernel_size=3
    )
    output = block(x)
    assert output.shape == (batch_size, out_channels, height, width)


def test_resnet_shape_no_pooling():
    """
    Test if ResNet maintains spatial dimensions when pooling is disabled.
    """
    batch_size = 4
    height = 32
    width = 32
    in_channels = 3
    block_dimensions = [in_channels, 64, 128, 256]

    model = ResNet(
        block_dimensions=block_dimensions,
        kernel_size=3,
        pooling=False,
    )

    x = torch.randn(batch_size, in_channels, height, width)
    output = model(x)

    # Output should have the last block's channel dimension
    assert output.shape == (batch_size, block_dimensions[-1], height, width)


def test_resnet_shape_with_pooling():
    """
    Test if ResNet properly reduces spatial dimensions when pooling is enabled.
    """
    batch_size = 4
    height = 32
    width = 32
    in_channels = 3
    block_dimensions = [in_channels, 64, 128, 256]

    model = ResNet(
        block_dimensions=block_dimensions,
        kernel_size=3,
        pooling=True,
    )

    x = torch.randn(batch_size, in_channels, height, width)
    output = model(x)

    # Each pooling operation halves the spatial dimensions
    expected_height = height // (2 ** (len(block_dimensions) - 1))
    expected_width = width // (2 ** (len(block_dimensions) - 1))

    assert output.shape == (
        batch_size,
        block_dimensions[-1],
        expected_height,
        expected_width,
    )


def test_resnet_gradient_flow():
    """
    Test if gradients flow properly through the ResNet.
    """
    batch_size = 4
    height = 32
    width = 32
    in_channels = 3
    block_dimensions = [in_channels, 64, 128, 256]

    model = ResNet(
        block_dimensions=block_dimensions,
        kernel_size=3,
        pooling=False,
    )

    x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    output = model(x)

    # Create a dummy loss and check if gradients flow
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_resnet_skip_connection():
    """
    Test if the skip connection in ResNetBlock is functioning properly.
    """
    batch_size = 4
    height = 32
    width = 32
    in_channels = 64
    out_channels = 64

    block = ResNetBlock(
        in_channels=in_channels, out_channels=out_channels, kernel_size=3
    )
    x = torch.ones(batch_size, in_channels, height, width)

    # Store the initial output
    initial_output = block(x)

    # Zero out the main convolution path
    with torch.no_grad():
        block.conv1[0].weight.zero_()
        block.conv2[0].weight.zero_()

    # The output should now be equal to the input (through skip connection)
    # plus any bias terms
    skip_output = block(x)

    # The outputs should be different when the convolution path is active vs zeroed
    assert not torch.allclose(initial_output, skip_output)

    # But the skip_output should still have non-zero values due to the skip connection
    assert not torch.allclose(skip_output, torch.zeros_like(skip_output))


def test_resnet_padding_modes():
    """
    Test if ResNet works with different padding modes.
    """
    batch_size = 4
    height = 32
    width = 32
    in_channels = 3
    block_dimensions = [in_channels, 64, 128, 256]

    padding_modes = ["zeros", "reflect", "replicate"]

    for mode in padding_modes:
        model = ResNet(
            block_dimensions=block_dimensions,
            kernel_size=3,
            pooling=False,
            padding_mode=mode,
        )

        x = torch.randn(batch_size, in_channels, height, width)
        output = model(x)

        assert output.shape == (batch_size, block_dimensions[-1], height, width)
