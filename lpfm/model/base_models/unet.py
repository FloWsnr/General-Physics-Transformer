import torch
import torch.nn as nn
from typing import Optional


class UNetDownBlock(nn.Module):
    """
    U-Net Downsampling Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels after convolution.
    kernel_size : int, optional
        Size of the convolutional kernels. Default is 3.
    padding : Union[int, tuple, str], optional
        Padding for the convolutional layers. Default is 'same'.
    padding_mode : str, optional
        Padding mode for convolutional layers. Default is 'zeros'.

    Notes
    -----
    Performs two convolutional operations followed by a max pooling operation
    to reduce the spatial dimensions of the input tensor while increasing the
    number of feature channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding="same",
        padding_mode="zeros",
    ):
        super(UNetDownBlock, self).__init__()
        self.padding_mode = padding_mode

        self.doubleConv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.doubleConv(x)
        x = self.pool(x)
        return x


class UNetUpBlock(nn.Module):
    """
    U-Net Upsampling Block.

    Performs upsampling using a transposed convolutional layer, concatenates
    the corresponding skip connection from the downsampling path, and applies two
    convolutional operations to refine the features.

    Parameters
    ----------
    in_channels : int
        Number of input channels from the previous layer.
    out_channels : int
        Number of output channels after convolution.
    skip_channels : int
        Number of channels from the skip connection.
    kernel_size : int, optional
        Size of the convolutional kernels. Default is 3.
    padding_mode : str, optional
        Padding mode for convolutional layers. Default is 'zeros'.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        kernel_size=3,
        padding="same",
        padding_mode="zeros",
    ):
        super(UNetUpBlock, self).__init__()

        out_channels_up = in_channels // 2

        self.upConv = nn.ConvTranspose2d(
            in_channels,
            out_channels_up,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        # Calculate the number of input channels for the convolution after concatenation
        conv_in_channels = out_channels_up + skip_channels

        # Define the double convolution layers with LeakyReLU activations
        self.doubleConv = nn.Sequential(
            nn.Conv2d(
                conv_in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x, skip_connection: torch.Tensor):
        # Apply transposed convolution to upsample
        x = self.upConv(x)
        # Concatenate along the channel dimension
        x = torch.cat((skip_connection, x), dim=1)
        # Apply double convolution
        x = self.doubleConv(x)
        return x


class DoubleConv(nn.Module):
    """
    Double Convolution Block.

    Performs two convolutional operations followed by a LeakyReLU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels after convolution.
    kernel_size : int, optional
        Size of the convolutional kernels. Default is 3.
    padding : Union[int, tuple, str], optional
        Padding for the convolutional layers. Default is 'same'.
    padding_mode : str, optional
        Padding mode for convolutional layers. Default is 'zeros'.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding="same",
        padding_mode="zeros",
    ):
        super(DoubleConv, self).__init__()
        self.padding_mode = padding_mode

        self.doubleConv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.doubleConv(x)


class Unet(nn.Module):
    """
    U-Net Model.

    Constructs a U-Net architecture with customizable depth and feature dimensions.
    Uses skip connections between corresponding layers in the encoder and decoder paths.

    Parameters
    ----------
    block_dimensions : list of int
        List of feature dimensions for each block.
    output_channels : int
        Number of output channels of the final layer.
    kernel_size : int, optional
        Size of the convolutional kernels. Default is 3.
    padding : Union[int, tuple, str], optional
        Padding for the convolutional layers. Default is 'same'.
    padding_mode : str, optional
        Padding mode for convolutional layers. Default is 'zeros'.
    """

    def __init__(
        self,
        block_dimensions,
        input_channels,
        output_channels,
        kernel_size=3,
        padding="same",
        padding_mode="zeros",
    ):
        super(Unet, self).__init__()
        self.padding_mode = padding_mode

        # Initial double convolution layer with LeakyReLU activations
        self.doubleConv = DoubleConv(
            in_channels=input_channels,
            out_channels=block_dimensions[0],
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        # Downsampling path
        self.downBlocks = nn.ModuleList()  # List to hold downsampling blocks
        in_channels = block_dimensions[0]  # Initialize input channels

        # Construct the downsampling blocks
        for out_channels in block_dimensions[1:]:
            down_block = UNetDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            )
            self.downBlocks.append(down_block)
            in_channels = out_channels

        # Upsampling path
        self.upBlocks = nn.ModuleList()
        num_up_blocks = len(block_dimensions) - 1

        # Initialize input channels for the first upsampling block
        in_channels = block_dimensions[-1]

        # Construct the upsampling blocks
        for idx in range(num_up_blocks):
            if idx != num_up_blocks - 1:
                out_channels = block_dimensions[-(idx + 2)]
            else:
                out_channels = output_channels

            # Create an upsampling block
            up_block = UNetUpBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                skip_channels=block_dimensions[-(idx + 2)],
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            )
            self.upBlocks.append(up_block)
            # Update input channels for the next block
            in_channels = out_channels

        # Final convolution to map to the desired number of output channels
        self.finalConv = DoubleConv(
            output_channels, output_channels, kernel_size, padding, padding_mode
        )

    def forward(self, x):
        x = self.doubleConv(x)
        skip_connections = [x]

        # Downsampling path
        for down_block in self.downBlocks:
            x = down_block(x)
            skip_connections.append(x)

        # Upsampling path
        for idx, up_block in enumerate(self.upBlocks):
            skip_connection = skip_connections[-(idx + 2)]
            x = up_block(x, skip_connection)

        # Apply final convolution to get the desired output channels
        x = self.finalConv(x)
        return x
