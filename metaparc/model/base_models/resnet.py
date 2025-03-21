# Utilities/ResNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class ResNetBlock(nn.Module):
    """
    Residual Block for ResNet with support for changing feature dimensions.

    x --> Conv2d --> ReLU --> Conv2d --> ReLU
       |                              |
       -------- Identity/Conv2d -------

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int
        Stride for the convolutional layers.
    padding : Union[int, tuple, str]
        Padding for the convolutional layers.
    padding_mode : str
        Padding mode for convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, tuple, str] = "same",
        padding_mode: str = "zeros",
    ):
        super(ResNetBlock, self).__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_size = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.ReLU(),
        )

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
        # Final activation after addition
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip_conv(x)
        out = self.relu2(out + skip)
        return out


class ResNet(nn.Module):
    """
    ResNet model consisting of multiple residual blocks with varying feature dimensions.

    The model initializes with a convolutional layer and iteratively adds residual blocks
    based on the provided `block_dimensions`. Optionally, it applies max pooling after
    each residual block to reduce spatial dimensions.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    block_dimensions : List[int]
        List specifying the number of feature channels for each residual block.
    kernel_size : int, optional
        Size of the convolutional kernel. Default is 3.
    pooling : bool, optional
        Whether to apply max pooling after each residual block. Default is False.
    padding : Union[int, tuple, str], optional
        Padding for the convolutional layers. Default is 'zeros'.
    padding_mode : str, optional
        Padding mode for convolutional layers. Default is 'zeros'.
    stride : int, optional
        Stride for the convolutional layers. Default is 1.
    """

    def __init__(
        self,
        in_channels: int,
        block_dimensions: List[int],
        kernel_size: int = 3,
        pooling: bool = False,
        padding: Union[int, tuple, str] = "same",
        padding_mode: str = "zeros",
        stride: int = 1,
    ):
        super(ResNet, self).__init__()
        self.pooling = pooling
        self.padding_mode = padding_mode
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

        # Initial convolutions to go to block size
        self.conv_init = nn.Sequential(
            nn.Conv2d(
                in_channels,
                block_dimensions[0],
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.ReLU(),
            nn.Conv2d(
                block_dimensions[0],
                block_dimensions[0],
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.ReLU(),
        )

        # The blocks
        module_list = []
        for dim_in, dim_out in zip(block_dimensions[:-1], block_dimensions[1:]):
            res_block = ResNetBlock(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                stride=stride,
            )
            module_list.append(res_block)
            if self.pooling:
                module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.resnet = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_init(x)
        out = self.resnet(out)
        return out
