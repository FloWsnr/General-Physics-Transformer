# Utilities/ResNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResNetBlock(nn.Module):
    """
    Residual Block for ResNet with support for changing feature dimensions.

    x --> Conv2d --> ReLU --> Conv2d --> ReLU
       |                              |
       -------- Identity/Conv2d -------

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        padding_mode (str): Padding mode for convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding_mode: str,
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
                kernel_size=kernel_size,
                padding=0,
                padding_mode="zeros",
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=0,
                padding_mode="zeros",
            ),
            nn.ReLU(),
        )

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.skip_conv = nn.Identity()
        # Final activation after addition
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pad(
            x,
            pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            mode=self.padding_mode,
            value=0.0,
        )
        out = self.conv1(out)
        out = F.pad(
            out,
            pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            mode=self.padding_mode,
            value=0.0,
        )
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

    Args:
        in_channels (int): Number of input channels.
        block_dimensions (List[int]): List specifying the number of feature channels for each residual block.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        pooling (bool, optional): Whether to apply max pooling after each residual block. Default is False.
        padding_mode (str, optional): Padding mode for convolutional layers. Default is 'constant'.
    """

    def __init__(
        self,
        in_channels: int,
        block_dimensions: List[int],
        kernel_size: int = 3,
        pooling: bool = False,
        padding_mode: str = "constant",
    ):
        super(ResNet, self).__init__()
        self.padding_mode = padding_mode
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.pad_size = (kernel_size - 1) // 2

        # Double convolution + ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                block_dimensions[0],
                kernel_size=kernel_size,
                padding=0,
                padding_mode="zeros",
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                block_dimensions[0],
                block_dimensions[0],
                kernel_size=kernel_size,
                padding=0,
                padding_mode="zeros",
            ),
            nn.ReLU(),
        )
        module_list = []
        # The blocks
        for i in range(len(block_dimensions)):
            if i == 0:
                in_channels = block_dimensions[0]
            else:
                in_channels = block_dimensions[i - 1]
            res_block = ResNetBlock(
                in_channels=in_channels,
                out_channels=block_dimensions[i],
                kernel_size=kernel_size,
                padding_mode=padding_mode,
            )
            module_list.append(res_block)
            if self.pooling:
                module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.path = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pad(
            x,
            pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            mode=self.padding_mode,
            value=0.0,
        )
        out = self.conv1(out)
        out = F.pad(
            out,
            pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            mode=self.padding_mode,
            value=0.0,
        )
        out = self.conv2(out)
        out = self.path(out)
        return out
