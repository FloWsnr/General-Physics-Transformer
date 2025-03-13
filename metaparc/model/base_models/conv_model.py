"""
UNet model implementation.
By: Florian Wiesner
Date: 2025-02-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """A convolutional block with batch normalization and ReLU activation."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTransBlock(nn.Module):
    """A convolutional transpose block with batch normalization and ReLU activation."""

    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvTransBlock, self).__init__()
        self.conv_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, padding=0, stride=stride
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvModel(nn.Module):
    """A UNet architecture for image-to-image tasks.

    Architecture:
        - Encoder: Series of Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d blocks
        - Bridge: Conv2d block at bottom of U
        - Decoder: Series of Upsample -> Conv2d -> BatchNorm2d -> ReLU blocks with skip connections

    Parameters
    ----------
        input_channels : int
            Number of input channels
        output_dim : int
            Number of output dimensions/classes
        hidden_channels : int, optional
            Number of features in first hidden layer. Default is 32.
        dropout_rate : float, optional
            Dropout probability. Default is 0.1.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 4,
        hidden_channels: int = 32,
    ):
        super(ConvModel, self).__init__()

        # Encoder path with funnel structure
        self.enc = nn.Sequential(
            ConvBlock(input_channels, hidden_channels, stride=2),
            ConvBlock(hidden_channels, hidden_channels * 2, stride=2),
            ConvBlock(hidden_channels * 2, hidden_channels * 4, stride=2),
        )

        # Bridge
        self.bridge = nn.Sequential(
            ConvBlock(hidden_channels * 4, hidden_channels * 4, stride=1),
        )

        # Decoder path with expanding structure
        self.dec = nn.Sequential(
            ConvTransBlock(hidden_channels * 4, hidden_channels * 2, stride=2),
            ConvTransBlock(hidden_channels * 2, hidden_channels, stride=2),
            ConvTransBlock(hidden_channels, output_channels, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Encoder with funnel structure
        x = self.enc(x)

        # Bridge
        x = self.bridge(x)

        # Decoder with skip connections
        x = self.dec(x)

        # x = self.final(x)
        # Final output
        return x
