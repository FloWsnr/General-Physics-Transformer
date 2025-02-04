"""
Basic convolutional inner model for testing purposes.
By: Florian Wiesner
Date: 2025-02-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModel(nn.Module):
    """A standard convolutional neural network with batch normalization and dropout.

    Architecture:
        - Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        - Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        - Dropout
        - Fully Connected Layer

    Parameters
    ----------
        input_channels : int
            Number of input channels
        output_dim : int
            Number of output dimensions/classes
        hidden_channels : int, optional
            Number of features in hidden layers. Default is 32.
        dropout_rate : float, optional
            Dropout probability. Default is 0.1.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden_channels: int = 32,
        dropout_rate: float = 0.1,
    ):
        super(ConvModel, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels * 2, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate the flattened size for the fully connected layer
        self.fc1 = nn.Linear(hidden_channels * 2 * 16, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels)
        """
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)

        # Flatten and apply dropout
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # Final fully connected layer
        x = self.fc1(x)

        return x
