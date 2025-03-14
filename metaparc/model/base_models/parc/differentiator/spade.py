# Utilities/Spade.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """
    Spatially-Adaptive Normalization (SPADE) layer implementation in PyTorch.

    This class normalizes the input feature map and modulates it with scaling (gamma) and shifting (beta)
    parameters that are functions of a spatially-varying mask.

    Args:
        in_channels (int): Number of channels in the input feature map.
        mask_channels (int): Number of channels in the input mask.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 3.
        epsilon (float, optional): Small constant for numerical stability. Default is 1e-5.
        padding_mode (str, optional): Padding mode for `F.pad`. Default is 'constant' (for zero-padding).
    """

    def __init__(
        self,
        in_channels: int,
        mask_channels: int,
        kernel_size: int = 3,
        epsilon: float = 1e-5,
        padding_mode: str = "constant",
    ):
        super(SPADE, self).__init__()
        self.epsilon = epsilon
        self.padding_mode = padding_mode  # Renamed from pad_mode
        self.in_channels = in_channels
        self.mask_channels = mask_channels
        self.kernel_size = kernel_size

        # Define the initial convolutional layer with ReLU activation
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                mask_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=0,
                padding_mode="zeros",
            ),  # Zero padding in Conv2d
            nn.ReLU(),
        )

        # Convolutional layers to generate gamma and beta parameters
        self.gamma_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=0,
            padding_mode="zeros",
        )
        self.beta_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=0,
            padding_mode="zeros",
        )

    def forward(self, x, mask):
        """
        Forward pass of the SPADE layer.

        Args:
            x (torch.Tensor): Input feature map to be normalized. Shape: [N, C, H, W].
            mask (torch.Tensor): Input mask providing spatial modulation. Shape: [N, M, H, W].

        Returns:
            torch.Tensor: The output tensor after applying SPADE normalization. Shape: [N, C, H, W].
        """
        # Compute padding size for convolution to mimic 'valid' padding with external padding
        pad_size = (self.kernel_size - 1) // 2

        # Pad the mask before the initial convolution
        mask_padded = F.pad(
            mask,
            pad=(pad_size, pad_size, pad_size, pad_size),
            mode=self.padding_mode,
            value=0.0,
        )  # Updated parameter name

        # Apply the initial convolution and activation to the mask
        mask_feat = self.initial_conv(mask_padded)

        # Pad the result again before generating gamma and beta
        mask_feat_padded = F.pad(
            mask_feat,
            pad=(pad_size, pad_size, pad_size, pad_size),
            mode=self.padding_mode,
            value=0.0,
        )  # Updated parameter name

        # Generate spatially-adaptive gamma and beta parameters
        gamma = self.gamma_conv(mask_feat_padded)  # Scale parameter
        beta = self.beta_conv(mask_feat_padded)  # Shift parameter

        # Compute the mean and variance of the input tensor across N, H, and W dimensions
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.epsilon)

        # Normalize the input tensor
        x_normalized = (x - mean) / std

        # Apply the spatially-adaptive modulation
        out = gamma * x_normalized + beta

        return out


class SPADEGeneratorUnit(nn.Module):
    """
    SPADE Generator Unit implementation in PyTorch.

    This module represents a SPADE block used in generator architectures, consisting of:
    - Gaussian noise addition
    - Two sequential SPADE-Conv blocks with LeakyReLU activations
    - A skip connection with a SPADE-Conv block

    Args:
        in_channels (int): Number of channels in the input feature map `x`.
        out_channels (int): Number of output channels after convolution.
        mask_channels (int): Number of channels in the input mask `mask`.
        kernel_size (int, optional): Size of the convolutional kernels not in SPADE. Default is 1.
        spade_kernel_size (int, optional): Size of the convolutional kernels in SPADE. Default is 3.
        padding_mode (str, optional): Padding mode for `F.pad`. Default is 'constant'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask_channels: int,
        kernel_size: int = 1,
        spade_kernel_size: int = 3,
        padding_mode: str = "constant",
    ):
        super(SPADEGeneratorUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_channels = mask_channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode  # Renamed from pad_mode

        # Standard deviation for Gaussian noise
        self.noise_std = 0.05

        # Activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # SPADE and convolution layers in the main path
        self.spade1 = SPADE(
            in_channels,
            mask_channels,
            spade_kernel_size,
            padding_mode=padding_mode,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            padding_mode="zeros",
        )

        self.spade2 = SPADE(
            out_channels,
            mask_channels,
            spade_kernel_size,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            padding_mode="zeros",
        )

        # SPADE and convolution layers in the skip connection
        self.spade_skip = SPADE(
            in_channels,
            mask_channels,
            spade_kernel_size,
            padding_mode=padding_mode,
        )
        self.conv_skip = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            padding_mode="zeros",
        )

    def forward(self, x, mask, add_noise: bool):
        """
        Forward pass of the SPADEGeneratorUnit.

        Args:
            x (torch.Tensor): Input feature map. Shape: [N, C_in, H, W].
            mask (torch.Tensor): Input mask for spatial modulation. Shape: [N, M, H', W'].
            add_noise (bool, optional): Whether to add Gaussian noise. If None, defaults to self.training.

        Returns:
            torch.Tensor: The output tensor after processing. Shape: [N, C_out, H', W'].
        """
        if add_noise and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Compute padding size for convolutions
        pad_size = (self.kernel_size - 1) // 2

        # Main path
        spade1_out = self.spade1(x, mask)
        relu1_out = self.leaky_relu(spade1_out)
        # Pad before convolution to mimic 'valid' padding with external padding
        relu1_padded = F.pad(
            relu1_out,
            pad=(pad_size, pad_size, pad_size, pad_size),
            mode=self.padding_mode,
            value=0.0,
        )
        conv1_out = self.conv1(relu1_padded)

        spade2_out = self.spade2(conv1_out, mask)
        relu2_out = self.leaky_relu(spade2_out)
        relu2_padded = F.pad(
            relu2_out,
            pad=(pad_size, pad_size, pad_size, pad_size),
            mode=self.padding_mode,
            value=0.0,
        )
        conv2_out = self.conv2(relu2_padded)

        # Skip connection
        spade_skip_out = self.spade_skip(x, mask)
        relu_skip_out = self.leaky_relu(spade_skip_out)
        relu_skip_padded = F.pad(
            relu_skip_out,
            pad=(pad_size, pad_size, pad_size, pad_size),
            mode=self.padding_mode,
            value=0.0,
        )
        conv_skip_out = self.conv_skip(relu_skip_padded)

        # Add the outputs of the main path and the skip connection
        out = conv2_out + conv_skip_out

        return out
