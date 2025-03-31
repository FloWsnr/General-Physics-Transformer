import torch
import torch.nn as nn


class SPADE(nn.Module):
    """
    Spatially-Adaptive Normalization (SPADE) layer implementation in PyTorch.

    This class normalizes the input feature map and modulates it with scaling (gamma) and shifting (beta)
    parameters that are functions of a spatially-varying mask.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map.
    mask_channels : int
        Number of channels in the input mask.
    kernel_size : int, optional
        Size of the convolutional kernels. Default is 3.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-5.
    """

    def __init__(
        self,
        in_channels: int,
        mask_channels: int,
        kernel_size: int = 3,
        epsilon: float = 1e-5,
    ):
        super(SPADE, self).__init__()
        self.epsilon = epsilon
        self.in_channels = in_channels
        self.mask_channels = mask_channels
        self.kernel_size = kernel_size

        # Define the initial convolutional layer with ReLU activation
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                mask_channels,
                in_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode="zeros",
            ),  # Zero padding in Conv2d
            nn.ReLU(),
        )

        # Convolutional layers to generate gamma and beta parameters
        self.gamma_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="zeros",
        )
        self.beta_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
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
        # Apply the initial convolution and activation to the mask
        mask_feat = self.initial_conv(mask)

        # Generate spatially-adaptive gamma and beta parameters
        gamma = self.gamma_conv(mask_feat)  # Scale parameter
        beta = self.beta_conv(mask_feat)  # Shift parameter

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

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map `x`.
    out_channels : int
        Number of output channels after convolution.
    mask_channels : int
        Number of channels in the input mask `mask`.
    kernel_size : int, optional
        Size of the convolutional kernels not in SPADE. Default is 1.
    spade_kernel_size : int, optional
        Size of the convolutional kernels in SPADE. Default is 3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask_channels: int,
        kernel_size: int = 1,
        spade_kernel_size: int = 3,
    ):
        super(SPADEGeneratorUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_channels = mask_channels
        self.kernel_size = kernel_size

        # Standard deviation for Gaussian noise
        self.noise_std = 0.05

        # Activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # SPADE and convolution layers in the main path
        self.spade1 = SPADE(
            in_channels,
            mask_channels,
            spade_kernel_size,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="zeros",
        )

        self.spade2 = SPADE(
            out_channels,
            mask_channels,
            spade_kernel_size,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="zeros",
        )

        # SPADE and convolution layers in the skip connection
        self.spade_skip = SPADE(
            in_channels,
            mask_channels,
            spade_kernel_size,
        )
        self.conv_skip = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="zeros",
        )

    def forward(self, x, mask, add_noise: bool):
        """
        Forward pass of the SPADEGeneratorUnit.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map. Shape: [N, C_in, H, W].
        mask : torch.Tensor
            Input mask for spatial modulation. Shape: [N, M, H, W].
        add_noise : bool, optional
            Whether to add Gaussian noise. If None, defaults to self.training.

        Returns
        -------
        torch.Tensor
            The output tensor after processing. Shape: [N, C_out, H, W].
        """
        if add_noise and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Main path
        out = self.spade1(x, mask)
        out = self.leaky_relu(out)

        out = self.conv1(out)

        out = self.spade2(out, mask)
        out = self.leaky_relu(out)

        out = self.conv2(out)

        # Skip connection
        skip_out = self.spade_skip(x, mask)
        skip_out = self.leaky_relu(skip_out)

        skip_out = self.conv_skip(skip_out)

        # Add the outputs of the main path and the skip connection
        out = out + skip_out

        return out
