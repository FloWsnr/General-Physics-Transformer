"""
Normalization Layers

By: Florian Wiesner
Date: 2025-03-31

Reversible Instance Normalization
Taken from: https://github.com/ts-kim/RevIN

Spatially-Adaptive Normalization
From PARC: https://github.com/baeklab/PARCtorch
"""

import torch
import torch.nn as nn

from einops import rearrange


class RevLN(nn.Module):
    def __init__(self, height: int, width: int, eps=1e-5):
        """
        Reversible Layer Normalization for tensors with shape (B, Time, H, W, C).
        Normalize only over the channel dimension.

        Parameters
        ----------
        num_channels : int
            The number of channels
        eps : float, optional
            A value added for numerical stability, by default 1e-5
        """
        super(RevLN, self).__init__()
        self.height = height
        self.width = width
        self.eps = eps
        # initialize RevLN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.height, self.width))
        self.affine_bias = nn.Parameter(torch.zeros(self.height, self.width))

    def forward(self, x, mode: str):
        """
        Forward pass for RevLN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, H, W, C)
        mode : str
            Mode of operation: 'norm' for normalization, 'denorm' for denormalization

        Returns
        -------
        torch.Tensor
            Normalized or denormalized tensor
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # For (B, T, H, W, C), reduce over time, channel dimension
        dim2reduce = (1, 4)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        )

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        # Reshape affine parameters to match the channel dimension
        weight = self.affine_weight.view(1, 1, self.height, self.width, 1)
        bias = self.affine_bias.view(1, 1, self.height, self.width, 1)
        x = x * weight
        x = x + bias
        return x

    def _denormalize(self, x):
        # Reshape affine parameters to match the channel dimension
        weight = self.affine_weight.view(1, 1, self.height, self.width, 1)
        bias = self.affine_bias.view(1, 1, self.height, self.width, 1)
        x = x - bias
        x = x / (weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class RevIN(nn.Module):
    def __init__(self, num_channels: int, eps=1e-5):
        """
        Reversible Instance Normalization for tensors with shape (B, Time, H, W, C).
        Normalizes each channel independently over the Time, Height, and Width dimensions.
        Affine transformation is done over the channel dimension, one parameter for all
        timesteps.

        Parameters
        ----------
        num_channels : int
            The number of channels
        eps : float, optional
            A value added for numerical stability, by default 1e-5
        """
        super(RevIN, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_channels))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x, mode: str):
        """
        Forward pass for RevIN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, H, W, C)
        mode : str
            Mode of operation: 'norm' for normalization, 'denorm' for denormalization

        Returns
        -------
        torch.Tensor
            Normalized or denormalized tensor
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # For (B, T, H, W, C), reduce over time, height, width dimensions (1, 2, 3)
        dim2reduce = (1, 2, 3)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        )

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        # Reshape affine parameters to match the channel dimension
        weight = self.affine_weight.view(1, 1, 1, 1, -1)
        bias = self.affine_bias.view(1, 1, 1, 1, -1)
        x = x * weight
        x = x + bias
        return x

    def _denormalize(self, x):
        # Reshape affine parameters to match the channel dimension
        weight = self.affine_weight.view(1, 1, 1, 1, -1)
        bias = self.affine_bias.view(1, 1, 1, 1, -1)
        x = x - bias
        x = x / (weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class RevSPADE_3D(nn.Module):
    """
    Spatially-Adaptive Normalization (SPADE) layer implementation in PyTorch.

    Normalization is done over the time, height, and width dimensions and thus per instance.
    In other words, per sample and per channel, different scaling and shifting parameters are used.

    Affine transformation is spatially adaptive (i.e. changes across the images).
    It is done over the channel dimension but averaged over the time dimension.

    This 3D version uses 3D convolutions over the time, height, and width dimensions.

    Addtionally, with an option to reverse the normalization, similar to RevIN.

    Input shape: (B, Time, C, H, W)
    Mask shape: (B, Time, Mask, H, W)


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
        eps: float = 1e-5,
    ):
        super(RevSPADE_3D, self).__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.mask_channels = mask_channels
        self.kernel_size = kernel_size

        self.register_buffer("mean", torch.zeros(in_channels))
        self.register_buffer("stdev", torch.ones(in_channels))
        self.beta = torch.zeros(in_channels)
        self.gamma = torch.ones(in_channels)

        # Define the initial convolutional layer with ReLU activation
        self.initial_conv = nn.Sequential(
            nn.Conv3d(
                mask_channels,
                in_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode="zeros",
            ),  # Zero padding in Conv2d
            nn.GELU(),
        )

        # Convolutional layers to generate gamma and beta parameters
        self.gamma_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="zeros",
        )
        self.beta_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="zeros",
        )

    def forward(self, x, mask, mode: str):
        """
        Forward pass of the SPADE layer.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map to be normalized. Shape: [B, T, C, H, W].
        mask : torch.Tensor
            Input mask providing spatial modulation. Shape: [B, T, M, H, W].
        mode : str
            Mode of operation, either "norm" for normalization or "denorm" for denormalization.

        Returns
        -------
        torch.Tensor
            The output tensor after applying SPADE normalization or denormalization. Shape: [B, T, C, H, W].
        """

        if mode == "norm":
            # move time dimension after channels
            mask = rearrange(mask, "b t m h w -> b m t h w")

            # Apply the initial convolution and activation to the mask
            mask_feat = self.initial_conv(mask)

            # Generate spatially-adaptive gamma and beta parameters
            gamma = self.gamma_conv(mask_feat)  # Scale parameter
            beta = self.beta_conv(mask_feat)  # Shift parameter

            # move time dimension back
            gamma = rearrange(gamma, "b m t h w -> b t m h w")
            beta = rearrange(beta, "b m t h w -> b t m h w")

            # Average over the time dimension
            gamma = torch.mean(gamma, dim=1, keepdim=True)
            beta = torch.mean(beta, dim=1, keepdim=True)
            self.gamma = gamma
            self.beta = beta

            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # For (B, T, C, H, W), reduce over T, H, W dimensions (1, 3, 4)
        dim2reduce = (1, 3, 4)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        )

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        x = x * self.gamma
        x = x + self.beta
        return x

    def _denormalize(self, x):
        x = x - self.beta
        x = x / self.gamma
        x = x * self.stdev
        x = x + self.mean
        return x


class RevSPADE_2D(nn.Module):
    """
    Spatially-Adaptive Normalization (SPADE) layer implementation in PyTorch.

    Normalization and affine transformation is done over the time, height, and width dimensions
    and thus per instance. In other words, per sample, per channel different
    scaling and shifting parameters are used.

    This 2D version uses 2D convolutions over the height and width dimensions.
    The time dimension is combined with the channel dimension.

    Addtionally, with an option to reverse the normalization, similar to RevIN.

    Input shape: (B, Time, C, H, W)
    Mask shape: (B, Time, Mask, H, W)


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
        eps: float = 1e-5,
    ):
        super(RevSPADE_2D, self).__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.mask_channels = mask_channels
        self.kernel_size = kernel_size

        self.register_buffer("mean", torch.zeros(in_channels))
        self.register_buffer("stdev", torch.ones(in_channels))
        self.beta = torch.zeros(in_channels)
        self.gamma = torch.ones(in_channels)

        # Define the initial convolutional layer with ReLU activation
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                mask_channels,
                in_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode="zeros",
            ),  # Zero padding in Conv2d
            nn.GELU(),
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

    def forward(self, x, mask, mode: str):
        """
        Forward pass of the SPADE layer.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map to be normalized. Shape: [B, T, C, H, W].
        mask : torch.Tensor
            Input mask providing spatial modulation. Shape: [B, T, M, H, W].
        mode : str
            Mode of operation, either "norm" for normalization or "denorm" for denormalization.

        Returns
        -------
        torch.Tensor
            The output tensor after applying SPADE normalization or denormalization. Shape: [B, T, C, H, W].
        """

        B, T, M, H, W = mask.shape

        if mode == "norm":
            # move time dimension after channels
            mask = rearrange(mask, "b t m h w -> b (t m) h w")

            # Apply the initial convolution and activation to the mask
            mask_feat = self.initial_conv(mask)

            # Generate spatially-adaptive gamma and beta parameters
            gamma = self.gamma_conv(mask_feat)  # Scale parameter
            beta = self.beta_conv(mask_feat)  # Shift parameter

            # move time dimension back
            gamma = rearrange(gamma, "b (t m) h w -> b t m h w", t=T)
            beta = rearrange(beta, "b (t m) h w -> b t m h w", t=T)

            # Average over the time dimension
            gamma = torch.mean(gamma, dim=1, keepdim=True)
            beta = torch.mean(beta, dim=1, keepdim=True)

            self.gamma = gamma
            self.beta = beta

            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # For (B, T, C, H, W), reduce over T, H, W dimensions (1, 3, 4)
        dim2reduce = (1, 3, 4)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        )

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        x = x * self.gamma
        x = x + self.beta
        return x

    def _denormalize(self, x):
        x = x - self.beta
        x = x / self.gamma
        x = x * self.stdev
        x = x + self.mean
        return x
