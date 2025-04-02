"""
Tokenizer module for the transformer model.
By: Florian Wiesner
Date: 2025-04-01
"""

import torch
import torch.nn as nn

from einops import rearrange


class SpatioTemporalTokenization(nn.Module):
    """
    Tokenizes the input tensor (b, time, height, width, channels) into spatio-temporal tokens.

    Tokenization is done by two 3D convolutions with GELU and instance normalization.
    The final patch size is the product of the two patch sizes, i.e.
    time size = conv1_size[0] * conv2_size[0]
    height size = conv1_size[1] * conv2_size[1]
    width size = conv1_size[2] * conv2_size[2]

    For pure spatial tokenization, use a time size of 1.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor (number of physics states).
    dim_embed : int
        Dimension of the embedding.
    conv1_size : tuple (time, height, width)
        Size of the patch for the first convolutional layer.
    conv2_size : tuple (time, height, width)
        Size of the patch for the second convolutional layer.
    """

    def __init__(
        self,
        in_channels: int,
        dim_embed: int,
        conv1_size: tuple,
        conv2_size: tuple,
    ):
        super().__init__()

        token_net = [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=dim_embed // 2,
                kernel_size=conv1_size,
                stride=conv1_size,
                padding="valid",
                bias=False,
            ),
            nn.InstanceNorm3d(
                num_features=dim_embed // 2,
                affine=True,
            ),
            nn.GELU(),
            nn.Conv3d(
                in_channels=dim_embed // 2,
                out_channels=dim_embed,
                kernel_size=conv2_size,
                stride=conv2_size,
                padding="valid",
                bias=False,
            ),
            nn.InstanceNorm3d(
                num_features=dim_embed,
                affine=True,
            ),
            nn.GELU(),
        ]

        self.token_net = nn.Sequential(*token_net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenizes the input tensor into spatial and temporal tokens.
        """
        b, t, h, w, c = x.shape
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.token_net(x)
        x = rearrange(x, "b c t h w -> b t h w c")
        return x


class SpatioTemporalDetokenization(nn.Module):
    """
    Converts back spatio-temporal tokens into a physical tensor.

    The detokenization is done by two transposed convolutions with GELU and instance normalization.

    Parameters
    ----------
    dim_embed : int
        Dimension of the incoming embedding.

    out_channels : int
        Number of channels in the output tensor (number of physics states).

    conv1_size : tuple (time, height, width)
        Size of the patch for the first transposed convolutional layer.

    conv2_size : tuple (time, height, width)
        Size of the patch for the second transposed convolutional layer.
    """

    def __init__(
        self,
        dim_embed: int,
        out_channels: int,
        conv1_size: tuple,
        conv2_size: tuple,
    ):
        super().__init__()
        self.register_buffer("patch_size", torch.tensor(16))
        self.in_channels = dim_embed
        self.out_channels = out_channels
        de_patch_net = [
            nn.ConvTranspose3d(
                in_channels=self.in_channels,
                out_channels=self.in_channels // 2,
                kernel_size=conv1_size,
                stride=conv1_size,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm3d(
                num_features=self.in_channels // 2,
                affine=True,
            ),
            nn.GELU(),
            nn.ConvTranspose3d(
                in_channels=self.in_channels // 2,
                out_channels=self.out_channels,
                kernel_size=conv2_size,
                stride=conv2_size,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm3d(
                num_features=self.out_channels,
                affine=True,
            ),
            nn.GELU(),
        ]

        self.de_patch_net = nn.Sequential(*de_patch_net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a 5D input tensor (batch_size, time, height, width, channels) to a 5D output tensor of images.
        """
        b, t, h, w, c = x.shape
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.de_patch_net(x)
        x = rearrange(x, "b c t h w -> b t h w c")
        return x
