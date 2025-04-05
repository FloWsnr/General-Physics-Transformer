"""
Tokenizer module for the transformer model.
By: Florian Wiesner
Date: 2025-04-01
"""

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

class LinearPatchifier(nn.Module):
    """
    Use a linear layer to project the input tensor into patches.

    Parameters
    ----------
    img_size : tuple
        The size of the input (time, height, width).

    patch_size : tuple
        The size of the patches to split the image into (time, height, width).

    in_channels : int
        The number of channels in the input image.

    dim_embed : int
        The dimension of the embedding.
    """
    def __init__(self, img_size: tuple, patch_size: tuple, in_channels: int, dim_embed: int):
        super().__init__()
        img_time, img_height, img_width = img_size
        patch_time, patch_height, patch_width = patch_size

        num_t_patches = img_time // patch_time
        num_h_patches = img_height // patch_height
        num_w_patches = img_width // patch_width
        patch_dim = in_channels * patch_time * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b (t p_t) (h p_h) (w p_w) c -> b (t h w) (p_t p_h p_w c)", p_t = patch_time, p_h = patch_height, p_w = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim_embed),
            nn.LayerNorm(dim_embed),
            Rearrange("b (t h w) d -> b t h w d", t = num_t_patches, h = num_h_patches, w = num_w_patches),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the input tensor into patches.
        """
        x = self.to_patch_embedding(x)
        return x


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

    Parameters
    ----------
    dim_embed : int
        Dimension of the incoming embedding.

    out_channels : int
        Number of channels in the output tensor (number of physics states).

    conv1_size : tuple (time, height, width)
        Size of the patch for the first convolutional layer.

    conv2_size : tuple (time, height, width)
        Size of the patch for the second convolutional layer.
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
