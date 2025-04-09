"""
Tokenizer module for the transformer model.
By: Florian Wiesner
Date: 2025-04-01
"""

from typing import Optional
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from metaparc.model.base_models.autoencoder import Encoder, Decoder


def get_patch_conv_size(
    patch_size: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Get the conv sizes depending on the desired patch size.

    Parameters
    ----------
    patch_size : tuple
        The size of the patches to split the image into (time, height, width).

    Returns
    -------
    conv1_size : tuple
        The size of the patch for the first convolutional layer.
    conv2_size : tuple
        The size of the patch for the second convolutional layer.
    """
    t1, t2 = find_closest_factors_power_of_two(patch_size[0])
    h1, h2 = find_closest_factors_power_of_two(patch_size[1])
    w1, w2 = find_closest_factors_power_of_two(patch_size[2])

    conv1_size = (t1, h1, w1)
    conv2_size = (t2, h2, w2)
    return conv1_size, conv2_size


def find_closest_factors_power_of_two(n: int) -> tuple[int, int]:
    """
    Finds two factors (a, b) of a power of two 'n' such that a*b=n
    and 'a' and 'b' are the closest possible powers of two.

    Parameters
    ----------
    n : int
        The number to find the closest factors of.

    Returns
    -------
    factor1 : int
        The first factor.
    factor2 : int
        The second factor.
    """
    if n <= 0 or (n & (n - 1) != 0):
        raise ValueError("Input must be a positive power of two.")

    # Using the exponent method
    k = n.bit_length() - 1  # Efficient way to get log2(n) for powers of 2

    exp1 = k // 2  # Integer division equivalent to floor(k/2)
    exp2 = (k + 1) // 2  # Equivalent to ceil(k/2) for integers

    factor1 = 1 << exp1  # Efficient power of 2: 2**exp1
    factor2 = 1 << exp2  # Efficient power of 2: 2**exp2

    return factor1, factor2


class Tokenizer(nn.Module):
    """
    Base class for tokenizers.

    Parameters
    ----------
    patch_size : tuple
        The size of the patches to split the image into (time, height, width).
    in_channels : int
        The number of channels in the input image.
    dim_embed : int
        The dimension of the embedding.
    mode : str
        The mode of the tokenizer. Can be "linear" or "non_linear".
        Non-linear uses two 3D convolutions with GELU and instance normalization.
    autoencoder_channels : list, optional
        The hidden channels of the autoencoder.
    """

    def __init__(
        self,
        patch_size: tuple,
        in_channels: int,
        dim_embed: int,
        mode: str,
        decoder_channels: Optional[list] = None,
    ):
        super().__init__()

        self.register_buffer("patch_size", torch.tensor(patch_size))

        self.in_channels = in_channels
        self.dim_embed = dim_embed
        self.mode = mode

        if self.mode == "linear":
            self.tokenizer = LinearTokenizer(
                patch_size=patch_size,
                in_channels=in_channels,
                dim_embed=dim_embed,
            )
        elif self.mode == "non_linear":
            self.tokenizer = NonlinearTokenizer(
                in_channels=in_channels,
                dim_embed=dim_embed,
                patch_size=patch_size,
            )
        elif self.mode == "autoencoder":
            self.tokenizer = Encoder(
                channels=[in_channels, *decoder_channels, dim_embed],
                patch_size=patch_size,
            )
        else:
            raise ValueError(f"Invalid tokenizer mode: {self.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenizer(x)


class Detokenizer(nn.Module):
    """
    Base class for detokenizers.

    Parameters
    ----------
    patch_size : tuple
        The size of the incoming patches (time, height, width).
    dim_embed : int
        The dimension of the embedding.
    out_channels : int
        The number of channels in the output image.
    mode : str
        The mode of the detokenizer. Can be "linear" or "non_linear".
        Non-linear uses two 3D convolutions with GELU and instance normalization.
    """

    def __init__(
        self,
        patch_size: tuple,
        dim_embed: int,
        out_channels: int,
        mode: str,
        decoder_channels: Optional[list] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.dim_embed = dim_embed
        self.mode = mode

        if self.mode == "linear":
            self.detokenizer = LinearDetokenizer(
                patch_size=patch_size,
                out_channels=out_channels,
                dim_embed=dim_embed,
            )
        elif self.mode == "non_linear":
            self.detokenizer = NonlinearDetokenizer(
                dim_embed=dim_embed,
                out_channels=out_channels,
                patch_size=patch_size,
            )
        elif self.mode == "autoencoder":
            self.detokenizer = Decoder(
                channels=[dim_embed, *decoder_channels, out_channels],
                patch_size=patch_size,
            )
        else:
            raise ValueError(f"Invalid tokenizer mode: {self.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.detokenizer(x)


class LinearTokenizer(nn.Module):
    """
    Use a linear layer to project the input tensor into patches.

    Parameters
    ----------
    patch_size : tuple
        The size of the patches to split the image into (time, height, width).

    in_channels : int
        The number of channels in the input image.

    dim_embed : int
        The dimension of the embedding.
    """

    def __init__(self, patch_size: tuple, in_channels: int, dim_embed: int):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=dim_embed,
                kernel_size=patch_size,
                stride=patch_size,
                padding="valid",
            ),
            Rearrange("b c t h w -> b t h w c"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the input tensor into patches.
        """
        x = self.to_patch_embedding(x)
        return x


class LinearDetokenizer(nn.Module):
    """
    Converts the patches back into an image using linear projections.

    This is the inverse operation of the LinearTokenizer.

    Parameters
    ----------
    patch_size : tuple
        The size of the patches (time, height, width).

    out_channels : int
        The number of channels in the output image.

    dim_embed : int
        The dimension of the embedding.
    """

    def __init__(self, patch_size: tuple, out_channels: int, dim_embed: int):
        super().__init__()

        self.from_patch_embedding = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),
            nn.ConvTranspose3d(
                in_channels=dim_embed,
                out_channels=out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=False,
            ),
            Rearrange("b c t h w -> b t h w c"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert the patches back into an image.
        """
        x = self.from_patch_embedding(x)
        return x


class NonlinearTokenizer(nn.Module):
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
    patch_size : tuple (time, height, width)
        The desired final patch size. This will be split into two conv layers.
        Must be a power of two.
    """

    def __init__(
        self,
        in_channels: int,
        dim_embed: int,
        patch_size: tuple,
    ):
        super().__init__()

        # Calculate conv sizes from patch size
        t1, t2 = find_closest_factors_power_of_two(patch_size[0])
        h1, h2 = find_closest_factors_power_of_two(patch_size[1])
        w1, w2 = find_closest_factors_power_of_two(patch_size[2])

        conv1_size = (t1, h1, w1)
        conv2_size = (t2, h2, w2)

        tokenizer_net = [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=dim_embed // 2,
                kernel_size=conv1_size,
                stride=conv1_size,
                padding=0,
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
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm3d(
                num_features=dim_embed,
                affine=True,
            ),
            nn.GELU(),
        ]

        self.tokenizer = nn.Sequential(*tokenizer_net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a 5D input tensor (batch_size, time, height, width, channels) to a 5D output tensor of images.
        """
        b, t, h, w, c = x.shape
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.tokenizer(x)
        x = rearrange(x, "b c t h w -> b t h w c")
        return x


class NonlinearDetokenizer(nn.Module):
    """
    Converts back spatio-temporal tokens into a physical tensor.

    Parameters
    ----------
    dim_embed : int
        Dimension of the incoming embedding.
    out_channels : int
        Number of channels in the output tensor (number of physics states).
    patch_size : tuple (time, height, width)
        The desired final patch size. This will be split into two conv layers.
    """

    def __init__(
        self,
        dim_embed: int,
        out_channels: int,
        patch_size: tuple,
    ):
        super().__init__()
        self.in_channels = dim_embed
        self.out_channels = out_channels

        # Calculate conv sizes from patch size
        t1, t2 = find_closest_factors_power_of_two(patch_size[0])
        h1, h2 = find_closest_factors_power_of_two(patch_size[1])
        w1, w2 = find_closest_factors_power_of_two(patch_size[2])

        conv1_size = (t1, h1, w1)
        conv2_size = (t2, h2, w2)

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
