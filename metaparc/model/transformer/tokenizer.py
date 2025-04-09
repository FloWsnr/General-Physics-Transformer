"""
Tokenizer module for the transformer model.
By: Florian Wiesner
Date: 2025-04-01
"""

from typing import Optional
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


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
        The mode of the tokenizer. Can be "linear" or "conv_net".
        Non-linear uses two 3D convolutions with GELU and instance normalization.
    conv_net_channels : list, optional
        The hidden channels of the conv net.
    """

    def __init__(
        self,
        patch_size: tuple,
        in_channels: int,
        dim_embed: int,
        mode: str,
        conv_net_channels: Optional[list] = None,
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
        elif self.mode == "conv_net":
            self.tokenizer = ConvNetTokenizer(
                channels=[in_channels, *conv_net_channels, dim_embed],
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
        The mode of the detokenizer. Can be "linear" or "conv_net".
    conv_net_channels : list, optional
        The hidden channels of the conv net.
    """

    def __init__(
        self,
        patch_size: tuple,
        dim_embed: int,
        out_channels: int,
        mode: str,
        conv_net_channels: Optional[list] = None,
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
        elif self.mode == "conv_net":
            self.detokenizer = ConvNetDetokenizer(
                channels=[dim_embed, *conv_net_channels, out_channels],
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


def _calculate_strides(
    dimension_size: int,
    num_layers: int,
    default_stride: int = 2,
) -> list[int]:
    """
    Calculate the strides for each layer to achieve the desired downsampling.

    Parameters
    ----------
    dimension_size : int
        The size of the dimension to downsample (time, height, or width)
    num_layers : int
        The number of layers in the encoder
    default_stride : int
        The default stride to use if the dimension size is not divisible by the total downsampling factor

    Returns
    -------
    list[int]
        A list of stride values for each layer
    """
    # Initialize all strides to 2 (default downsampling factor)
    strides = [default_stride] * num_layers

    # Calculate the total downsampling factor with all strides=2
    total_downsampling = default_stride**num_layers

    # If we need more downsampling, increase some strides
    if total_downsampling < dimension_size:
        remaining_factor = dimension_size // total_downsampling

        # Find the number of additional powers of 2 needed
        additional_power = 0
        while 2**additional_power < remaining_factor:
            additional_power += 1

        # Distribute the additional downsampling across layers
        for i in range(additional_power):
            if i < num_layers:
                strides[i] *= 2

    # If we need less downsampling, decrease some strides
    elif total_downsampling > dimension_size:
        # Find the number of powers of 2 we need to remove
        excess_power = 0
        while total_downsampling // (2**excess_power) > dimension_size:
            excess_power += 1

        # Adjust strides to achieve the desired downsampling
        for i in range(excess_power):
            if i < num_layers:
                strides[num_layers - i - 1] = 1

    return strides


class ConvNetTokenizer(nn.Module):
    """
    ConvNet module that downsamples an input tensor to a latent representation.

    Parameters
    ----------
    channels : list
        List of channel dimensions for each layer. The first element is the input channels,
        and the last element is the output (latent) channels.
    patch_size : tuple
        Tuple of (time, height, width) indicating the total downsampling factor.
    """

    def __init__(
        self,
        channels: list[int],
        patch_size: tuple[int, int, int],
    ):
        super().__init__()

        num_layers = len(channels) - 1

        # Calculate strides based on patch size and number of layers
        time_stride = _calculate_strides(patch_size[0], num_layers)
        height_stride = _calculate_strides(patch_size[1], num_layers)
        width_stride = _calculate_strides(patch_size[2], num_layers)

        modules = []
        for i in range(num_layers):
            padding = [1, 1, 1]
            stride = (time_stride[i], height_stride[i], width_stride[i])
            kernel_size = [4, 4, 4]
            for j in range(3):
                if stride[j] == 1:
                    kernel_size[j] = 1
                    padding[j] = 0

            modules.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.InstanceNorm3d(
                        num_features=channels[i + 1],
                        affine=True,
                    ),
                    nn.GELU(),
                )
            )

        self.encoder = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),  # Rearrange for Conv3d
            *modules,
            Rearrange("b c t h w -> b t h w c"),  # Rearrange back to original format
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time, height, width, channels)

        Returns
        -------
        torch.Tensor
            Encoded representation of shape
            (batch_size, encoded_time, encoded_height, encoded_width, out_channels)
        """
        return self.encoder(x)


class ConvNetDetokenizer(nn.Module):
    """
    ConvNet module that upsamples a latent representation back to the original dimensions.

    Parameters
    ----------
    channels : list
        List of channel dimensions for each layer. The first element is the latent channels,
        and the last element is the output channels.
    patch_size : tuple
        Tuple of (time, height, width) indicating the total downsampling factor to reverse.
    """

    def __init__(
        self,
        channels: list,
        patch_size: tuple,
    ):
        super().__init__()

        # Calculate stride for each dimension and layer
        num_layers = len(channels) - 1

        # Calculate strides for each layer to achieve the desired upsampling
        time_ratio, height_ratio, width_ratio = patch_size

        # Calculate the stride for each layer
        time_stride = _calculate_strides(time_ratio, num_layers)
        height_stride = _calculate_strides(height_ratio, num_layers)
        width_stride = _calculate_strides(width_ratio, num_layers)

        # reverse the strides
        time_stride = time_stride[::-1]
        height_stride = height_stride[::-1]
        width_stride = width_stride[::-1]

        print(time_stride, height_stride, width_stride)

        modules = []
        for i in range(num_layers):
            stride = (time_stride[i], height_stride[i], width_stride[i])
            kernel_size = stride
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=0,
                    ),
                    nn.InstanceNorm3d(
                        num_features=channels[i + 1],
                        affine=True,
                    ),
                    nn.GELU(),
                )
            )

        self.decoder = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),  # Rearrange for ConvTranspose3d
            *modules,
            Rearrange("b c t h w -> b t h w c"),  # Rearrange back to original format
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape
            (batch_size, encoded_time, encoded_height, encoded_width, channels)

        Returns
        -------
        torch.Tensor
            Decoded representation of shape
            (batch_size, time, height, width, out_channels)
        """
        return self.decoder(x)
