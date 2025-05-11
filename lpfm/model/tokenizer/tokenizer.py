"""
Tokenizer module for the transformer model.
By: Florian Wiesner
Date: 2025-04-01
"""

from typing import Optional
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from lpfm.model.tokenizer.tokenizer_utils import _calculate_strides


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
    overlap : int, optional
        The number of pixels to overlap between patches for the linear tokenizer.
        Must be even number.
    """

    def __init__(
        self,
        patch_size: tuple,
        in_channels: int,
        dim_embed: int,
        mode: str,
        conv_net_channels: Optional[list] = None,
        overlap: int = 0,
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
                overlap=overlap,
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
    overlap : int, optional
        The number of pixels to overlap between patches for the linear detokenizer.
        Must be even number.
    img_size : tuple, optional
        The size of the input image. Needed for the squash_time option.
    """

    def __init__(
        self,
        patch_size: tuple,
        dim_embed: int,
        out_channels: int,
        mode: str,
        conv_net_channels: Optional[list] = None,
        overlap: int = 0,
        img_size: Optional[tuple] = None,
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
                overlap=overlap,
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
    Use a linear layer to project the input tensor into patches with overlap.

    Parameters
    ----------
    patch_size : tuple
        The size of the patches to split the image into (time, height, width).
    in_channels : int
        The number of channels in the input image.
    dim_embed : int
        The dimension of the embedding.
    overlap : int, optional
        Number of pixels to overlap between patches.
        Must be even number.
    """

    def __init__(
        self,
        patch_size: tuple,
        in_channels: int,
        dim_embed: int,
        overlap: int = 0,
    ):
        super().__init__()

        if overlap % 2 != 0:
            raise ValueError(f"Overlap must be an even number, got {overlap}")

        # Calculate kernel sizes for each dimension
        kernel_size = tuple(int(ps + overlap) for ps in patch_size)

        # Calculate padding to maintain output size
        padding = tuple(overlap // 2 for _ in patch_size)

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=dim_embed,
                kernel_size=kernel_size,
                stride=patch_size,
                padding=padding,
                padding_mode="zeros",
            ),
            Rearrange("b c t h w -> b t h w c"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the input tensor into overlapping patches.
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

    overlap : int, optional
        Number of pixels to overlap between patches.
        Must be even number.

    img_size : tuple, optional
        The size of the input image. Needed for the squash_time option.
    """

    def __init__(
        self,
        patch_size: tuple,
        out_channels: int,
        dim_embed: int,
        overlap: int = 0,
    ):
        super().__init__()

        if overlap % 2 != 0:
            raise ValueError(f"Overlap must be an even number, got {overlap}")

        # Stride
        stride = patch_size

        # Calculate kernel sizes for each dimension
        kernel_size = tuple(int(ps + overlap) for ps in patch_size)

        # Calculate padding to maintain output size
        padding = tuple(overlap // 2 for _ in patch_size)

        self.from_patch_embedding = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),
            nn.ConvTranspose3d(
                in_channels=dim_embed,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="zeros",
                bias=True,
            ),
            Rearrange("b c t h w -> b t h w c"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert the patches back into an image.
        """
        x = self.from_patch_embedding(x)
        return x


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
        for i in range(num_layers - 1):
            padding = [1, 1, 1]
            stride = (time_stride[i], height_stride[i], width_stride[i])
            kernel_size = [4, 4, 4]

            # if the stride is 1, we need to set the kernel size to 1 and the padding to 0
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

        # final layer
        stride = (time_stride[-1], height_stride[-1], width_stride[-1])
        kernel_size = [4, 4, 4]
        padding = [1, 1, 1]

        # if the stride is 1, we need to set the kernel size to 1 and the padding to 0
        for j in range(3):
            if stride[j] == 1:
                kernel_size[j] = 1
                padding[j] = 0

        final_layer = nn.Sequential(
            nn.Conv3d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

        self.encoder = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),  # Rearrange for Conv3d
            *modules,
            final_layer,
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

        modules = []
        for i in range(num_layers - 1):
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

        # final layer
        stride = (time_stride[-1], height_stride[-1], width_stride[-1])
        kernel_size = stride

        final_layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            ),
        )
        self.decoder = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),  # Rearrange for ConvTranspose3d
            *modules,
            final_layer,
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


class ConvNet2DTokenizer(nn.Module):
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
        for i in range(num_layers - 1):
            padding = [1, 1, 1]
            stride = (time_stride[i], height_stride[i], width_stride[i])
            kernel_size = [4, 4, 4]

            # if the stride is 1, we need to set the kernel size to 1 and the padding to 0
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

        # final layer
        stride = (time_stride[-1], height_stride[-1], width_stride[-1])
        kernel_size = [4, 4, 4]
        padding = [1, 1, 1]

        # if the stride is 1, we need to set the kernel size to 1 and the padding to 0
        for j in range(3):
            if stride[j] == 1:
                kernel_size[j] = 1
                padding[j] = 0

        final_layer = nn.Sequential(
            nn.Conv3d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

        self.encoder = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),  # Rearrange for Conv3d
            *modules,
            final_layer,
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


class ConvNet2DDetokenizer(nn.Module):
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

        modules = []
        for i in range(num_layers - 1):
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

        # final layer
        stride = (time_stride[-1], height_stride[-1], width_stride[-1])
        kernel_size = stride

        final_layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            ),
        )
        self.decoder = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),  # Rearrange for ConvTranspose3d
            *modules,
            final_layer,
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
