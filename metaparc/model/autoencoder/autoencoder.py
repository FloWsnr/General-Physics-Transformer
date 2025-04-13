import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


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


class Encoder(nn.Module):
    """
    Encoder module that downsamples an input tensor to a latent representation.

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


class Decoder(nn.Module):
    """
    Decoder module that upsamples a latent representation back to the original dimensions.

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


if __name__ == "__main__":
    channels = [128, 64, 32, 3]
    patch_size = (4, 16, 16)

    batch_size = 3
    time = 2
    height = 4
    width = 4

    decoder = Decoder(channels=channels, patch_size=patch_size)
    x = torch.randn(batch_size, time, height, width, channels[0])
    print(decoder(x).shape)
