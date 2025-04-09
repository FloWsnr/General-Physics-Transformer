import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class Encoder(nn.Module):
    """
    Encoder module that compresses input data into a latent representation.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels in the latent space.
    num_layers : int
        Number of convolutional layers in the encoder.
    patch_size : tuple[int, int, int]
        Size of patches to encode (time, height, width).
    
    Notes
    -----
    The encoder progressively downsamples the input using convolutional layers
    to create a compressed latent representation. The patch_size determines
    the ratio between input and latent space dimensions.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        patch_size: tuple[int, int, int],
    ):
        super().__init__()
        # Calculate the stride for each layer based on patch size and number of layers
        # Ensure minimum stride of 2 to avoid issues with stride=1
        t_stride = max(2, int(patch_size[0] ** (1 / num_layers)))
        h_stride = max(2, int(patch_size[1] ** (1 / num_layers)))
        w_stride = max(2, int(patch_size[2] ** (1 / num_layers)))
        
        # Validate that the calculated strides will properly downsample to the target patch size
        if t_stride ** num_layers < patch_size[0] or h_stride ** num_layers < patch_size[1] or w_stride ** num_layers < patch_size[2]:
            raise ValueError(
                f"Calculated strides ({t_stride}, {h_stride}, {w_stride}) with {num_layers} layers "
                f"cannot achieve the target patch size {patch_size}. Consider reducing num_layers."
            )
        stride = (t_stride, h_stride, w_stride)
        kernel_size = stride  # Using same values for kernel size as stride
        
        # Calculate intermediate channel sizes with geometric progression
        channels = [in_channels]
        for i in range(num_layers):
            # Gradually increase channels from in_channels to out_channels
            next_channels = in_channels + int((out_channels - in_channels) * ((i + 1) / num_layers))
            channels.append(next_channels)
        
        # Build encoder layers
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv3d(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding="valid",
                    bias=False,
                )
            )
            # Don't add normalization and activation to the last layer
            if i < num_layers - 1:
                layers.extend([
                    nn.InstanceNorm3d(num_features=channels[i+1], affine=True),
                    nn.GELU(),
                ])
        self.encoder = nn.Sequential(
            Rearrange("b t h w c -> b c t h w"),  # Rearrange for Conv3d
            *layers,
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
            Encoded representation of shape (batch_size, encoded_time, encoded_height, encoded_width, out_channels)
        """
        return self.encoder(x)
