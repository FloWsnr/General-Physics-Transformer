import torch
import torch.nn as nn
from typing import Literal, Union
from einops import rearrange

from gphyt.model.resnet import ResBlock
from gphyt.model.model_specs import DeepONet_S, DeepONet_M, DeepONet_L


class ResNetBranch(nn.Module):
    """
    ResNet-based branch network for DeepONet.

    Parameters
    ----------
    input_channels : int
        Number of input physics field channels
    n_steps_input : int
        Number of input time steps
    n_blocks : int
        Number of ResNet blocks
    hidden_channels : int
        Number of hidden channels in ResNet blocks
    output_dim : int
        Output dimension of the branch network
    """

    def __init__(
        self,
        input_channels: int,
        n_steps_input: int,
        n_blocks: int,
        hidden_channels: int,
        output_dim: int,
    ):
        super().__init__()

        # Initial convolution to process merged time-channel input
        self.conv_in = nn.Conv2d(
            input_channels * n_steps_input, hidden_channels, kernel_size=3, padding=1
        )

        # ResNet blocks
        self.blocks = nn.Sequential(
            *[
                ResBlock(hidden_channels, hidden_channels, hidden_channels)
                for _ in range(n_blocks)
            ]
        )

        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final linear layer to get desired output dimension
        self.fc_out = nn.Linear(hidden_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ResNet branch network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_steps, height, width, channels)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        # Merge time and channel dimensions
        x = rearrange(x, "b t h w c -> b (t c) h w")

        # Pass through ResNet
        x = self.conv_in(x)
        x = self.blocks(x)

        # Global pooling and flatten
        x = self.global_pool(x)  # (batch_size, hidden_channels, 1, 1)
        x = rearrange(x, "b c 1 1 -> b c")  # (batch_size, hidden_channels)

        # Final linear transformation
        x = self.fc_out(x)  # (batch_size, output_dim)

        return x


class ResNetTrunk(nn.Module):
    """
    ResNet-based trunk network for DeepONet.

    Parameters
    ----------
    n_blocks : int
        Number of ResNet blocks
    hidden_channels : int
        Number of hidden channels in ResNet blocks
    output_dim : int
        Output dimension of the trunk network
    """

    def __init__(
        self,
        n_blocks: int,
        hidden_channels: int,
        output_dim: int,
    ):
        super().__init__()
        # Initial convolution to process coordinate input
        self.conv_in = nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1)

        # ResNet blocks
        self.blocks = nn.Sequential(
            *[
                ResBlock(hidden_channels, hidden_channels, hidden_channels)
                for _ in range(n_blocks)
            ]
        )

        # Final convolution to get desired output channels
        self.conv_out = nn.Conv2d(hidden_channels, output_dim, kernel_size=3, padding=1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ResNet trunk network.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinate tensor of shape (height, width, 2)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (height*width, output_dim)
        """
        h, w, _ = coords.shape

        # Reshape coordinates to (1, 2, h, w) for conv input
        coords = coords.view(h, w, 2).permute(2, 0, 1).unsqueeze(0)  # (1, 2, h, w)

        # Pass through ResNet
        x = self.conv_in(coords)
        x = self.blocks(x)
        x = self.conv_out(x)  # (1, output_dim, h, w)

        # Reshape to (h*w, output_dim)
        x = rearrange(x, "1 c h w -> (h w) c")  # (h*w, output_dim)

        return x


class DeepONet(nn.Module):
    """
    DeepONet implementation for physics operator learning using ResNet architectures.

    Compatible with the physics dataset format from gphyt.data.phys_dataset.PhysicsDataset.
    Input format: (batch_size, n_steps, height, width, channels)
    Output format: (batch_size, 1, height, width, channels)

    Parameters
    ----------
    input_channels : int
        Number of input physics field channels
    branch_n_blocks : int
        Number of ResNet blocks in branch network
    branch_hidden_channels : int
        Number of hidden channels in branch ResNet blocks
    trunk_n_blocks : int
        Number of ResNet blocks in trunk network
    trunk_hidden_channels : int
        Number of hidden channels in trunk ResNet blocks
    latent_dim : int
        Latent dimension for branch-trunk interaction
    img_size : tuple[int, int]
        Spatial dimensions (height, width)
    n_steps_input : int
        Number of input time steps
    """

    def __init__(
        self,
        input_channels: int,
        branch_n_blocks: int = 3,
        branch_hidden_channels: int = 64,
        trunk_n_blocks: int = 3,
        trunk_hidden_channels: int = 64,
        latent_dim: int = 256,
        img_size: tuple[int, int] = (256, 128),
        n_steps_input: int = 4,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.img_size = img_size
        self.n_steps_input = n_steps_input
        self.latent_dim = latent_dim

        # Calculate spatial size
        self.spatial_size = img_size[0] * img_size[1]

        # Ensure latent_dim is divisible by input_channels for multi-channel outputs
        assert latent_dim % input_channels == 0, (
            f"Latent dim {latent_dim} must be divisible by input_channels {input_channels}"
        )

        # Branch network (ResNet-based)
        self.branch_net = ResNetBranch(
            input_channels=input_channels,
            n_steps_input=n_steps_input,
            n_blocks=branch_n_blocks,
            hidden_channels=branch_hidden_channels,
            output_dim=latent_dim,
        )

        # Trunk network (ResNet-based)
        self.trunk_net = ResNetTrunk(
            n_blocks=trunk_n_blocks,
            hidden_channels=trunk_hidden_channels,
            output_dim=latent_dim,
        )

        # Create coordinate grid
        self.register_buffer("coords", self._create_coord_grid())

    def _create_coord_grid(self) -> torch.Tensor:
        """Create normalized coordinate grid for trunk network."""
        h, w = self.img_size
        y_coords = torch.linspace(-1, 1, h).view(-1, 1).repeat(1, w)
        x_coords = torch.linspace(-1, 1, w).view(1, -1).repeat(h, 1)
        coords = torch.stack([x_coords, y_coords], dim=-1)  # (h, w, 2)
        return coords  # Keep as (h, w, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DeepONet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_steps, height, width, channels)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1, height, width, channels)
        """
        # shape
        batch_size, n_steps, h, w, c = x.shape

        # Branch network forward pass
        branch_out = self.branch_net(x)  # (batch_size, latent_dim)

        # Trunk network forward pass
        trunk_out = self.trunk_net(self.coords)  # (spatial_size, latent_dim)

        # DeepONet: G(u)(y) = sum_k b_k(u) * t_k(y)
        # where b_k are branch network outputs and t_k are trunk network outputs

        # For multiple output channels, reshape the networks appropriately
        # Reshape branch output to (batch_size, input_channels, latent_dim // input_channels)
        branch_per_channel = rearrange(
            branch_out, "b (c f) -> b c f", c=self.input_channels
        )

        # Reshape trunk output to (spatial_size, input_channels, latent_dim // input_channels)
        trunk_per_channel = rearrange(
            trunk_out, "s (c f) -> s c f", c=self.input_channels
        )

        # Expand dimensions for broadcasting
        branch_expanded = rearrange(
            branch_per_channel, "b c f -> b 1 c f"
        )  # (batch_size, 1, input_channels, features)
        trunk_expanded = rearrange(
            trunk_per_channel, "s c f -> 1 s c f"
        )  # (1, spatial_size, input_channels, features)

        # Element-wise multiplication and sum over feature dimension
        output = (branch_expanded * trunk_expanded).sum(
            dim=-1
        )  # (batch_size, spatial_size, input_channels)

        # Reshape to spatial format
        h, w = self.img_size
        output = rearrange(output, "b (h w) c -> b h w c", h=h, w=w)

        # Add time dimension and return in expected format
        output = rearrange(
            output, "b h w c -> b 1 h w c"
        )  # (batch_size, 1, h, w, channels)

        return output


def get_deeponet_model(
    config: Union[DeepONet_S, DeepONet_M, DeepONet_L],
    input_channels: int,
    img_size: tuple[int, int],
    n_steps_input: int,
) -> DeepONet:
    """
    Factory function to create DeepONet model from dataclass config.

    Parameters
    ----------
    config : Union[DeepONet_S, DeepONet_M, DeepONet_L]
        DeepONet configuration dataclass
    input_channels : int
        Number of input physics field channels
    img_size : tuple[int, int]
        Spatial dimensions (height, width)
    n_steps_input : int
        Number of input time steps

    Returns
    -------
    DeepONet
        Configured DeepONet model
    """
    branch_n_blocks = config.branch_n_blocks
    branch_hidden_channels = config.branch_hidden_channels
    trunk_n_blocks = config.trunk_n_blocks
    trunk_hidden_channels = config.trunk_hidden_channels
    latent_dim = config.latent_dim

    # Ensure latent_dim is divisible by input_channels
    latent_dim = (latent_dim // input_channels) * input_channels

    return DeepONet(
        input_channels=input_channels,
        branch_n_blocks=branch_n_blocks,
        branch_hidden_channels=branch_hidden_channels,
        trunk_n_blocks=trunk_n_blocks,
        trunk_hidden_channels=trunk_hidden_channels,
        latent_dim=latent_dim,
        img_size=img_size,
        n_steps_input=n_steps_input,
    )
