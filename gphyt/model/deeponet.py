import torch
import torch.nn as nn
from typing import Union
from einops import rearrange

from gphyt.model.unet import DownBlock, UpBlock
from gphyt.model.model_specs import DeepONet_S, DeepONet_M


class UNetBranch(nn.Module):
    """
    UNet-based branch network for DeepONet.

    Parameters
    ----------
    input_channels : int
        Number of input physics field channels
    n_steps_input : int
        Number of input time steps
    starting_hidden_dim : int
        Starting hidden dimension, doubled at each down block
    n_down_blocks : int
        Number of down blocks (equals number of up blocks)
    output_dim : int
        Output dimension of the branch network
    """

    def __init__(
        self,
        input_channels: int,
        n_steps_input: int,
        starting_hidden_dim: int,
        n_down_blocks: int,
        output_dim: int,
    ):
        super().__init__()
        self.n_down_blocks = n_down_blocks

        # Input convolution
        self.conv_in = nn.Conv2d(
            input_channels * n_steps_input,
            starting_hidden_dim,
            kernel_size=3,
            padding=1,
        )

        # Down blocks
        self.down_blocks = nn.ModuleList()
        current_dim = starting_hidden_dim
        for i in range(n_down_blocks):
            next_dim = current_dim * 2
            self.down_blocks.append(DownBlock(current_dim, next_dim))
            current_dim = next_dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_dim, current_dim * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(current_dim * 2, eps=1e-5),
            nn.ReLU(),
            nn.Conv2d(current_dim * 2, current_dim * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(current_dim * 2, eps=1e-5),
            nn.ReLU(),
        )

        # Up blocks
        self.up_blocks = nn.ModuleList()
        current_dim = current_dim * 2  # After bottleneck
        for i in range(n_down_blocks):
            skip_dim = current_dim // 2  # Skip connection dimension
            next_dim = current_dim // 2
            self.up_blocks.append(UpBlock(current_dim, skip_dim, next_dim))
            current_dim = next_dim

        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final linear layer to get desired output dimension
        self.fc_out = nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of UNet branch network.

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

        # Input convolution
        x = self.conv_in(x)

        # Down path
        skip_connections = []
        for down_block in self.down_blocks:
            skip, x = down_block(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Up path
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 1)]  # Reverse order
            x = up_block(x, skip)

        # Global pooling and flatten
        x = self.global_pool(x)  # (batch_size, current_dim, 1, 1)
        x = rearrange(x, "b c 1 1 -> b c")  # (batch_size, current_dim)

        # Final linear transformation
        x = self.fc_out(x)  # (batch_size, output_dim)

        return x


class MLPTrunk(nn.Module):
    """
    Simple MLP-based trunk network for DeepONet.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the MLP
    output_dim : int
        Output dimension of the trunk network
    """

    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()

        # Build MLP layers
        layers = []

        # Input layer (2D coordinates -> hidden_dim)
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP trunk network.

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

        # Reshape coordinates to (h*w, 2)
        coords_flat = rearrange(coords, "h w c -> (h w) c")  # (h*w, 2)

        # Pass through MLP
        x = self.mlp(coords_flat)  # (h*w, output_dim)

        return x


class DeepONet(nn.Module):
    """
    DeepONet implementation for physics operator learning using UNet branch and MLP trunk.

    Compatible with the physics dataset format from gphyt.data.phys_dataset.PhysicsDataset.
    Input format: (batch_size, n_steps, height, width, channels)
    Output format: (batch_size, 1, height, width, channels)

    Parameters
    ----------
    input_channels : int
        Number of input physics field channels
    branch_n_down_blocks : int
        Number of down blocks in UNet branch
    latent_dim : int
        Latent dimension for branch-trunk interaction
        Same as the starting hidden dimension of the branch network
        and the hidden dimension of the trunk network
    img_size : tuple[int, int]
        Spatial dimensions (height, width)
    n_steps_input : int
        Number of input time steps
    """

    def __init__(
        self,
        input_channels: int,
        branch_n_down_blocks: int = 3,
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

        # Branch network (UNet-based)
        self.branch_net = UNetBranch(
            input_channels=input_channels,
            n_steps_input=n_steps_input,
            starting_hidden_dim=latent_dim,
            n_down_blocks=branch_n_down_blocks,
            output_dim=latent_dim,
        )

        # Trunk network (MLP-based)
        self.trunk_net = MLPTrunk(
            hidden_dim=latent_dim,
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


def get_model(
    config: Union[DeepONet_S, DeepONet_M],
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
    latent_dim = config.latent_dim

    # Ensure latent_dim is divisible by input_channels
    latent_dim = (latent_dim // input_channels) * input_channels

    return DeepONet(
        input_channels=input_channels,
        branch_n_down_blocks=config.branch_down_blocks,
        latent_dim=latent_dim,
        img_size=img_size,
        n_steps_input=n_steps_input,
    )
