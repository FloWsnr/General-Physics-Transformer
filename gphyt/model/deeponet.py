import torch
import torch.nn as nn
from typing import Union
from einops import rearrange

from gphyt.model.unet import UNet
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
    hidden_dim : int
        Starting hidden dimension, doubled at each down block.
        Final dim after up blocks is also hidden_dim.
    n_down_blocks : int
        Number of down blocks (equals number of up blocks)
    """

    def __init__(
        self,
        input_channels: int,
        n_steps_input: int,
        hidden_dim: int,
        n_down_blocks: int,
    ):
        super().__init__()
        self.net = UNet(
            in_channels=input_channels,
            out_channels=hidden_dim,
            starting_hidden_dim=hidden_dim,
            n_down_blocks=n_down_blocks,
            n_time_steps=n_steps_input,
        )

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
            Output tensor of shape (batch_size, height, width, hidden_dim)
        """
        x = self.net(x)
        x = rearrange(x, "b 1 h w c-> b h w c")
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
            Output tensor of shape (height, width, hidden_dim)
        """
        h, w, _ = coords.shape
        coords_flat = rearrange(coords, "h w c -> (h w) c")

        # Pass through MLP
        x = self.mlp(coords_flat)  # (h*w, hidden_dim)

        # Reshape back to (h, w, hidden_dim)
        x = rearrange(x, "(h w) c -> h w c", h=h, w=w)
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

        # Ensure latent_dim is divisible by input_channels for multi-channel outputs
        assert latent_dim % input_channels == 0, (
            f"Latent dim {latent_dim} must be divisible by input_channels {input_channels}"
        )

        # Branch network (UNet-based)
        self.branch_net = UNetBranch(
            input_channels=input_channels,
            n_steps_input=n_steps_input,
            hidden_dim=latent_dim,
            n_down_blocks=branch_n_down_blocks,
        )

        # Trunk network (MLP-based)
        self.trunk_net = MLPTrunk(
            hidden_dim=latent_dim,
        )

        # Create coordinate grid
        self.register_buffer("coords", self._create_coord_grid())

        # Learnable bias parameter for each output channel
        self.b = nn.Parameter(torch.zeros(self.input_channels))

    def _create_coord_grid(self) -> torch.Tensor:
        """Create normalized coordinate grid for trunk network."""
        h, w = self.img_size
        y_coords = torch.linspace(-1, 1, h).view(-1, 1).repeat(1, w)
        x_coords = torch.linspace(-1, 1, w).view(1, -1).repeat(h, 1)
        coords = torch.stack([x_coords, y_coords], dim=-1)  # (h, w, 2)
        return coords

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
        branch_out = self.branch_net(x)  # (batch_size, h, w, latent_dim)
        # Trunk network forward pass
        trunk_out = self.trunk_net(self.coords)  # (h, w, latent_dim)

        # For multiple output channels, reshape the networks appropriately
        features_per_channel = self.latent_dim // self.input_channels
        trunk_per_channel = rearrange(
            trunk_out,
            "h w (c f) -> 1 h w c f",
            h=h,
            w=w,
            c=c,
            f=features_per_channel,
        )
        branch_per_channel = rearrange(
            branch_out,
            "b h w (c f) -> b h w c f",
            c=c,
            f=features_per_channel,
        )

        # DeepONet: G(u)(y) = sum_k b_k(u) * t_k(y)
        # where b_k are branch network outputs and t_k are trunk network outputs

        # Element-wise multiplication and sum over feature dimension
        output = (trunk_per_channel * branch_per_channel).sum(
            dim=-1
        )  # (batch_size, h, w, input_channels)

        # Add learnable bias term
        output = output + self.b  # (batch_size, h, w, input_channels)
        # Add time dimension back as 1
        output = rearrange(output, "b h w c -> b 1 h w c")
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
