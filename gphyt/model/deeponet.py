import torch
import torch.nn as nn
from typing import Union, List
from einops import rearrange

from gphyt.model.model_specs import DeepONet_S, DeepONet_M


class MLPBranch(nn.Module):
    """
    MLP-based branch network for DeepONet using tanh activations.

    Parameters
    ----------
    input_dim : int
        Input dimension (n_steps * input_channels)
    hidden_dims : List[int]
        List of hidden dimensions for each MLP layer
    output_dim : int
        Output dimension (latent_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Add hidden layers with tanh activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP branch network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_steps, height, width, channels)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, height, width, output_dim)
        """
        batch_size, n_steps, h, w, c = x.shape

        # Flatten spatial and temporal dimensions for MLP processing
        x_flat = rearrange(x, "b t h w c -> (b h w) (t c)")
        output_flat = self.mlp(x_flat)

        # Reshape back to spatial format
        output = rearrange(output_flat, "(b h w) d -> b h w d", b=batch_size, h=h, w=w)

        return output


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
    DeepONet implementation for physics operator learning using MLP branch and MLP trunk.

    Compatible with the physics dataset format from gphyt.data.phys_dataset.PhysicsDataset.
    Input format: (batch_size, n_steps, height, width, channels)
    Output format: (batch_size, 1, height, width, channels)

    Parameters
    ----------
    input_channels : int
        Number of input physics field channels
    branch_hidden_dims : List[int]
        List of hidden dimensions for MLP branch layers
    latent_dim : int
        Latent dimension for branch-trunk interaction
        Same as the output dimension of the branch network
        and the hidden dimension of the trunk network
    img_size : tuple[int, int]
        Spatial dimensions (height, width)
    n_steps_input : int
        Number of input time steps
    """

    def __init__(
        self,
        input_channels: int,
        branch_hidden_dims: List[int] = [128, 128],
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

        # Branch network (MLP-based)
        input_dim = n_steps_input * input_channels
        self.branch_net = MLPBranch(
            input_dim=input_dim,
            hidden_dims=branch_hidden_dims,
            output_dim=latent_dim,
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
    branch_hidden_dims = config.branch_hidden_dims

    # Ensure latent_dim is divisible by input_channels
    latent_dim = (latent_dim // input_channels) * input_channels

    return DeepONet(
        input_channels=input_channels,
        branch_hidden_dims=list(branch_hidden_dims),
        latent_dim=latent_dim,
        img_size=img_size,
        n_steps_input=n_steps_input,
    )
