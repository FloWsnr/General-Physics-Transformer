import torch
import torch.nn as nn
from typing import Literal


class DeepONet(nn.Module):
    """
    DeepONet implementation for physics operator learning.

    Compatible with the physics dataset format from gphyt.data.phys_dataset.PhysicsDataset.
    Input format: (batch_size, n_steps, height, width, channels)
    Output format: (batch_size, 1, height, width, channels)

    Parameters
    ----------
    input_channels : int
        Number of input physics field channels
    branch_layers : list[int]
        Layer sizes for the branch network
    trunk_layers : list[int]
        Layer sizes for the trunk network
    activation : Literal["relu", "tanh", "gelu"]
        Activation function to use
    img_size : tuple[int, int]
        Spatial dimensions (height, width)
    n_steps_input : int
        Number of input time steps (for branch network input size calculation)
    """

    def __init__(
        self,
        input_channels: int,
        branch_layers: list[int],
        trunk_layers: list[int],
        activation: Literal["relu"] = "relu",
        img_size: tuple[int, int] = (256, 128),
        n_steps_input: int = 4,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.img_size = img_size
        self.n_steps_input = n_steps_input

        # Calculate input/output dimensions
        self.spatial_size = img_size[0] * img_size[1]
        branch_input_size = n_steps_input * self.spatial_size * input_channels
        trunk_input_size = 2  # x, y coordinates

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Branch network (processes input functions)
        branch_sizes = [branch_input_size] + branch_layers
        self.branch_net = nn.ModuleList()
        for i in range(len(branch_sizes) - 1):
            self.branch_net.append(nn.Linear(branch_sizes[i], branch_sizes[i+1]))

        # Trunk network (processes coordinate locations)
        trunk_sizes = [trunk_input_size] + trunk_layers
        self.trunk_net = nn.ModuleList()
        for i in range(len(trunk_sizes) - 1):
            self.trunk_net.append(nn.Linear(trunk_sizes[i], trunk_sizes[i+1]))

        # Ensure branch and trunk networks have same output dimension
        assert branch_layers[-1] == trunk_layers[-1], "Branch and trunk networks must have same output dimension"

        # Ensure the output dimension is divisible by input_channels for multi-channel outputs
        assert branch_layers[-1] % input_channels == 0, f"Branch output dim {branch_layers[-1]} must be divisible by input_channels {input_channels}"

        # Create coordinate grid
        self.register_buffer("coords", self._create_coord_grid())

    def _create_coord_grid(self) -> torch.Tensor:
        """Create normalized coordinate grid for trunk network."""
        h, w = self.img_size
        y_coords = torch.linspace(-1, 1, h).view(-1, 1).repeat(1, w)
        x_coords = torch.linspace(-1, 1, w).view(1, -1).repeat(h, 1)
        coords = torch.stack([x_coords, y_coords], dim=-1)  # (h, w, 2)
        return coords.view(-1, 2)  # (h*w, 2)

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
        batch_size = x.shape[0]

        # Flatten input for branch network
        x_flat = x.reshape(batch_size, -1)  # (batch_size, n_steps * h * w * channels)

        # Branch network forward pass
        branch_out = x_flat
        for i, layer in enumerate(self.branch_net):
            branch_out = layer(branch_out)
            if i < len(self.branch_net) - 1:  # No activation after last layer
                branch_out = self.activation(branch_out)

        # Trunk network forward pass
        trunk_out = self.coords  # (h*w, 2)
        for i, layer in enumerate(self.trunk_net):
            trunk_out = layer(trunk_out)
            if i < len(self.trunk_net) - 1:  # No activation after last layer
                trunk_out = self.activation(trunk_out)

        # DeepONet: G(u)(y) = sum_k b_k(u) * t_k(y)
        # where b_k are branch network outputs and t_k are trunk network outputs

        # For multiple output channels, we need to reshape the networks appropriately
        # branch_out: (batch_size, branch_dim)
        # trunk_out: (spatial_size, trunk_dim)

        # Reshape branch output to (batch_size, input_channels, branch_dim // input_channels)
        branch_per_channel = branch_out.view(batch_size, self.input_channels, -1)

        # Reshape trunk output to (spatial_size, input_channels, trunk_dim // input_channels)
        trunk_per_channel = trunk_out.view(self.spatial_size, self.input_channels, -1)

        # Compute DeepONet combination for each channel
        # branch_per_channel: (batch_size, input_channels, features)
        # trunk_per_channel: (spatial_size, input_channels, features)

        # Expand dimensions for broadcasting
        branch_expanded = branch_per_channel.unsqueeze(1)  # (batch_size, 1, input_channels, features)
        trunk_expanded = trunk_per_channel.unsqueeze(0)    # (1, spatial_size, input_channels, features)

        # Element-wise multiplication and sum over feature dimension
        output = (branch_expanded * trunk_expanded).sum(dim=-1)  # (batch_size, spatial_size, input_channels)

        # Transpose to get channels last: (batch_size, spatial_size, input_channels)
        output = output.permute(0, 1, 2)  # Already in correct order

        # Reshape to spatial format
        h, w = self.img_size
        output = output.view(batch_size, h, w, self.input_channels)

        # Add time dimension and return in expected format
        output = output.unsqueeze(1)  # (batch_size, 1, h, w, channels)

        return output


def get_deeponet_model(model_config: dict) -> DeepONet:
    """
    Factory function to create DeepONet model from config.

    Parameters
    ----------
    model_config : dict
        Configuration dictionary with model parameters

    Returns
    -------
    DeepONet
        Configured DeepONet model
    """
    deeponet_config = model_config.get("deeponet", {})

    input_channels = model_config.get("transformer", {}).get("input_channels", 4)

    # Ensure default layer sizes are divisible by input_channels
    default_layer_size = ((256 // input_channels) * input_channels)

    return DeepONet(
        input_channels=input_channels,
        branch_layers=deeponet_config.get("branch_layers", [default_layer_size, default_layer_size, default_layer_size]),
        trunk_layers=deeponet_config.get("trunk_layers", [default_layer_size, default_layer_size, default_layer_size]),
        activation=deeponet_config.get("activation", "relu"),
        img_size=model_config.get("img_size", (64, 64)),
        n_steps_input=model_config.get("data", {}).get("n_steps_input", 1),
    )