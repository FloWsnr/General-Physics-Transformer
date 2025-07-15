import torch
import torch.nn as nn

from einops import rearrange

from gphyt.model.model_specs import ResNet_M, ResNet_S


class ResBlock(nn.Module):
    """
    Residual block using 2D convolutions.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_dim : int
        Number of hidden channels.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(hidden_dim, eps=1e-5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels, eps=1e-5)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-5),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, height, width).
        """
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Residual Network using 2D convolutions. Merges time and channel dimensions for input.

    Parameters
    ----------
    in_channels : int
        Number of input channels (before merging with time).
    out_channels : int
        Number of output channels (after restoring shape).
    hidden_dim : int
        Number of hidden channels.
    n_layers : int
        Number of residual blocks.
    n_time_steps : int
        Number of time steps.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        n_layers: int,
        n_time_steps: int,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels * n_time_steps, hidden_dim, kernel_size=3, padding=1
        )
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.conv_out = nn.Conv2d(
            hidden_dim, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, time, height, width, channels).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1, height, width, out_channels).
        """
        x = rearrange(x, "b t h w c -> b (t c) h w")
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(x)
        # Output shape: (b, out_channels, h, w)
        # Add time dimension back as 1
        x = rearrange(x, "b c h w -> b 1 h w c")
        return x


def get_model(model_config: ResNet_M | ResNet_S, n_time_steps: int = 4) -> ResNet:
    """
    Get a ResNet model from a configuration dataclass.

    Parameters
    ----------
    model_config : ResNet_M or ResNet_S
        Model configuration dataclass.

    n_time_steps : int
        Number of time steps.

    Returns
    -------
    ResNet
        Instantiated ResNet model.
    """
    if isinstance(model_config, ResNet_M):
        model_config = ResNet_M()
    elif isinstance(model_config, ResNet_S):
        model_config = ResNet_S()
    else:
        raise ValueError(f"Invalid model size: {model_config}")

    return ResNet(
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        hidden_dim=model_config.hidden_dim,
        n_layers=model_config.n_layers,
        n_time_steps=n_time_steps,
    )   
