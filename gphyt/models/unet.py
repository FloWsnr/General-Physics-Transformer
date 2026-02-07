import torch
import torch.nn as nn

from einops import rearrange

from gphyt.models.model_specs import UNet_M, UNet_S


class DownBlock(nn.Module):
    """
    UNet downsampling block with two convolutions and max pooling.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels, eps=1e-5)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the down block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Skip connection tensor and downsampled tensor.
        """
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.in2(x)
        skip = self.relu(x)
        x = self.maxpool(skip)
        return skip, x


class UpBlock(nn.Module):
    """
    UNet upsampling block with transposed convolution and two convolutions.

    Parameters
    ----------
    in_channels : int
        Number of input channels from previous layer.
    skip_channels : int
        Number of channels from skip connection.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv1 = nn.Conv2d(
            in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.in1 = nn.InstanceNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels, eps=1e-5)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the up block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor from previous layer.
        skip : torch.Tensor
            Skip connection tensor from corresponding down block.

        Returns
        -------
        torch.Tensor
            Output tensor after upsampling and concatenation.
        """
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    """
    UNet architecture for image-to-image tasks. Merges time and channel dimensions for input.

    Parameters
    ----------
    in_channels : int
        Number of input channels (before merging with time).
    out_channels : int
        Number of output channels (after restoring shape).
    starting_hidden_dim : int
        Starting hidden dimension, doubled at each down block.
    n_down_blocks : int
        Number of down blocks (equals number of up blocks).
    n_time_steps : int
        Number of time steps.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        starting_hidden_dim: int,
        n_down_blocks: int,
        n_time_steps: int,
        integrate: bool = False,
    ):
        super().__init__()
        self.n_down_blocks = n_down_blocks
        self.integrate = integrate

        # Input convolution
        self.conv_in = nn.Conv2d(
            in_channels * n_time_steps, starting_hidden_dim, kernel_size=3, padding=1
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

        # Output convolution
        self.conv_out = nn.Conv2d(current_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, time, height, width, channels).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1, height, width, out_channels).
        """
        original_input = x.clone()  # only used if integrate is True
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

        # Output convolution
        x = self.conv_out(x)

        # Add time dimension back as 1
        x = rearrange(x, "b c h w -> b 1 h w c")
        if self.integrate:
            # Add the last timestep of the original input as residual connection
            residual = original_input[:, -1, ...].unsqueeze(1)
            x = x + residual
        return x


def get_model(model_config: dict) -> UNet:
    """
    Get a UNet model from a configuration dataclass.

    Parameters
    ----------
    model_config : dict

    Returns
    -------
    UNet
        Instantiated UNet model.
    """
    n_time_steps = model_config.get("n_time_steps", 4)
    integrate = model_config.get("integrate", False)
    model_size = model_config.get("model_size", "UNet_M")
    if model_size == "UNet_M":
        unet_config = UNet_M()
    elif model_size == "UNet_S":
        unet_config = UNet_S()
    else:
        raise ValueError(f"Invalid model size: {model_size}")

    return UNet(
        in_channels=unet_config.in_channels,
        out_channels=unet_config.out_channels,
        starting_hidden_dim=unet_config.starting_hidden_dim,
        n_down_blocks=unet_config.n_down_blocks,
        n_time_steps=n_time_steps,
        integrate=integrate,
    )
