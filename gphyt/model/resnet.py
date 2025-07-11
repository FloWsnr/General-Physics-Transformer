import torch
import torch.nn as nn

from einops import rearrange

from gphyt.model.model_specs import ResNet_M, ResNet_S


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(hidden_dim, eps=1e-5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels, eps=1e-5)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.InstanceNorm3d(out_channels, eps=1e-5),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(
        self, in_channels: int, out_channels: int, hidden_dim: int, n_layers: int
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.conv_out = nn.Conv3d(hidden_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(x)
        x = rearrange(x, "b c t h w -> b t h w c")
        x = x[:, -1, :, :, :].unsqueeze(1)
        return x


def get_model(model_config: ResNet_M | ResNet_S) -> ResNet:
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
    )
