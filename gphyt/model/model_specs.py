from dataclasses import dataclass


@dataclass
class GPT_S:
    hidden_dim: int = 192
    mlp_dim: int = 768
    num_heads: int = 3
    num_layers: int = 12
    conv_channels: tuple[int, ...] = (64,)


@dataclass
class GPT_M:
    hidden_dim: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    num_layers: int = 12
    conv_channels: tuple[int, ...] = (128, 256)


@dataclass
class GPT_L:
    hidden_dim: int = 1024
    mlp_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 24
    conv_channels: tuple[int, ...] = (128, 256, 512)


@dataclass
class GPT_XL:
    hidden_dim: int = 1280
    mlp_dim: int = 5120
    num_heads: int = 16
    num_layers: int = 32
    conv_channels: tuple[int, ...] = (128, 256, 512)


@dataclass
class FNO_M:
    in_channels: int = 5
    out_channels: int = 5
    hidden_channels: int = 128
    n_layers: int = 4
    n_modes_height: int = 15
    n_modes_width: int = 15
    n_modes_time: int = 15


@dataclass
class FNO_S:
    in_channels: int = 5
    out_channels: int = 5
    hidden_channels: int = 64
    n_layers: int = 4
    n_modes_height: int = 10
    n_modes_width: int = 10
    n_modes_time: int = 10


@dataclass
class ResNet_M:
    in_channels: int = 5
    out_channels: int = 5
    hidden_dim: int = 1024
    n_layers: int = 6


@dataclass
class ResNet_S:
    in_channels: int = 5
    out_channels: int = 5
    hidden_dim: int = 256
    n_layers: int = 6


@dataclass
class DeepONet_S:
    branch_n_blocks: int = 3
    branch_hidden_channels: int = 64
    trunk_n_blocks: int = 3
    trunk_hidden_channels: int = 64
    latent_dim: int = 256


@dataclass
class DeepONet_M:
    branch_n_blocks: int = 4
    branch_hidden_channels: int = 128
    trunk_n_blocks: int = 4
    trunk_hidden_channels: int = 128
    latent_dim: int = 512


@dataclass
class DeepONet_L:
    branch_n_blocks: int = 6
    branch_hidden_channels: int = 256
    trunk_n_blocks: int = 6
    trunk_hidden_channels: int = 256
    latent_dim: int = 1024


@dataclass
class UNet_S:
    in_channels: int = 5
    out_channels: int = 5
    starting_hidden_dim: int = 32
    n_down_blocks: int = 3


@dataclass
class UNet_M:
    in_channels: int = 5
    out_channels: int = 5
    starting_hidden_dim: int = 64
    n_down_blocks: int = 4
