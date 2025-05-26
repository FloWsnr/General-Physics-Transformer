from dataclasses import dataclass


@dataclass
class LPFM_Ti:
    hidden_dim: int = 192
    mlp_dim: int = 768
    num_heads: int = 3
    num_layers: int = 12
    conv_channels: tuple[int, int] = (64, 128)


@dataclass
class LPFM_S:
    hidden_dim: int = 384
    mlp_dim: int = 1536
    num_heads: int = 6
    num_layers: int = 12
    conv_channels: tuple[int, int] = (64, 128)


@dataclass
class LPFM_M:
    hidden_dim: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    num_layers: int = 12
    conv_channels: tuple[int, int] = (64, 128)


@dataclass
class LPFM_L:
    hidden_dim: int = 1024
    mlp_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 24
    conv_channels: tuple[int, int] = (64, 128)


@dataclass
class LPFM_XL:
    hidden_dim: int = 1280
    mlp_dim: int = 5120
    num_heads: int = 16
    num_layers: int = 32
    conv_channels: tuple[int, int] = (64, 128)
