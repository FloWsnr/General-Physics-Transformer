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


def get_model(model_config: dict):
    """
    Returns the correct model
    """
    model_type = model_config.get("architecture")
    if model_type == "gphyt":
        from gphyt.model.transformer.model import get_model as get_gpt_model

        model = get_gpt_model(model_config)
    elif model_type == "unet":
        from gphyt.model.unet import get_model as get_unet_model

        model = get_unet_model(model_config)
    elif model_type == "fno":
        from gphyt.model.fno import get_model as get_fno_model

        model = get_fno_model(model_config)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return model
