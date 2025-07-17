import torch
import torch.nn as nn

from neuralop.models import FNO3d
from einops import rearrange

from gphyt.model.model_specs import FNO_M, FNO_S


class FNO(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        n_modes_height: int,
        n_modes_width: int,
        n_modes_time: int,
    ):
        super().__init__()

        self.fno = FNO3d(
            n_modes_height=n_modes_height,
            n_modes_width=n_modes_width,
            n_modes_depth=n_modes_time,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rearrange to (batch, channels, time, height, width) for FNO convs
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.fno(x)
        # rearrange back to (batch, time, height, width, channels)
        x = rearrange(x, "b c t h w -> b t h w c")
        # return only the last time step
        x = x[:, -1, :, :, :].unsqueeze(1)
        return x


def get_model(model_config: FNO_M | FNO_S):
    if isinstance(model_config, FNO_M):
        model_config = FNO_M()
    elif isinstance(model_config, FNO_S):
        model_config = FNO_S()
    else:
        raise ValueError(f"Invalid model size: {model_config}")

    return FNO(
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        hidden_channels=model_config.hidden_channels,
        n_layers=model_config.n_layers,
        n_modes_height=model_config.n_modes_height,
        n_modes_width=model_config.n_modes_width,
        n_modes_time=model_config.n_modes_time,
    )
