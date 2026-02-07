import torch
import torch.nn as nn

from neuralop.models import FNO as NeuralOpFNO
from einops import rearrange

from gphyt.models.model_specs import FNO_M, FNO_S


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

        self.fno = NeuralOpFNO(
            n_modes=(n_modes_time, n_modes_height, n_modes_width),
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


def get_model(model_config: dict):
    model_size = model_config.get("model_size", "FNO_M")
    if model_size == "FNO_M":
        fno_config = FNO_M()
    elif model_size == "FNO_S":
        fno_config = FNO_S()
    else:
        raise ValueError(f"Invalid model size: {model_size}")

    return FNO(
        in_channels=fno_config.in_channels,
        out_channels=fno_config.out_channels,
        hidden_channels=fno_config.hidden_channels,
        n_layers=fno_config.n_layers,
        n_modes_height=fno_config.n_modes_height,
        n_modes_width=fno_config.n_modes_width,
        n_modes_time=fno_config.n_modes_time,
    )
