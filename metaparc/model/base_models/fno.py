from neuralop.models import FNO


def get_fno(
    n_modes: int,
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    n_layers: int,
) -> FNO:
    return FNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        projection_channel_ratio=2,
    )
