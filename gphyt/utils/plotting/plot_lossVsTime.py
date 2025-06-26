from gphyt.utils.plotting.base_plotter import BasePlotter
from typing import Literal
import torch
import numpy as np


class LossVsTimePlotter(BasePlotter):
    def __init__(
        self,
        x_ticks: list[int],
        y_ticks: list[float],
        color: Literal["white", "black"] = "white",
        y_log: bool = False,
    ):
        super().__init__(color)

        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="Time steps",
            y_label="MSE",
            y_log=y_log,
        )

    def plot(
        self,
        mean_loss: torch.Tensor | np.ndarray,
        std_loss: torch.Tensor | np.ndarray | None = None,
    ):
        if isinstance(mean_loss, torch.Tensor):
            mean_loss = mean_loss.cpu().numpy()
        if isinstance(std_loss, torch.Tensor):
            std_loss = std_loss.cpu().numpy()

        for i, (channel, params) in enumerate(self.field_params.items()):
            color = params["color"]
            symbol = params["symbol"]

            time = np.arange(mean_loss.shape[0])
            mean_loss_channel = mean_loss[:, i]

            # plot mean loss

            self.plot_data(
                x_data=time,
                y_data=mean_loss_channel,
                color=color,
                label=channel,
                symbolstyle="None",
                markerstep=50,
            )
            if std_loss is not None:
                std_loss_channel = std_loss[:, i]
                self.plot_error_region(
                    x_data=time,
                    y_data=mean_loss_channel,
                    y_err=std_loss_channel,
                    color=color,
                )
