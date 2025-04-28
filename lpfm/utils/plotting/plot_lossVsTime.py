from lpfm.utils.plotting.base_plotter import BasePlotter
from typing import Literal
import torch
import numpy as np


class LossVsTimePlotter(BasePlotter):
    def __init__(self, color: Literal["white", "black"] = "white"):
        super().__init__(color)

        self.setup_figure(
            x_ticks=[0, 250, 500],
            y_ticks=[0, 0.005, 0.01],
            x_label="Time",
            y_label="Loss",
        )

    def plot(self, mean_loss: torch.Tensor, std_loss: torch.Tensor):
        mean_loss = mean_loss.cpu().numpy()
        std_loss = std_loss.cpu().numpy()
        for i, (channel, params) in enumerate(self.field_params.items()):
            color = params["color"]
            symbol = params["symbol"]

            time = np.arange(mean_loss.shape[0])
            mean_loss_channel = mean_loss[:, i]
            std_loss_channel = std_loss[:, i]

            # plot mean loss

            self.plot_data(
                x_data=time,
                y_data=mean_loss_channel,
                color=color,
                label=channel,
                symbolstyle=symbol,
                markerstep=50,
            )
            self.plot_error_region(
                x_data=time,
                y_data=mean_loss_channel,
                y_err=std_loss_channel,
                color=color,
            )
