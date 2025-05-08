from typing import Literal
import wandb
from wandb.apis.public import Run

import dotenv
import pandas as pd

from lpfm.utils.plotting.base_plotter import BasePlotter


class WandbLoader:
    def __init__(self, run_ids: list[str]):
        self.api = wandb.Api()
        entity, project = "wsnr-florian", "Large-Physics-Foundation-Model"
        self.runs: list[Run] = self.api.runs(
            f"{entity}/{project}", {"$or": [{"name": run_id} for run_id in run_ids]}
        )
        print(f"Found {len(self.runs)} runs, {len(run_ids)} requested")


class LossPlotter(WandbLoader):
    def __init__(self, run_ids: list[str], color: Literal["white", "black"] = "white"):
        super().__init__(run_ids)

        keys = [
            "training-losses/NMSE",
            "training/total_samples_trained",
            "training/total_batches_trained",
        ]
        train_data = {}
        for run in self.runs:
            train_data[run.name] = run.history(
                keys=keys, samples=100_000_000, pandas=True
            )
        self.train_data = pd.concat(train_data, axis=1)
        self.train_data.columns = pd.MultiIndex.from_tuples(self.train_data.columns)

        keys = [
            "validation-summary/samples_trained",
            "validation-summary/total-NMSE",
        ]
        val_data = {}
        for run in self.runs:
            val_data[run.name] = run.history(
                keys=keys, samples=100_000_000, pandas=True
            )
        self.val_data = pd.concat(val_data, axis=1)
        self.val_data.columns = pd.MultiIndex.from_tuples(self.val_data.columns)

        self.color = color

    def _running_avg(
        self, data: pd.DataFrame, window: int = 1000
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute running mean and standard deviation of the data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to compute statistics for
        window : int, optional
            Window size for the running average, by default 1000

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing (mean, std) DataFrames
        """
        mean = data.rolling(window=window, min_periods=1).mean()
        std = data.rolling(window=window, min_periods=1).std()
        return mean, std

    def plot_loss(
        self,
        x_ticks: list[float],
        y_ticks: list[float],
        x_type: Literal["samples", "batches"] = "samples",
        data_type: Literal["training", "validation"] = "training",
        running_avg: bool = False,
        window: int = 1000,
    ):
        """Plot loss curves over samples or batches for training or validation data.

        Parameters
        ----------
        x_ticks : list[float]
            The ticks for the x-axis
        y_ticks : list[float]
            The ticks for the y-axis
        x_type : Literal["samples", "batches"]
            Whether to plot over samples or batches
        data_type : Literal["training", "validation"]
            Whether to plot training or validation data
        running_avg : bool
            Whether to plot the running average of the loss
        window : int
            Window size for running average computation

        Returns
        -------
        None
        """
        plotter = BasePlotter(
            color=self.color,
        )

        x_label = "Samples Trained" if x_type == "samples" else "Batches Trained"
        x_key = (
            "training/total_samples_trained"
            if x_type == "samples"
            else "training/total_batches_trained"
        )
        y_key = (
            "training-losses/NMSE"
            if data_type == "training"
            else "validation-summary/total-NMSE"
        )
        data = self.train_data if data_type == "training" else self.val_data

        plotter.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label=x_label,
            y_label="NMSE",
            x_log=False,
            y_log=True,
        )

        # plot the data
        for run_name in data.columns.get_level_values(0).unique():
            x_data = data[run_name][x_key]
            y_data = data[run_name][y_key]
            color = next(plotter.color_cycler)

            if running_avg:
                mean, std = self._running_avg(y_data, window=window)
                plotter.plot_data(
                    x_data, mean, label=run_name, symbolstyle="", color=color
                )
                plotter.plot_error_region(x_data, mean, std, color=color)
            else:
                plotter.plot_data(
                    x_data, y_data, label=run_name, symbolstyle="", color=color
                )

        plotter.legend(title="Models", loc="upper right")
        plotter.show_figure()


if __name__ == "__main__":
    dotenv.load_dotenv()
    plotter = LossPlotter(
        [
            "ti-cyl-sym-flow-0003",
            "ti-cyl-sym-flow-0003-deriv",
            "ti-parc-0001",
            "ti-cyl-sym-flow-deriv-parc",
        ],
        color="black",
    )
    x_ticks = [0.0001, 30e6]
    y_ticks = [0.00001, 0.001, 0.1]
    plotter.plot_loss(
        x_ticks,
        y_ticks,
        x_type="samples",
        data_type="training",
        running_avg=True,
        window=1000,
    )
    # plotter.plot_loss(x_type="batches", data_type="training")
