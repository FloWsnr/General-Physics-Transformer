from pathlib import Path
import torch
from typing_extensions import Literal
import pandas as pd
import numpy as np

from gphyt.utils.plotting.base_plotter import BasePlotter
from gphyt.utils.plotting.base_plotter import (
    calculate_combined_stats_rollout,
    rollout_mean,
    rollout_mean_by_pattern,
)


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


def plot_rollout(base_dir):
    DATASETS = [
        [
            "cylinder_sym_flow_water",
            "cylinder_pipe_flow_water",
            "object_periodic_flow_water",
            "object_sym_flow_water",
            "object_sym_flow_air",
        ],
        ["rayleigh_benard", "rayleigh_benard_obstacle"],
        "twophase_flow",
        "shear_flow",
        "euler_multi_quadrants_periodicBC",
        ["heated_object_pipe_flow_air", "cooled_object_pipe_flow_air"],
        # "acoustic_scattering_inclusions",
    ]

    DS_NAMES = [
        "Obstacle flow",
        "Rayleigh-Bénard",
        "Twophase flow",
        "Shear flow",
        "Euler",
        "Thermal flow",
    ]

    x_ticks = [0, 25, 50]
    y_ticks = [0, 0.5, 1]
    plotter = LossVsTimePlotter(x_ticks=x_ticks, y_ticks=y_ticks, y_log=False)

    eval_dir = base_dir / "xl-main-03" / "eval" / "best_model"

    rollout_df = pd.read_csv(eval_dir / "rollout_losses.csv", header=[0, 1, 2])
    single_step_df = pd.read_csv(eval_dir / "single_step_losses.csv", header=[0, 1, 2])

    # Calculate combined means for different flow types

    combined_df = calculate_combined_stats_rollout(rollout_df, column_patterns=DATASETS)
    data_mean = rollout_mean_by_pattern(combined_df)
    data_mean = data_mean[:]["mean"].values

    for i, ds_name in enumerate(DS_NAMES):
        color = next(plotter.color_cycler)

        x_data = np.arange(data_mean.shape[0])
        y_data = data_mean[:, i]
        plotter.plot_data(
            x_data=x_data,
            y_data=y_data,
            label=ds_name,
            symbolstyle="None",
            color=color,
        )
        plotter.legend()
        plotter.save_figure(base_dir / "plots/rollout.png")
        plotter.save_figure(base_dir / "plots/rollout.svg")


def plot_rollout_generalization(base_dir):
    DATASETS = [
        "euler_multi_quadrants_openBC",
        "open_obj_water",
        "supersonic_flow",
        "turbulent_radiative_layer_2D",
    ]

    DS_NAMES = [
        "Euler open BC",
        "Obstacle flow open BC",
        "Supersonic flow",
        "Turbulent radiative layer",
    ]

    x_ticks = [0, 25, 50]
    y_ticks = [0, 0.5, 1]
    plotter = LossVsTimePlotter(x_ticks=x_ticks, y_ticks=y_ticks, y_log=False)

    eval_dir = base_dir / "xl-main-03" / "eval" / "generalization"

    rollout_df = pd.read_csv(eval_dir / "rollout_losses.csv", header=[0, 1, 2])
    single_step_df = pd.read_csv(eval_dir / "single_step_losses.csv", header=[0, 1, 2])

    # Calculate combined means for different flow types

    combined_df = calculate_combined_stats_rollout(rollout_df, column_patterns=DATASETS)
    data_mean = rollout_mean_by_pattern(combined_df)
    data_mean = data_mean[:]["mean"].values

    for i, ds_name in enumerate(DS_NAMES):
        color = next(plotter.color_cycler)

        x_data = np.arange(data_mean.shape[0])
        y_data = data_mean[:, i]
        plotter.plot_data(
            x_data=x_data,
            y_data=y_data,
            label=ds_name,
            symbolstyle="None",
            color=color,
        )
        plotter.legend()
        plotter.save_figure(base_dir / "plots/rollout_gen.png")
        plotter.save_figure(base_dir / "plots/rollout_gen.svg")


if __name__ == "__main__":
    base_dir = Path("/hpcwork/rwth1802/coding/General-Physics-Transformer/results")

    plot_rollout(base_dir)
    plot_rollout_generalization(base_dir)
