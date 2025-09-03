from pathlib import Path
import torch
from typing_extensions import Literal
import pandas as pd
import numpy as np

from gphyt.utils.plotting.base_plotter import BasePlotter
from gphyt.utils.plotting.base_plotter import (
    calculate_combined_stats_rollout,
    rollout_mean_by_pattern,
    rollout_median_by_pattern,
)


class LossVsTimePlotter(BasePlotter):
    def __init__(
        self,
        x_ticks: list[int],
        y_ticks: list[float],
        color: Literal["white", "black"] = "white",
        y_log: bool = False,
    ):
        super().__init__(color, figsize=(4.3, 2.15))

        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="Time steps",
            y_label="MSE",
            y_log=y_log,
            padding_factor=(0.1, 0.2),
        )


def plot_rollout(
    eval_dir: Path, datasets: list[str] | list[list[str]], names: list[str]
):
    x_ticks = [0, 25, 50]
    y_ticks = [0, 0.5, 1]
    plotter = LossVsTimePlotter(x_ticks=x_ticks, y_ticks=y_ticks, y_log=False)

    rollout_df = pd.read_csv(eval_dir / "rollout_mse_losses.csv", header=[0, 1, 2])

    # Calculate combined means for different flow types

    combined_df = calculate_combined_stats_rollout(rollout_df, column_patterns=datasets)
    data_mean = rollout_mean_by_pattern(combined_df)
    data_mean = data_mean[:]["mean"].values

    for i, ds_name in enumerate(names):
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
        plotter.save_figure(base_dir / f"plots/rollout_{eval_dir.name}.png")
        plotter.save_figure(base_dir / f"plots/rollout_{eval_dir.name}.svg")


if __name__ == "__main__":
    base_dir = Path("/hpcwork/rwth1802/coding/General-Physics-Transformer/results")
    DATASETS = [
        ["rayleigh_benard_dt_1", "rayleigh_benard_obstacle_dt_1"],
        [
            "cylinder_sym_flow_water_dt_1",
            "cylinder_pipe_flow_water_dt_1",
            "object_periodic_flow_water_dt_1",
            "object_sym_flow_water_dt_1",
            "object_sym_flow_air_dt_1",
        ],
        "twophase_flow_dt_1",
        ["heated_object_pipe_flow_air_dt_1", "cooled_object_pipe_flow_air_dt_1"],
        "shear_flow_dt_1",
        "euler_multi_quadrants_periodicBC_dt_1",
    ]

    DS_NAMES = [
        "Rayleigh-Bénard",
        "Obstacle flow",
        "Twophase flow",
        "Thermal flow",
        "Shear flow",
        "Euler",
    ]

    eval_dir = base_dir / "xl-main-03" / "eval" / "best_model"
    plot_rollout(eval_dir, DATASETS, DS_NAMES)

    ############################################################
    eval_dir = base_dir / "xl-main-03" / "eval" / "generalization"
    DATASETS = [
        "turbulent_radiative_layer_2D_dt_1",
        "euler_multi_quadrants_openBC_dt_1",
        "open_obj_water_dt_1",
        "supersonic_flow_dt_1",
    ]

    DS_NAMES = [
        "Turbulent layer",
        "Euler open",
        "Obs-flow open",
        "Supersonic",
    ]
    eval_dir = base_dir / "xl-main-03" / "eval" / "generalization"
    plot_rollout(eval_dir, DATASETS, DS_NAMES)
