from pathlib import Path
import pandas as pd
import numpy as np

from gphyt.utils.plotting.plot_lossVsTime import LossVsTimePlotter
from gphyt.utils.plotting.base_plotter import (
    calculate_combined_stats,
    calculate_combined_stats_rollout,
    rollout_mean,
    rollout_median,
)


if __name__ == "__main__":
    base_dir = Path("/scratch/zsa8rk/logs")

    # Get Mean, Median and STD from loss dataframes
    RUNS = [
        "xl-main-4-1-a",
    ]

    for run in RUNS:
        eval_dir = base_dir / run / "eval"
        last_dir = sorted(eval_dir.iterdir())[-1]

        rollout_df = pd.read_csv(last_dir / "rollout_losses.csv", header=[0, 1, 2])
        single_step_df = pd.read_csv(
            last_dir / "single_step_losses.csv", header=[0, 1, 2]
        )

        # Calculate combined means for different flow types
        flow_patterns = [
            "cylinder_sym_flow_water",
            "cylinder_pipe_flow_water",
            "object_periodic_flow_water",
            "object_sym_flow_water",
            "object_sym_flow_air",
            "rayleigh_benard",
            "rayleigh_benard_obstacle",
            "twophase_flow",
            "shear_flow",
            "euler_multi_quadrants_periodicBC",
            "heated_object_pipe_flow_air",
            "cooled_object_pipe_flow_air",
            "acoustic_scattering_inclusions",
        ]
        combined_df = calculate_combined_stats_rollout(
            rollout_df, column_patterns=flow_patterns
        )

        data_mean = rollout_mean(combined_df)
        x_ticks = [0, 5, 10]
        y_ticks = [0, 0.1, 0.2]
        plotter = LossVsTimePlotter(x_ticks=x_ticks, y_ticks=y_ticks)
        plotter.plot(mean_loss=data_mean[:11, :], std_loss=None)
        # plotter.legend("Mean")
        plotter.save_figure(base_dir.parent / "plots/rollout/mean.png")

        data_median = rollout_median(combined_df)
        x_ticks = [0, 5, 10]
        y_ticks = [0, 0.01, 0.02]
        plotter = LossVsTimePlotter(x_ticks=x_ticks, y_ticks=y_ticks)
        plotter.plot(mean_loss=data_median[:11, :], std_loss=None)
        # plotter.legend("Median")
        plotter.save_figure(base_dir.parent / "plots/rollout/median.png")

        # for pattern in flow_patterns:
        #     data_mean = combined_df[pattern]["mean"].values
        #     data_std = combined_df[pattern]["std"].values

        #     # Plot the combined dfs over time
        #     x_ticks = [0, 25, 50]
        #     y_ticks = [0, 0.2]
        #     plotter = LossVsTimePlotter(x_ticks=x_ticks, y_ticks=y_ticks)
        #     plotter.plot(mean_loss=data_mean, std_loss=None)
        #     plotter.save_figure(base_dir.parent / f"plots/rollout/{pattern}.png")
        # # plotter.legend("cylinder_sym_flow_water")
