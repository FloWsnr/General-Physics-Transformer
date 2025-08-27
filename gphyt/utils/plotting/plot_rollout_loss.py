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
    base_dir = Path("/hpcwork/rwth1802/coding/General-Physics-Transformer/results")

    # Get Mean, Median and STD from loss dataframes
    RUNS = [
        "l-main-05",
    ]

    for run in RUNS:
        eval_dir = base_dir / run / "eval" / "best_model"

        rollout_df = pd.read_csv(eval_dir / "rollout_losses.csv", header=[0, 1, 2])
        single_step_df = pd.read_csv(
            eval_dir / "single_step_losses.csv", header=[0, 1, 2]
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
        x_ticks = [0, 15, 30]
        y_ticks = [0, 0.1, 1]
        plotter = LossVsTimePlotter(x_ticks=x_ticks, y_ticks=y_ticks)
        plotter.plot(mean_loss=data_mean[:, :], std_loss=None)
        # plotter.legend("Mean")
        plotter.save_figure(base_dir / "plots/rollout/mean.png")
