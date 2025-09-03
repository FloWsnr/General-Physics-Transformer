import json
from pathlib import Path

import pandas as pd
import numpy as np

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


RUNS = [
    ("fno-m", "FNO-M"),
    ("unet-m-04", "UNet-M"),
    ("s-main-03", "GPₕᵧT-S"),
    ("m-main-03", "GPₕᵧT-M"),
    ("l-main-05", "GPₕᵧT-L"),
    ("xl-main-03", "GPₕᵧT-XL"),
]

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


class LossPlotter(BasePlotter):
    def __init__(self):
        super().__init__(figsize=(4.3 * 2, 4.3))
        self.setup_figure(
            x_ticks=[
                (i, display_name) for i, (run_name, display_name) in enumerate(RUNS)
            ],
            y_ticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            x_label=None,
            y_label="Mean Squared Error",
            y_log=True,
            padding_factor=(0.1, 0.1),
            minor_ticks=(False, True),
        )

    def plot_grouped_bars(self, mse_data, rvmse_data):
        """Plot grouped bars for MSE and RVMSE losses."""
        x = np.arange(len(RUNS))  # the label locations
        width = 0.35  # the width of the bars

        # Plot bars
        rects1 = self.ax.bar(x - width / 2, mse_data, width, label="Average")
        rects2 = self.ax.bar(x + width / 2, rvmse_data, width, label="Median")

        # Set x-ticks with model names
        model_names = [display_name for _, display_name in RUNS]
        self.ax.set_xticks(x, model_names)
        self.ax.tick_params(axis="x", rotation=30, length=0)

        # Add legend
        self.ax.legend(loc="upper right")


if __name__ == "__main__":
    # RUNS = ["m-main-4-1"]

    base_dir = Path("/hpcwork/rwth1802/coding/General-Physics-Transformer/results")
    plotter = LossPlotter()

    mse_data = []
    rvmse_data = []
    median_data = []

    for run_name, display_name in RUNS:
        run_dir = base_dir / run_name / "eval/best_model"
        # load df
        df_mse = pd.read_csv(run_dir / "mse_losses.csv")
        stats_mse = calculate_combined_stats(df_mse, DATASETS)
        mse = stats_mse.loc["OVERALL", "Combined Mean"]
        mse_data.append(mse)

        df_mse = pd.read_csv(run_dir / "mse_losses.csv")
        stats_mse = calculate_combined_stats(df_mse, DATASETS)
        median = stats_mse.loc["OVERALL", "Combined Median"]
        median_data.append(median)

        # df_rvmse = pd.read_csv(run_dir / "rvmse_losses.csv", index_col=0)
        # stats_rvmse = calculate_combined_stats(df_rvmse, DATASETS)
        # rvmse = stats_rvmse.loc["OVERALL", "Combined Mean"]
        # rvmse_data.append(rvmse)

    # Plot the grouped bars
    plotter.plot_grouped_bars(mse_data, median_data)

    plotter.save_figure(base_dir / "plots/model_comp_median.svg")
    plotter.save_figure(base_dir / "plots/model_comp_median.png")
