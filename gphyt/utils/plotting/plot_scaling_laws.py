import json
from pathlib import Path

import pandas as pd

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


RUNS = [
    "s-main-03",
    "m-main-03",
    "l-main-05",
    "xl-main-03",
]

DATASETS = [
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


class ScalingLawPlotter(BasePlotter):
    def __init__(self):
        super().__init__(figsize=(4.3, 4.3))

        x_ticks = [(0, "9M"), (1, "112M"), (2, "380M"), (3, "780M")]
        y_ticks = [1e-4, 1e-3, 1e-2]
        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="Number of Parameters",
            y_label="MSE",
            x_log=False,
            y_log=True,
            padding_factor=(0.1, 0.1),
            minor_ticks=False,
        )


if __name__ == "__main__":
    # RUNS = ["m-main-4-1"]

    base_dir = Path("General-Physics-Transformer/results")
    plotter = ScalingLawPlotter()
    losses = []
    steps = [0, 1, 2, 3]
    for run in RUNS:
        run_dir = base_dir / run / "eval/best_model"
        # load df
        df = pd.read_csv(run_dir / "mse_losses.csv", index_col=0)
        stats = calculate_combined_stats(df, DATASETS)
        loss = stats.loc["OVERALL", "Combined Mean"]
        losses.append(loss)
        # plot

    plotter.plot_data(steps, losses)
    plotter.save_figure(base_dir / "plots/scaling_laws_mean.svg")
    plotter.save_figure(base_dir / "plots/scaling_laws_mean.png")
