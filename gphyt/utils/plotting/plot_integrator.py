import json
from pathlib import Path

import pandas as pd

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


RUNS = [
    "ti-main-4-1-noparc",
    "ti-main-4-1",
    "ti-main-4-1-heun",
    "ti-main-4-1-rk4",
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


class IntegratorPlotter(BasePlotter):
    def __init__(self):
        super().__init__(figsize=(4.3, 4.3))

        x_ticks = [(0, "None"), (1, "Euler"), (2, "Heun"), (3, "RK4")]
        y_ticks = [1e-4, 1e-3, 1e-2]
        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="Integrator order",
            y_label="MSE",
            x_log=False,
            y_log=True,
            padding_factor=(0.1, 0.1),
            minor_ticks=False,
        )


if __name__ == "__main__":
    base_dir = Path("/scratch/zsa8rk/logs")
    # RUNS = ["m-main-4-1"]

    plotter = IntegratorPlotter()

    losses = []
    steps = [0, 1, 2, 3]
    for step, run in zip(steps, RUNS):
        print(f"Processing {run}")
        run_dir = base_dir / run / "eval"
        eval_dir = sorted(run_dir.iterdir())[-1]

        with open(eval_dir / "checkpoint_info.json", "r") as f:
            checkpoint_info = json.load(f)
        # load df
        df = pd.read_csv(eval_dir / "losses.csv", index_col=0)
        stats = calculate_combined_stats(df, DATASETS)
        loss = stats.loc["OVERALL", "Combined Median"]
        losses.append(loss)

    # plot
    plotter.plot_data(steps, losses)
    plotter.save_figure(base_dir.parent / "plots/integrators.svg")
    plotter.save_figure(base_dir.parent / "plots/integrators.png")
