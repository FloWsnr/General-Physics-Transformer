import json
from pathlib import Path

import pandas as pd

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


RUNS_PROMPT = [
    "m-main-1-1-01",
    "m-main-2-1-01",
    "m-main-03",
    "m-main-8-1-01",
]

RUNS_PATCH = [
    "m-main-03",
    "m-main-4-2-01",
    "m-main-4-4-01",
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
]


class PromptSizePlotter(BasePlotter):
    def __init__(self):
        super().__init__(figsize=(4.3, 4.3 / 2))

        x_ticks = [1, 2, 4, 8]
        y_ticks = [1e-3, 1e-2, 1e-1]
        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label=r"$\mathregular{N_{input}}$",
            y_label="MSE",
            x_log=False,
            y_log=True,
            padding_factor=(0.1, 0.22),
            minor_ticks=False,
        )


class PatchSizePlotter(BasePlotter):
    def __init__(self):
        super().__init__(figsize=(4.3, 4.3 / 2))

        x_ticks = [1, 2, 4]
        y_ticks = [1e-3, 1e-2, 1e-1]
        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="Temporal patch size",
            y_label="MSE",
            x_log=False,
            y_log=True,
            padding_factor=(0.1, 0.22),
            minor_ticks=False,
        )


if __name__ == "__main__":
    base_dir = Path("/hpcwork/rwth1802/coding/General-Physics-Transformer/results")
    # RUNS = ["m-main-4-1"]

    plotter = PromptSizePlotter()

    losses = []
    steps = [1, 2, 4, 8]
    for step, run in zip(steps, RUNS_PROMPT):
        print(f"Processing {run}")
        run_dir = base_dir / run / "eval" / "best_model"

        with open(run_dir / "checkpoint_info.json", "r") as f:
            checkpoint_info = json.load(f)
        # load df
        df = pd.read_csv(run_dir / "mse_losses.csv", index_col=0)
        stats = calculate_combined_stats(df, DATASETS)
        loss = stats.loc["OVERALL", "Combined Mean"]
        losses.append(loss)

    # plot
    plotter.plot_data(steps, losses)
    plotter.save_figure(base_dir / "plots/prompt_size.png")
    plotter.save_figure(base_dir / "plots/prompt_size.svg")

    plotter = PatchSizePlotter()

    losses = []
    steps = [1, 2, 4]
    for step, run in zip(steps, RUNS_PATCH):
        print(f"Processing {run}")
        run_dir = base_dir / run / "eval" / "best_model"

        with open(run_dir / "checkpoint_info.json", "r") as f:
            checkpoint_info = json.load(f)
        # load df
        df = pd.read_csv(run_dir / "mse_losses.csv", index_col=0)
        stats = calculate_combined_stats(df, DATASETS)
        loss = stats.loc["OVERALL", "Combined Mean"]
        losses.append(loss)

    # plot
    plotter.plot_data(steps, losses)
    plotter.save_figure(base_dir / "plots/patch_size.png")
    plotter.save_figure(base_dir / "plots/patch_size.svg")
