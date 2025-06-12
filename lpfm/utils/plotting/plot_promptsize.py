import json
from pathlib import Path

import pandas as pd

from lpfm.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


RUNS_PROMPT = [
    "m-main-1-1",
    "m-main-2-1",
    "m-main-4-1",
    "m-main-8-1",
]

RUNS_PATCH = [
    "m-main-4-1",
    "m-main-4-2",
    "m-main-4-4",
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


class PromptSizePlotter(BasePlotter):
    def __init__(self):
        super().__init__()

        x_ticks = [1, 2, 4, 8]
        y_ticks = [1e-5, 1e-4, 1e-3]
        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="N input time steps",
            y_label="MSE",
            x_log=True,
            y_log=True,
        )


class PatchSizePlotter(BasePlotter):
    def __init__(self):
        super().__init__()

        x_ticks = [1, 2, 4]
        y_ticks = [1e-5, 1e-4, 1e-3]
        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="Temporal patch size",
            y_label="MSE",
            x_log=True,
            y_log=True,
        )


if __name__ == "__main__":
    # RUNS = ["m-main-4-1"]

    base_dir = Path("/scratch/zsa8rk/logs")
    plotter = PromptSizePlotter()

    losses = []
    steps = [1, 2, 4, 8]
    for step, run in zip(steps, RUNS_PROMPT):
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
    plotter.save_figure(base_dir.parent / "plots/prompt_size.png")

    plotter = PatchSizePlotter()

    losses = []
    steps = [1, 2, 4]
    for step, run in zip(steps, RUNS_PATCH):
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
    plotter.save_figure(base_dir.parent / "plots/patch_size.png")
