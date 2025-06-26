import json
from pathlib import Path

import pandas as pd

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


RUNS = [
    "ti-main-4-1",
    "m-main-4-1",
    "l-main-4-1",
    "xl-main-4-1-a",
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
        super().__init__()

        x_ticks = [1e4, 1e5, 1e6]
        y_ticks = [1e-5, 1e-4, 1e-3]
        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="Number of Updates",
            y_label="Loss",
            x_log=True,
            y_log=True,
        )


if __name__ == "__main__":
    # RUNS = ["m-main-4-1"]

    base_dir = Path("/scratch/zsa8rk/logs")
    plotter = ScalingLawPlotter()
    for run in RUNS:
        run_dir = base_dir / run / "eval"

        num_updates = []
        num_samples = []
        losses = []

        for eval_checkpoint in sorted(run_dir.iterdir()):
            if not eval_checkpoint.is_dir():
                continue
            with open(eval_checkpoint / "checkpoint_info.json", "r") as f:
                checkpoint_info = json.load(f)
            batches = checkpoint_info["batches_trained"]
            samples = checkpoint_info["samples_trained"]
            num_updates.append(batches)
            num_samples.append(samples)

            # load df
            df = pd.read_csv(eval_checkpoint / "losses.csv", index_col=0)
            stats = calculate_combined_stats(df, DATASETS)
            loss = stats.loc["OVERALL", "Combined Mean"]
            losses.append(loss)
            # printing
            print(
                f"Checkpoint {eval_checkpoint.name}: N_updates: {batches} N_samples: {samples} Loss: {loss:.2e}"
            )
        # plot
        color = next(plotter.color_cycler)
        plotter.plot_data(num_updates, losses, color=color)

    plotter.save_figure(base_dir.parent / "plots/scaling_laws_mean.png")
