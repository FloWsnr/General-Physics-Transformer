import json
from pathlib import Path

import pandas as pd

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


RUNS = ["m-main-euler-01", "m-main-euler-rb", "m-main-03"]

DATASETS = [
    "euler_multi_quadrants_periodicBC",
]


class TransferPlotter(BasePlotter):
    def __init__(self):
        super().__init__(figsize=(4.3, 4.3 / 2))

        x_ticks = [1, 2, 7]
        y_ticks = [1e-2]
        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label=r"$\mathregular{N_{datasets}}$",
            y_label="MSE",
            x_log=False,
            y_log=True,
            padding_factor=(0.1, 0.22),
            minor_ticks=False,
        )


if __name__ == "__main__":
    base_dir = Path("General-Physics-Transformer/results")
    # RUNS = ["m-main-4-1"]

    plotter = TransferPlotter()

    losses = []
    steps = [1, 2, 7]
    for step, run in zip(steps, RUNS):
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
    plotter.save_figure(base_dir / "plots/transferLearning.png")
    plotter.save_figure(base_dir / "plots/transferLearning.svg")
