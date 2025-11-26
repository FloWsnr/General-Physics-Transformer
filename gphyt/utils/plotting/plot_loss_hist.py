from pathlib import Path

import pandas as pd
import numpy as np

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


RUNS = [
    # ("fno-m", "FNO-M"),
    # ("s-main-03", "GPₕᵧT-S"),
    ("m-main-03", "GPₕᵧT-M"),
    ("poseidon", "Poseidon"),
    ("dpot", "DPOT"),
    ("unet-m-04", "UNet-M"),
    # ("mpp", "MPP"),
    # ("l-main-05", "GPₕᵧT-L"),
    # ("xl-main-03", "GPₕᵧT-XL"),
]

DATASET_GROUPS = {
    "Incomp. NS": [
        "cylinder_sym_flow_water",
        "cylinder_pipe_flow_water",
        "object_periodic_flow_water",
        "object_sym_flow_water",
        "object_sym_flow_air",
    ],
    "Shear Flow": "shear_flow",
    "Flow + Heat": ["heated_object_pipe_flow_air", "cooled_object_pipe_flow_air"],
    "Two-Phase Flow": "twophase_flow",
    "Rayleigh-Bénard": ["rayleigh_benard", "rayleigh_benard_obstacle"],
    "Euler": "euler_multi_quadrants_periodicBC",
}

# For compatibility with calculate_combined_stats
DATASETS = list(DATASET_GROUPS.values())


class LossPlotter(BasePlotter):
    def __init__(self):
        super().__init__(figsize=(12, 4.3))
        self.setup_figure(
            x_ticks=[],  # Will be set dynamically
            y_ticks=[1e-4, 1e-3, 1e-2],
            x_label="",
            y_label="NMSE",
            y_log=True,
            padding_factor=(0.1, 0.5),
            minor_ticks=(False, True),
        )

    def plot_grouped_bars(self, data_by_model):
        """Plot dataset groups as bars grouped by model.

        Args:
            data_by_model: Dict mapping model names to list of losses per dataset group
        """
        n_models = len(RUNS)
        n_groups = len(DATASET_GROUPS)

        # Calculate positions
        bar_width = 0.8 / n_groups  # Total group width is 0.8
        group_positions = np.arange(n_models)

        # Plot bars for each dataset group
        for i, group_name in enumerate(DATASET_GROUPS.keys()):
            group_losses = []
            for run_name, _ in RUNS:
                group_losses.append(data_by_model[run_name][i])

            # Get color from the base plotter's color cycler
            color = next(self.color_cycler)

            positions = group_positions + (i - n_groups / 2) * bar_width + bar_width / 2
            self.ax.bar(
                positions, group_losses, bar_width, label=group_name, color=color
            )

        # Set x-ticks with model names
        model_names = [display_name for _, display_name in RUNS]
        self.ax.set_xticks(group_positions, model_names)
        self.ax.tick_params(axis="x", rotation=30, length=0)

        # Add legend
        self.ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)


if __name__ == "__main__":
    base_dir = Path("/hpcwork/rwth1802/coding/General-Physics-Transformer/results")
    plotter = LossPlotter()

    # Dictionary to store losses for each model and dataset group
    data_by_model = {}

    for run_name, display_name in RUNS:
        run_dir = base_dir / run_name / "eval/all_horizons"
        # Load dataframe
        df_nmse = pd.read_csv(run_dir / "nmse_losses_h1.csv")

        # Use the existing calculate_combined_stats function
        stats_nmse = calculate_combined_stats(df_nmse, DATASETS)

        # Extract mean for each dataset group (excluding the OVERALL row)
        group_losses = []
        for i in range(len(DATASET_GROUPS)):
            group_mean = stats_nmse.iloc[i]["Combined Median"]
            group_losses.append(group_mean)

        data_by_model[run_name] = group_losses

    # Plot the grouped bars
    plotter.plot_grouped_bars(data_by_model)

    plotter.save_figure(base_dir / "01_new_plots/model_comp.png")
