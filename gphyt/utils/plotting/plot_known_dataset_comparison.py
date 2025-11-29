from typing import Literal
from pathlib import Path
import pandas as pd
import numpy as np

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


# Dataset groups for known physics
DATASET_GROUPS_KNOWN = {
    "incompressible_flow": [
        "cylinder_sym_flow_water",
        "cylinder_pipe_flow_water",
        "object_periodic_flow_water",
        "object_sym_flow_water",
        "object_sym_flow_air",
    ],
    "rayleigh_benard": ["rayleigh_benard", "rayleigh_benard_obstacle"],
    "twophase_flow": ["twophase_flow"],
    "shear_flow": ["shear_flow"],
    "euler_periodic": ["euler_multi_quadrants_periodicBC"],
    "heated_flow": ["heated_object_pipe_flow_air", "cooled_object_pipe_flow_air"],
}

# Display names for dataset groups
DATASET_GROUP_DISPLAY_NAMES = {
    "incompressible_flow": "Incompressible flow",
    "rayleigh_benard": "Rayleigh-Bénard",
    "twophase_flow": "Two-phase flow",
    "shear_flow": "Shear flow",
    "euler_periodic": "Euler periodic BC",
    "heated_flow": "Heated flow",
}

# Y-axis ticks for each dataset group
Y_TICKS_BY_DATASET_GROUP = {
    "incompressible_flow": [1e-4, 1e-2, 1e0],
    "rayleigh_benard": [1e-4, 1e-2, 1e0],
    "twophase_flow": [1e-4, 1e-2, 1e0],
    "shear_flow": [1e-4, 1e-2, 1e0],
    "euler_periodic": [1e-1, 1e1],
    "heated_flow": [1e-4, 1e-2, 1e0],
}


class LossVsTimePlotter(BasePlotter):
    def __init__(
        self,
        x_ticks: list[int],
        y_ticks: list[float],
        color: Literal["white", "black"] = "white",
        y_log: bool = False,
    ):
        super().__init__(color, figsize=(4.3, 4.3 / 2))

        self.setup_figure(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="Time steps",
            y_label="NMSE",
            y_log=y_log,
            minor_ticks=(False, True),
            padding_factor=(0.1, 0.2),
        )

    def plot(
        self,
        x_data: np.ndarray,
        mean_loss: np.ndarray,
        label: str,
        y_err: np.ndarray | None = None,
    ):
        color = next(self.color_cycler)
        symbol = next(self.symbol_cycler)

        self.plot_data(
            x_data=x_data,
            y_data=mean_loss,
            color=color,
            label=label,
            symbolstyle=symbol,
            markerstep=1,
        )
        if y_err is not None:
            self.plot_error_region(
                x_data=x_data,
                y_data=mean_loss,
                y_err=y_err,
                color=color,
                edgecolor=color,
            )


# Models to compare
RUNS_KNOWN = [
    ("m-main-03", "GPₕᵧT"),
    ("poseidon", "Poseidon"),
    ("dpot", "DPOT"),
    ("mpp", "MPP"),
    ("unet-m-04", "UNet"),
]


if __name__ == "__main__":
    base_dir = Path("/home/flwi01/coding/General-Physics-Transformer/results")
    horizons = [1, 4, 8, 12, 16, 20, 24]

    # Create one plot for each known dataset group
    for dataset_group_key, dataset_list in DATASET_GROUPS_KNOWN.items():
        print(f"\nProcessing dataset group: {dataset_group_key}")

        # Get y-axis ticks for this dataset group
        y_ticks = Y_TICKS_BY_DATASET_GROUP[dataset_group_key]

        # Create plotter for this dataset group
        plotter = LossVsTimePlotter(
            x_ticks=horizons,
            y_ticks=y_ticks,
            y_log=True,
        )

        # Process each model for this dataset group
        for run_name, display_name in RUNS_KNOWN:
            print(f"  Processing model: {run_name}")
            median_data = []
            percentile_25 = []
            percentile_75 = []
            time_steps = []

            for time_horizon in horizons:
                run_dir = base_dir / run_name / "eval/all_horizons"
                file_name = run_dir / f"nmse_losses_h{time_horizon}.csv"

                if not file_name.exists():
                    print(f"    Warning: File {file_name} does not exist. Skipping.")
                    continue

                # Load dataframe
                df_nmse = pd.read_csv(file_name)

                # Extract statistics for this specific dataset group
                stats_nmse = calculate_combined_stats(df_nmse, [dataset_list])

                # Get median and percentiles for this dataset group
                # The key in stats_nmse will be the string representation of the list
                # We need to find the right key
                if len(dataset_list) == 1:
                    # Single dataset
                    dataset_key = dataset_list[0]
                else:
                    # Group of datasets - stats will be under the list representation
                    # We need to use "OVERALL" or find the actual key
                    # Looking at calculate_combined_stats, it should create a row for the group
                    dataset_key = "OVERALL"

                median = stats_nmse.loc[dataset_key, "Combined Median"]
                q25 = stats_nmse.loc[dataset_key, "Combined 25th"]
                q75 = stats_nmse.loc[dataset_key, "Combined 75th"]

                median_data.append(median)
                percentile_25.append(q25)
                percentile_75.append(q75)
                time_steps.append(time_horizon)

            # Plot the data for this model
            if median_data:  # Only plot if we have data
                # Convert percentiles to error format (distance from median)
                median_array = np.array(median_data)
                p25_array = np.array(percentile_25)
                p75_array = np.array(percentile_75)

                # y_err should be [[lower_errors], [upper_errors]]
                y_err = np.array(
                    [
                        median_array
                        - p25_array,  # lower error (distance to 25th percentile)
                        p75_array
                        - median_array,  # upper error (distance to 75th percentile)
                    ]
                )

                plotter.plot(
                    x_data=np.array(time_steps),
                    mean_loss=median_array,
                    label=display_name,
                    # y_err=y_err,
                )

        # Add legend and save figure
        plotter.legend(loc="upper left")

        output_dir = base_dir / "01_new_plots/known_datasets"
        output_dir.mkdir(parents=True, exist_ok=True)

        plotter.save_figure(
            output_dir / f"known_{dataset_group_key}_model_comparison.png"
        )
        plotter.save_figure(
            output_dir / f"known_{dataset_group_key}_model_comparison.svg"
        )
        print(
            f"  Saved: {output_dir / f'known_{dataset_group_key}_model_comparison.png'}"
        )

    print("\nDone! All dataset comparison plots have been generated.")
