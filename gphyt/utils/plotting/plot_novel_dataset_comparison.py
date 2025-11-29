from typing import Literal
from pathlib import Path
import pandas as pd
import numpy as np

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


# Dataset names for novel physics
DATASETS_NOVEL = [
    "euler_multi_quadrants_openBC",
    "open_obj_water",
    "supersonic_flow",
    "turbulent_radiative_layer_2D",
]

# Display names for datasets
DATASET_DISPLAY_NAMES = {
    "euler_multi_quadrants_openBC": "Euler open BC",
    "open_obj_water": "Open obstacle water",
    "supersonic_flow": "Supersonic flow",
    "turbulent_radiative_layer_2D": "Turbulent radiative layer",
}

# Y-axis ticks for each dataset
Y_TICKS_BY_DATASET = {
    "euler_multi_quadrants_openBC": [1e-2, 1e0],
    "open_obj_water": [1e-2, 1e0],
    "supersonic_flow": [1e-2, 1e0],
    "turbulent_radiative_layer_2D": [1e-2, 1e0],
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
RUNS_NOVEL = [
    ("m-main-03", "GPₕᵧT"),
    ("poseidon", "Poseidon"),
    ("dpot", "DPOT"),
    ("mpp", "MPP"),
]


if __name__ == "__main__":
    base_dir = Path("/home/flwi01/coding/General-Physics-Transformer/results")
    horizons = [1, 4, 8, 12, 16, 20, 24]

    # Define output directory
    output_dir = base_dir / "01_new_plots/novel_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create one plot for each novel dataset
    for dataset in DATASETS_NOVEL:
        print(f"\nProcessing dataset: {dataset}")

        # Get y-axis ticks for this dataset
        y_ticks = Y_TICKS_BY_DATASET[dataset]

        # Create plotter for this dataset
        plotter = LossVsTimePlotter(
            x_ticks=horizons,
            y_ticks=y_ticks,
            y_log=True,
        )

        # Process each model for this dataset
        for run_name, display_name in RUNS_NOVEL:
            print(f"  Processing model: {run_name}")
            median_data = []
            percentile_25 = []
            percentile_75 = []
            time_steps = []

            for time_horizon in horizons:
                run_dir = base_dir / run_name / "eval/all_horizons_novel"
                file_name = run_dir / f"nmse_losses_h{time_horizon}.csv"

                if not file_name.exists():
                    print(f"    Warning: File {file_name} does not exist. Skipping.")
                    continue

                # Load dataframe
                df_nmse = pd.read_csv(file_name)

                # Extract statistics for this specific dataset
                stats_nmse = calculate_combined_stats(df_nmse, [dataset])

                # Get median and percentiles for this dataset
                median = stats_nmse.loc[dataset, "Combined Median"]
                q25 = stats_nmse.loc[dataset, "Combined 25th"]
                q75 = stats_nmse.loc[dataset, "Combined 75th"]

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
        plotter.legend(loc="upper left", columns=2)

        plotter.save_figure(output_dir / f"novel_{dataset}_model_comparison.png")
        plotter.save_figure(output_dir / f"novel_{dataset}_model_comparison.svg")
        print(f"  Saved: {output_dir / f'novel_{dataset}_model_comparison.png'}")
        print(f"  Saved: {output_dir / f'novel_{dataset}_model_comparison.svg'}")

    print("\nDone! All individual dataset comparison plots have been generated.")
