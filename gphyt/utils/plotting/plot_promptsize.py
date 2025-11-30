from typing import Literal
from pathlib import Path
import pandas as pd
import torch
import numpy as np


from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats

# Datasets for known physics
DATASETS_KNOWN = [
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

# Datasets for novel physics
DATASETS_NOVEL = [
    "euler_multi_quadrants_openBC",
    "open_obj_water",
    "supersonic_flow",
    "turbulent_radiative_layer_2D",
]


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
        mean_loss: torch.Tensor | np.ndarray,
        label: str,
        y_err: torch.Tensor | np.ndarray | None = None,
    ):
        if isinstance(mean_loss, torch.Tensor):
            mean_loss = mean_loss.cpu().numpy()
        if isinstance(y_err, torch.Tensor):
            y_err = y_err.cpu().numpy()

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


# Model sizes for comparison
MODEL_SIZES = [
    ("m-main-1-1-01", "1/1"),
    ("m-main-2-1-01", "2/1"),
    ("m-main-03", "4/1"),
    ("m-main-8-1-01", "8/1"),
    # ("m-main-4-2-01", "4/2"),
    # ("m-main-4-4-01", "4/4"),
]

if __name__ == "__main__":
    base_dir = Path("/home/flwi01/coding/General-Physics-Transformer/results")
    horizons = [1, 4, 8]

    # Create plotters for known physics
    plotter_known_mean = LossVsTimePlotter(
        x_ticks=horizons,
        y_ticks=[1e-3, 1e-2, 1e-1],
        y_log=True,
    )
    plotter_known_median = LossVsTimePlotter(
        x_ticks=horizons,
        y_ticks=[1e-3, 1e-2, 1e-1],
        y_log=True,
    )

    # Create plotters for novel physics
    plotter_novel_mean = LossVsTimePlotter(
        x_ticks=horizons,
        y_ticks=[1e-1, 1e0, 1e1],
        y_log=True,
    )
    plotter_novel_median = LossVsTimePlotter(
        x_ticks=horizons,
        y_ticks=[1e-1, 1e0, 1e1],
        y_log=True,
    )

    # Process known physics data
    print("Processing known physics data...")
    for run_name, display_name in MODEL_SIZES:
        print(f"  Processing {run_name}...")
        mse_data = []
        median_data = []
        std_data = []
        time_steps = []
        percentiles = []

        for time_horizon in horizons:
            run_dir = base_dir / run_name / "eval/all_horizons"
            file_name = run_dir / f"nmse_losses_h{time_horizon}.csv"
            if not file_name.exists():
                print(f"    Warning: File {file_name} does not exist. Skipping.")
                continue
            # load df
            df_nmse = pd.read_csv(file_name)
            stats_nmse = calculate_combined_stats(df_nmse, DATASETS_KNOWN)
            mse = stats_nmse.loc["OVERALL", "Combined Mean"]
            median = stats_nmse.loc["OVERALL", "Combined Median"]
            std = stats_nmse.loc["OVERALL", "Combined Std"]
            q25 = stats_nmse.loc["OVERALL", "Combined 25th"]
            q75 = stats_nmse.loc["OVERALL", "Combined 75th"]

            mse_data.append(mse)
            median_data.append(median)
            std_data.append(std)
            time_steps.append(time_horizon)
            percentiles.append((q25, q75))

        plotter_known_mean.plot(
            x_data=np.array(time_steps),
            mean_loss=np.array(mse_data),
            label=display_name,
            # y_err=np.array(std_data),
        )
        plotter_known_median.plot(
            x_data=np.array(time_steps),
            mean_loss=np.array(median_data),
            label=display_name,
            # y_err=np.array(percentiles).T,
        )

    # Process novel physics data
    print("\nProcessing novel physics data...")
    for run_name, display_name in MODEL_SIZES:
        print(f"  Processing {run_name}...")
        mse_data = []
        median_data = []
        std_data = []
        time_steps = []
        for time_horizon in horizons:
            run_dir = base_dir / run_name / "eval/all_horizons_novel"
            file_name = run_dir / f"nmse_losses_h{time_horizon}.csv"
            if not file_name.exists():
                print(f"    Warning: File {file_name} does not exist. Skipping.")
                continue
            # load df
            df_nmse = pd.read_csv(file_name)
            stats_nmse = calculate_combined_stats(df_nmse, DATASETS_NOVEL)
            mse = stats_nmse.loc["OVERALL", "Combined Mean"]
            median = stats_nmse.loc["OVERALL", "Combined Median"]
            std = stats_nmse.loc["OVERALL", "Combined Std"]

            mse_data.append(mse)
            median_data.append(median)
            std_data.append(std)
            time_steps.append(time_horizon)

        plotter_novel_mean.plot(
            x_data=np.array(time_steps),
            mean_loss=np.array(mse_data),
            # std_loss=np.array(std_data),
            label=display_name,
        )
        plotter_novel_median.plot(
            x_data=np.array(time_steps),
            mean_loss=np.array(median_data),
            # std_loss=np.array(std_data),
            label=display_name,
        )

    # Add legends
    plotter_known_mean.legend(loc="upper left")
    plotter_known_median.legend(loc="upper left")
    plotter_novel_mean.legend(loc="upper left")
    plotter_novel_median.legend(loc="upper left")

    # Save all figures
    print("\nSaving figures...")
    output_dir = base_dir / "01_new_plots/promptsize"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save known physics mean plot
    plotter_known_mean.save_figure(output_dir / "model_promptsize_rollout_mean.png")
    plotter_known_mean.save_figure(output_dir / "model_promptsize_rollout_mean.svg")

    # Save known physics median plot
    plotter_known_median.save_figure(output_dir / "model_promptsize_rollout_median.png")
    plotter_known_median.save_figure(output_dir / "model_promptsize_rollout_median.svg")

    # Save novel physics mean plot
    plotter_novel_mean.save_figure(
        output_dir / "model_promptsize_novel_rollout_mean.png"
    )
    plotter_novel_mean.save_figure(
        output_dir / "model_promptsize_novel_rollout_mean.svg"
    )
    # Save novel physics median plot
    plotter_novel_median.save_figure(
        output_dir / "model_promptsize_novel_rollout_median.png"
    )
    plotter_novel_median.save_figure(
        output_dir / "model_promptsize_novel_rollout_median.svg"
    )

    print("Done! All plots have been generated in both PNG and SVG formats.")
