from typing import Literal
from pathlib import Path
import pandas as pd
import numpy as np

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


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
    ("unet-m-04", "UNet"),
]


if __name__ == "__main__":
    base_dir = Path("/home/flwi01/coding/General-Physics-Transformer/results")
    horizons = [1, 4, 8, 12, 16, 20, 24]

    # Define output directory
    output_dir = base_dir / "01_new_plots/novel_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined plot 1: Supersonic + Turbulent
    print("\nProcessing combined dataset: Supersonic + Turbulent")
    combined_datasets_1 = [["supersonic_flow", "turbulent_radiative_layer_2D"]]
    combined_name_1 = "supersonic_flow + turbulent_radiative_layer_2D"

    plotter_combined_1 = LossVsTimePlotter(
        x_ticks=horizons,
        y_ticks=[1e-1, 1e1],
        y_log=True,
    )

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

            df_nmse = pd.read_csv(file_name)
            stats_nmse = calculate_combined_stats(df_nmse, combined_datasets_1)

            median = stats_nmse.loc[combined_name_1, "Combined Median"]
            q25 = stats_nmse.loc[combined_name_1, "Combined 25th"]
            q75 = stats_nmse.loc[combined_name_1, "Combined 75th"]

            median_data.append(median)
            percentile_25.append(q25)
            percentile_75.append(q75)
            time_steps.append(time_horizon)

        if median_data:
            median_array = np.array(median_data)
            p25_array = np.array(percentile_25)
            p75_array = np.array(percentile_75)

            y_err = np.array(
                [
                    median_array - p25_array,
                    p75_array - median_array,
                ]
            )

            plotter_combined_1.plot(
                x_data=np.array(time_steps),
                mean_loss=median_array,
                label=display_name,
            )

    plotter_combined_1.legend(loc="upper left", columns=2)
    plotter_combined_1.save_figure(
        output_dir / "novel_supersonic_turbulent_combined.png"
    )
    plotter_combined_1.save_figure(
        output_dir / "novel_supersonic_turbulent_combined.svg"
    )
    print(f"  Saved: {output_dir / 'novel_supersonic_turbulent_combined.png'}")
    print(f"  Saved: {output_dir / 'novel_supersonic_turbulent_combined.svg'}")

    # Combined plot 2: Open obstacle + Euler open BC
    print("\nProcessing combined dataset: Open obstacle + Euler open BC")
    combined_datasets_2 = [["open_obj_water", "euler_multi_quadrants_openBC"]]
    combined_name_2 = "open_obj_water + euler_multi_quadrants_openBC"

    plotter_combined_2 = LossVsTimePlotter(
        x_ticks=horizons,
        y_ticks=[1e-1, 1e1],
        y_log=True,
    )

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

            df_nmse = pd.read_csv(file_name)
            stats_nmse = calculate_combined_stats(df_nmse, combined_datasets_2)

            median = stats_nmse.loc[combined_name_2, "Combined Median"]
            q25 = stats_nmse.loc[combined_name_2, "Combined 25th"]
            q75 = stats_nmse.loc[combined_name_2, "Combined 75th"]

            median_data.append(median)
            percentile_25.append(q25)
            percentile_75.append(q75)
            time_steps.append(time_horizon)

        if median_data:
            median_array = np.array(median_data)
            p25_array = np.array(percentile_25)
            p75_array = np.array(percentile_75)

            y_err = np.array(
                [
                    median_array - p25_array,
                    p75_array - median_array,
                ]
            )

            plotter_combined_2.plot(
                x_data=np.array(time_steps),
                mean_loss=median_array,
                label=display_name,
            )

    plotter_combined_2.legend(loc="upper left", columns=2)
    plotter_combined_2.save_figure(output_dir / "novel_open_obj_euler_combined.png")
    plotter_combined_2.save_figure(output_dir / "novel_open_obj_euler_combined.svg")
    print(f"  Saved: {output_dir / 'novel_open_obj_euler_combined.png'}")
    print(f"  Saved: {output_dir / 'novel_open_obj_euler_combined.svg'}")

    print("\nDone! All combined dataset plots have been generated.")
