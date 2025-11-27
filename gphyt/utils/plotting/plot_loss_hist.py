from pathlib import Path

import pandas as pd
import numpy as np

from gphyt.utils.plotting.base_plotter import BasePlotter, calculate_combined_stats


RUNS = [
    # ("fno-m", "FNO-M"),
    # ("s-main-03", "GPₕᵧT-S"),
    ("m-main-03", "GPₕᵧT"),
    ("poseidon", "Poseidon"),
    # ("dpot", "DPOT"),
    # ("mpp", "MPP"),
    # ("unet-m-04", "UNet"),
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

# Y-axis ticks for each horizon
Y_TICKS_BY_HORIZON = {
    1: [1e-4, 1e-3, 1e-2],
    4: [1e-3, 1e-2, 1e-1],
    8: [1e-2, 1e-1, 1e0],
    16: [1e-1, 1e0, 1e1],
    20: [1e-1, 1e0, 1e1],
    24: [1e-1, 1e0, 1e1],
}

BLACK_COLOR = "#292929"


class LossPlotter(BasePlotter):
    def __init__(self, y_ticks):
        super().__init__(figsize=(10, 4.3))
        self.setup_figure(
            x_ticks=[],
            y_ticks=y_ticks,
            x_label="",
            y_label="NMSE",
            y_log=True,
            padding_factor=(0.1, 0.5),
            minor_ticks=(False, True),
        )

    def plot_grouped_bars(
        self,
        data_by_model,
        overall_by_model,
        errors_by_model,
        overall_errors_by_model,
        group_by="model",
    ):
        """Plot bars grouped by model or dataset.

        Args:
            data_by_model: Dict mapping model names to list of losses per dataset group
            overall_by_model: Dict mapping model names to overall loss
            errors_by_model: Dict mapping model names to list of (lower_err, upper_err) tuples
            overall_errors_by_model: Dict mapping model names to (lower_err, upper_err) tuple
            group_by: "model" to group by model (datasets as colors), "dataset" to group by dataset (models as colors)
        """
        if group_by == "model":
            self._plot_grouped_by_model(
                data_by_model,
                overall_by_model,
                errors_by_model,
                overall_errors_by_model,
            )
        elif group_by == "dataset":
            self._plot_grouped_by_dataset(
                data_by_model,
                overall_by_model,
                errors_by_model,
                overall_errors_by_model,
            )
        else:
            raise ValueError(f"Invalid group_by: {group_by}")

    def _plot_grouped_by_model(
        self, data_by_model, overall_by_model, errors_by_model, overall_errors_by_model
    ):
        """Plot dataset groups as bars grouped by model."""
        n_models = len(RUNS)
        n_groups = len(DATASET_GROUPS) + 1  # +1 for OVERALL

        bar_width = 0.8 / n_groups
        group_positions = np.arange(n_models)

        # Plot OVERALL bar first
        overall_losses = [overall_by_model[run_name] for run_name, _ in RUNS]
        overall_errors = [overall_errors_by_model[run_name] for run_name, _ in RUNS]
        yerr = np.array(
            [[err[0] for err in overall_errors], [err[1] for err in overall_errors]]
        )
        color = next(self.color_cycler)
        i = 0
        positions = group_positions + (i - n_groups / 2) * bar_width + bar_width / 2

        # Use lighter facecolor for black bars to make error bars visible
        facecolor = BLACK_COLOR if color == "black" or color == "#000000" else color
        edgecolor = color if color == "black" or color == "#000000" else None

        self.ax.bar(
            positions,
            overall_losses,
            bar_width,
            label="Overall",
            facecolor=facecolor,
            edgecolor=edgecolor,
            yerr=yerr,
            capsize=2,
            error_kw={"linewidth": 1},
        )

        # Plot bars for each dataset group
        for i, group_name in enumerate(DATASET_GROUPS.keys(), start=1):
            group_losses = []
            group_errors = []
            for run_name, _ in RUNS:
                group_losses.append(data_by_model[run_name][i - 1])
                group_errors.append(errors_by_model[run_name][i - 1])

            yerr = np.array(
                [[err[0] for err in group_errors], [err[1] for err in group_errors]]
            )
            color = next(self.color_cycler)
            positions = group_positions + (i - n_groups / 2) * bar_width + bar_width / 2

            # Use lighter facecolor for black bars to make error bars visible
            facecolor = BLACK_COLOR if color == "black" or color == "#000000" else color
            edgecolor = color if color == "black" or color == "#000000" else None

            self.ax.bar(
                positions,
                group_losses,
                bar_width,
                label=group_name,
                facecolor=facecolor,
                edgecolor=edgecolor,
                yerr=yerr,
                capsize=2,
                error_kw={"linewidth": 1},
            )

        # Set x-ticks with model names (no tick marks)
        model_names = [display_name for _, display_name in RUNS]
        self.ax.set_xticks(group_positions, model_names)
        self.ax.tick_params(
            axis="x", which="both", rotation=30, bottom=False, top=False
        )

        # Add legend
        self.ax.legend(loc="upper left", fontsize=9, ncol=3, framealpha=0.8)

    def _plot_grouped_by_dataset(
        self, data_by_model, overall_by_model, errors_by_model, overall_errors_by_model
    ):
        """Plot models as bars grouped by dataset."""
        n_models = len(RUNS)
        n_groups = len(DATASET_GROUPS) + 1  # +1 for OVERALL

        bar_width = 0.8 / n_models
        group_positions = np.arange(n_groups)

        # Plot bars for each model
        for i, (run_name, display_name) in enumerate(RUNS):
            model_losses = [overall_by_model[run_name]] + data_by_model[run_name]
            model_errors = [overall_errors_by_model[run_name]] + errors_by_model[
                run_name
            ]

            yerr = np.array(
                [[err[0] for err in model_errors], [err[1] for err in model_errors]]
            )
            color = next(self.color_cycler)
            positions = group_positions + (i - n_models / 2) * bar_width + bar_width / 2

            # Use lighter facecolor for black bars to make error bars visible
            facecolor = BLACK_COLOR if color == "black" or color == "#000000" else color
            edgecolor = color if color == "black" or color == "#000000" else None

            self.ax.bar(
                positions,
                model_losses,
                bar_width,
                label=display_name,
                facecolor=facecolor,
                edgecolor=edgecolor,
                yerr=yerr,
                capsize=2,
                error_kw={"linewidth": 1},
            )

        # Set x-ticks with Overall first, then dataset group names (no tick marks)
        dataset_names = ["Overall"] + list(DATASET_GROUPS.keys())
        self.ax.set_xticks(group_positions, dataset_names)
        self.ax.tick_params(
            axis="x", which="both", rotation=30, bottom=False, top=False
        )

        # Add legend
        self.ax.legend(loc="upper left", fontsize=9, ncol=5, framealpha=0.8)


def load_data_for_horizon(base_dir, horizon):
    """Load and process data for a given horizon."""
    data_by_model = {}
    overall_by_model = {}
    errors_by_model = {}
    overall_errors_by_model = {}

    for run_name, display_name in RUNS:
        run_dir = base_dir / run_name / "eval/all_horizons"
        df_nmse = pd.read_csv(run_dir / f"nmse_losses_h{horizon}.csv")

        stats_nmse = calculate_combined_stats(df_nmse, DATASETS)

        # Extract median and percentiles for each dataset group
        group_losses = []
        group_errors = []
        for i in range(len(DATASET_GROUPS)):
            median = stats_nmse.iloc[i]["Combined Median"]
            q25 = stats_nmse.iloc[i]["Combined 25th"]
            q75 = stats_nmse.iloc[i]["Combined 75th"]
            group_losses.append(median)
            group_errors.append((median - q25, q75 - median))

        data_by_model[run_name] = group_losses
        errors_by_model[run_name] = group_errors

        # Extract OVERALL loss and percentiles
        overall_median = stats_nmse.loc["OVERALL", "Combined Median"]
        overall_q25 = stats_nmse.loc["OVERALL", "Combined 25th"]
        overall_q75 = stats_nmse.loc["OVERALL", "Combined 75th"]
        overall_by_model[run_name] = overall_median
        overall_errors_by_model[run_name] = (
            overall_median - overall_q25,
            overall_q75 - overall_median,
        )

    return data_by_model, overall_by_model, errors_by_model, overall_errors_by_model


if __name__ == "__main__":
    base_dir = Path("/home/flwi01/coding/General-Physics-Transformer/results")
    output_dir = base_dir / "01_new_plots/histograms_02"
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = [1, 4, 8, 16, 20, 24]

    for horizon in horizons:
        print(f"Processing horizon {horizon}...")

        # Load data once for this horizon
        data = load_data_for_horizon(base_dir, horizon)

        # Get y-axis ticks for this horizon
        y_ticks = Y_TICKS_BY_HORIZON[horizon]

        # Create plot grouped by model
        plotter_model = LossPlotter(y_ticks)
        plotter_model.plot_grouped_bars(*data, group_by="model")
        plotter_model.save_figure(output_dir / f"model_comp_by_model_h{horizon}.png")
        plotter_model.save_figure(output_dir / f"model_comp_by_model_h{horizon}.svg")

        # Create plot grouped by dataset
        plotter_dataset = LossPlotter(y_ticks)
        plotter_dataset.plot_grouped_bars(*data, group_by="dataset")
        plotter_dataset.save_figure(
            output_dir / f"model_comp_by_dataset_h{horizon}.png"
        )
        plotter_dataset.save_figure(
            output_dir / f"model_comp_by_dataset_h{horizon}.svg"
        )

    print("Done! All plots have been generated.")
