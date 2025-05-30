from collections import OrderedDict
from itertools import cycle
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

import numpy as np
import pandas as pd

# Load the CVT style
white_style = Path(__file__).parent / "white.mplstyle"
black_style = Path(__file__).parent / "black.mplstyle"


class BasePlotter:
    def __init__(self, color: Literal["white", "black"] = "white") -> None:
        if color == "white":
            plt.style.use(white_style)
        else:
            plt.style.use(black_style)

        ##################################################
        ########### Plotting parameters ##################
        ##################################################

        symbol1 = "o"  # circle
        symbol2 = "^"  # triangle
        symbol3 = "D"  # diamond
        symbol4 = "s"  # square
        symbol5 = "v"  # triangle down
        symbol6 = "p"  # pentagon
        self.symbol_cycler = cycle(
            [symbol1, symbol2, symbol3, symbol4, symbol5, symbol6]
        )

        if color == "white":
            color1 = "#000000"  # Black
            color2 = "#9C9E9F"  # light grey (AVT 50%)
            color3 = "#00549F"  # blue (AVT)
            color4 = "#70AD47"  # green (AVT)
            color5 = "#646567"  # grey (AVT)
            color6 = "#8EBAE5"  # light blue (AVT 50%)
            color7 = "#407FB7"  # light blue (AVT 75%)
        else:
            color1 = "#FFFFFF"  # White
            color2 = "#9C9E9F"  # light grey (AVT 50%)
            color3 = "#00549F"  # blue (AVT)
            color4 = "#70AD47"  # green (AVT)
            color5 = "#646567"  # grey (AVT)
            color6 = "#8EBAE5"  # light blue (AVT 50%)
            color7 = "#407FB7"  # light blue (AVT 75%)

        self.color_cycler = cycle(
            [color1, color2, color3, color4, color5, color6, color7]
        )

        self.field_params = OrderedDict(
            {
                "pressure": {
                    "color": color1,
                    "symbol": symbol1,
                },
                "density": {
                    "color": color3,
                    "symbol": symbol2,
                },
                "temperature": {
                    "color": color4,
                    "symbol": symbol3,
                },
                "velocity_x": {
                    "color": color2,
                    "symbol": symbol4,
                },
                "velocity_y": {
                    "color": color5,
                    "symbol": symbol5,
                },
            }
        )

        self.alpha = 0.2  # transparency of error region
        self.linestyle = "-"
        self.linestyle_dashed = (0, (3, 1))  # dashed

        self.open_symbol = True

        self.legend_fontsize = 8
        self.draw_legend = True

        ##################################################
        ########### Data parameters ######################
        ##################################################
        self.fig, self.ax = plt.subplots()

    def plot_data(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        linestyle: str = "-",
        symbolstyle: str = "o",
        color: str = "#000000",
        label="",
        open_symbol: bool = False,
        markerstep: int = 1,
        layer: int = 3,
        markersize: int = None,
        ax=None,
    ):
        """
        Function to plot open and closed symbols

        Parameters
        ----------

        x_data : np.ndarray
            x-data of the plot

        y_data : np.ndarray
            y-data of the plot

        linestyle : str, optional
            Linestyle of the line, by default "-"

        symbolstyle : str, optional
            Symbolstyle of the data points, by default "o"

        color : str, optional
            Color of the line and the symbols, by default "#000000" (black)

        label : str, optional
            Label of the plot, by default ""

        open_symbol : bool, optional
            If True, the symbols are open, by default False

        markerstep : int, optional
            Every nth data point is plotted, by default 1

        layer : int, optional
            Layer of the plot, by default 1
            Which layer is plotted on top of which layer is determined by the "layer" parameter

        """

        if ax is None:
            ax = self.ax

        if not open_symbol:
            markerfacecolor = color
            markeredgecolor = color
        else:
            markerfacecolor = "black"
            markeredgecolor = color

        if markersize is None:
            markersize = matplotlib.rcParams["lines.markersize"]

        ax.plot(
            x_data,
            y_data,
            linestyle=linestyle,
            marker=symbolstyle,
            markevery=markerstep,
            mfc=markerfacecolor,
            mec=markeredgecolor,
            color=color,
            zorder=layer,
            label=label,
            markersize=markersize,
        )

    def plot_error_bars(
        self,
        x_data,
        y_data,
        x_err=None,
        y_err=None,
        color="#000000",
        errorstep=1,
        layer: int = 2,
    ):
        """
        Function to plot error bars

        Parameters
        ----------

        x_data : np.ndarray
            x-data of the plot

        y_data : np.ndarray
            y-data of the plot

        x_err : np.ndarray, optional
            x-error of the plot, by default None

        y_err : np.ndarray, optional
            y-error of the plot, by default None

        color : str, optional
            Color of the error bars, by default "#000000" (black)

        errorstep : int, optional
            Every nth data point is plotted, by default 1

        layer : int, optional
            Layer of the plot, by default 2 (behind of the data points)

        """

        self.ax.errorbar(
            x_data,
            y_data,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            ecolor=color,
            elinewidth=matplotlib.rcParams["axes.linewidth"] / 2,
            capthick=matplotlib.rcParams["axes.linewidth"] / 2,
            errorevery=errorstep,
            zorder=layer,
        )

    def plot_error_region(
        self, x_data, y_data, y_err, color, alpha=0.2, edgecolor=None, layer: int = 1
    ):
        """
        Function to plot error regions

        Parameters
        ----------

        x_data : np.ndarray
            x-data of the plot

        y_data : np.ndarray
            y-data of the plot

        y_err : np.ndarray
            y-error of the plot

        color : str
            Color of the error region

        alpha : float, optional
            Transparency of the error region, by default 0.2

        edgecolor : str, optional
            Color of the edge of the error region, by default None (same as color)

        layer : int, optional
            Layer of the plot, by default 3 (behind of the data points and error bars)
        """

        self.ax.fill_between(
            x_data,
            y1=y_data - y_err,
            y2=y_data + y_err,
            color=color,
            alpha=alpha,
            zorder=layer,
            edgecolor=edgecolor,
        )

    def setup_figure(
        self,
        x_ticks: list = [],
        y_ticks: list = [],
        x_label: str = "x",
        y_label: str = "y",
        x_log=False,
        y_log=False,
        padding_factor: float = 0.1,
    ) -> None:
        """
        Setup the figure with the given parameters

        Parameters
        ----------

        x_ticks : list, optional
            List of the x-ticks, by default []

        y_ticks : list, optional
            List of the y-ticks, by default []

        x_label : str, optional
            Label of the x-axis, by default "x"

        y_label : str, optional
            Label of the y-axis, by default "y"

        x_log : bool, optional
            If True, the x-axis is logarithmic, by default False

        y_log : bool, optional
            If True, the y-axis is logarithmic, by default False

        padding_factor : float, optional
            Padding of the x-axis and y-axis, by default 0.1
        """

        # Set the labels
        self.ax.set_xlabel(x_label, fontweight="normal")
        self.ax.set_ylabel(y_label, fontweight="normal")

        # Padding: Make the x-axis larger than the biggest and smallest x value
        if x_log:
            self.ax.set_xscale("log")
            x_span = np.log10(x_ticks[-1]) - np.log10(x_ticks[0])
            padding = x_span * padding_factor
            x_min = x_ticks[0] / 10**padding
            x_max = x_ticks[-1] * 10**padding

        else:
            x_span = x_ticks[-1] - x_ticks[0]
            padding = x_span * padding_factor
            x_min = x_ticks[0] - padding
            x_max = x_ticks[-1] + padding

            # Minor ticks only work on non-log scale
            self.ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        self.ax.set_xlim([x_min, x_max])
        self.ax.set_xticks(x_ticks)

        # Make the y-axis larger than the biggest and smallest y value
        if y_log:
            self.ax.set_yscale("log")
            y_span = np.log10(y_ticks[-1]) - np.log10(y_ticks[0])
            padding = y_span * padding_factor
            y_min = y_ticks[0] / 10**padding
            y_max = y_ticks[-1] * 10**padding

        else:
            y_span = y_ticks[-1] - y_ticks[0]
            padding = y_span * padding_factor
            y_min = y_ticks[0] - padding
            y_max = y_ticks[-1] + padding

            # Minor ticks only work on non-log scale
            self.ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        self.ax.set_ylim([y_min, y_max])
        self.ax.set_yticks(y_ticks)

        self.ax.xaxis.set_major_formatter(ScalarFormatter())

    def add_second_y_axis(self, y_ticks: list = [], y_label: str = "y2"):
        ax2 = self.ax.twinx()

        # Set the labels
        ax2.set_ylabel(y_label, fontweight="normal")

        # Make the y-axis larger than the biggest and smallest y value
        y_span = y_ticks[-1] - y_ticks[0]
        padding = y_span * 0.1
        y_min = y_ticks[0] - padding
        y_max = y_ticks[-1] + padding

        ax2.set_ylim([y_min, y_max])
        ax2.set_yticks(y_ticks)
        ax2.xaxis.set_major_formatter(ScalarFormatter())
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

        self.ax2 = ax2
        self.ax.set_zorder(2)  # Original axes
        self.ax2.set_zorder(1)  # Twin axes
        self.ax.patch.set_visible(False)

    def save_figure_for_publication(self, path: Path, name: str, title: str):
        path_without_legend = path / "without_legend"
        path_without_legend.mkdir(parents=True, exist_ok=True)

        self.fig.savefig(path_without_legend / f"{name}.png")
        self.fig.savefig(path_without_legend / f"{name}.svg")

        path_with_legend = path / "with_legend"
        path_with_legend.mkdir(parents=True, exist_ok=True)

        self.legend(title=title)

        self.fig.savefig(path_with_legend / f"{name}.png")
        self.fig.savefig(path_with_legend / f"{name}.svg")

    def save_figure(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        self.fig.savefig(path)

    def show_figure(self):
        plt.show()

    def legend(self, title=None, loc="upper left", frameon=True):
        num_points = 1  # number of marker points in the legend
        handlelength = 2.7  # length of the handle (line and marker) in the legend

        if self.draw_legend:
            self.ax.legend(
                title=title,
                loc=loc,
                frameon=frameon,
                fontsize=self.legend_fontsize,
                title_fontsize=self.legend_fontsize,
                alignment="left",
                numpoints=num_points,
                handlelength=handlelength,
            )

    def close(self):
        plt.close(self.fig)


def calculate_combined_stats(
    df: pd.DataFrame, column_patterns: list[str], level: int = 0
) -> pd.DataFrame:
    """
    Calculate the mean, median, and standard deviation of columns that match specific patterns.
    Is used to combine statistics of the same dataset with different delta t values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    column_patterns : list of str
        List of dataset names to combine statistics for
    level : int
        Level of the column to match

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the combined statistics for each pattern
    """
    results = []
    for pattern in column_patterns:
        # Find columns that match the pattern exactly
        matching_cols = [
            col
            for col in df.columns.get_level_values(level)
            if col.startswith(pattern + "_") or col == pattern
        ]
        if matching_cols:
            # Calculate statistics across matching columns
            combined_mean = df[matching_cols].mean(axis=1).mean()
            combined_median = df[matching_cols].median(axis=1).median()
            combined_std = df[matching_cols].std(axis=1).mean()
            results.append(
                {
                    "Pattern": pattern,
                    "Combined Mean": combined_mean,
                    "Combined Median": combined_median,
                    "Combined Std": combined_std,
                }
            )

    results = pd.DataFrame(results)
    # Calculate overall statistics across all columns
    overall_stats = pd.DataFrame(
        [
            {
                "Pattern": "OVERALL",
                "Combined Mean": np.nanmean(df.values),
                "Combined Median": np.nanmedian(df.values),
                "Combined Std": np.nanstd(df.values),
            }
        ]
    )
    data = pd.concat([results, overall_stats], ignore_index=True).T
    return data


def calculate_combined_stats_rollout(
    df: pd.DataFrame, column_patterns: list[str], level: int = 0
) -> pd.DataFrame:
    """
    Calculate statistics for multi-level columns while preserving the second level structure.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing multi-level column data
    column_patterns : list of str
        List of dataset names to combine statistics for
    level : int
        Level of the column to match (default=0 for first level)

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the combined statistics for each pattern,
        preserving the second level column structure
    """
    data = []
    index = []
    for pattern in column_patterns:
        # Find columns that match the pattern exactly in the first level
        matching_cols = [
            col
            for col in df.columns.get_level_values(level)
            if col.startswith(pattern + "_") or col == pattern
        ]
        if matching_cols:
            # Get all second level columns for the matching first level columns
            second_level_cols = df.columns.get_level_values(1).unique()
            third_level_cols = df.columns.get_level_values(2).unique()

            # Calculate statistics for each second level column
            for second_col in second_level_cols:
                for third_col in third_level_cols:
                    # Get all columns that match both the pattern and second level
                    cols_to_combine = [
                        col
                        for col in df.columns
                        if col[0] in matching_cols
                        and col[1] == second_col
                        and col[2] == third_col
                    ]

                    if cols_to_combine:
                        index.append((pattern, second_col, third_col))
                        # Calculate statistics across matching columns
                        data.append(df[cols_to_combine].mean(axis=1))

    index = pd.MultiIndex.from_tuples(index, names=["pattern", "metric", "channel"])
    df = pd.DataFrame(data, index=index).T

    return df
