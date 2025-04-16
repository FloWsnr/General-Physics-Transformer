"""
Visualize the training process (show predictions and targets)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image


def visualize_predictions(
    save_path: Path,
    inputs: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_samples: int = 1,
    svg: bool = False,
) -> None:
    """
    Visualize the inputs, predictions and targets.
    Inputs: (batch_size, time_steps, height, width, channels)
    Predictions: (batch_size, 1, height, width, channels)
    Targets: (batch_size, 1, height, width, channels)

    Plots a single sample with time steps (rows) and all channels (cols).
    Inputs, predictions and targets are plotted below each other.
    Each channel uses a different colormap for better visualization.

    Parameters
    ----------
    save_path : Path
        The path to save the figures.
    inputs : torch.Tensor
        The inputs.
    predictions : torch.Tensor
        The predictions.
    targets : torch.Tensor
        The targets.
    num_samples : int
        The number of samples to visualize.
    svg : bool
        Whether to (additionally) save the plots as svg.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = inputs.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    for sample_idx in range(num_samples):
        sample_inputs = inputs[sample_idx, ...]
        sample_predictions = predictions[sample_idx, ...]
        sample_targets = targets[sample_idx, ...]

        # swap x and y axis
        sample_inputs = np.transpose(sample_inputs, (0, 2, 1, 3))
        sample_predictions = np.transpose(sample_predictions, (0, 2, 1, 3))
        sample_targets = np.transpose(sample_targets, (0, 2, 1, 3))

        # Calculate absolute difference
        sample_differences = sample_predictions - sample_targets
        # Calculate relative error (avoiding division by zero)
        epsilon = 1e-8  # Small constant to avoid division by zero
        sample_relative_errors = np.abs(sample_differences) / (
            np.abs(sample_targets) + epsilon
        )

        T, W, H, C = sample_inputs.shape

        # change the matplotlib backend to non-interactive
        plt.switch_backend("Agg")

        # Define a list of distinct colormaps for each channel
        colormaps = [
            "inferno",
            "viridis",
            "magma",
            "viridis",
            "viridis",
        ]

        channel_names = [
            "Pressure",
            "Density",
            "Temperature",
            "Velocity-X",
            "Velocity-Y",
        ]

        # Only visualize the last 4 time steps if there are more than 4
        time_steps = min(T, 4)
        # start_idx = max(0, T - time_steps)
        # time steps of input plus 3 rows for predictions, targets, diff
        rows = time_steps + 3
        cols = C
        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 2 * rows))

        # Set column headers (channel names)
        for channel in range(C):
            axs[0, channel].set_title(
                channel_names[channel], pad=20, fontsize=14, fontweight="bold"
            )

        for channel in range(C):
            # traget and pred only have one time step
            pred_channel = sample_predictions[0, ..., channel]
            target_channel = sample_targets[0, ..., channel]
            diff_channel = sample_differences[0, ..., channel]
            # Create a common normalization for all plots
            vmin = min(
                sample_inputs[..., channel].min(),
                pred_channel.min(),
                target_channel.min(),
            )
            vmax = max(
                sample_inputs[..., channel].max(),
                pred_channel.max(),
                target_channel.max(),
            )

            # Get the colormap for this channel
            cmap = colormaps[channel % len(colormaps)]
            for j in range(time_steps):
                index = T - time_steps + j
                input = sample_inputs[index, ..., channel]
                img_input = axs[j, channel].imshow(
                    input, vmin=vmin, vmax=vmax, cmap=cmap
                )
                # Remove axis ticks
                axs[j, channel].set_xticks([])
                axs[j, channel].set_yticks([])

            img_pred = axs[-3, channel].imshow(
                pred_channel, vmin=vmin, vmax=vmax, cmap=cmap
            )
            img_target = axs[-2, channel].imshow(
                target_channel, vmin=vmin, vmax=vmax, cmap=cmap
            )
            img_diff = axs[-1, channel].imshow(diff_channel, cmap="RdBu_r")

            # Remove axis ticks for prediction, target and difference plots
            axs[-3, channel].set_xticks([])
            axs[-3, channel].set_yticks([])
            axs[-2, channel].set_xticks([])
            axs[-2, channel].set_yticks([])
            axs[-1, channel].set_xticks([])
            axs[-1, channel].set_yticks([])

            # Set row labels for the first column
            if channel == 0:
                axs[0, channel].set_ylabel(
                    "Input",
                    rotation=0,
                    labelpad=40,
                    va="center",
                    fontweight="bold",
                    fontsize=12,
                )
                axs[-3, channel].set_ylabel(
                    "Prediction",
                    rotation=0,
                    labelpad=40,
                    va="center",
                    fontweight="bold",
                    fontsize=12,
                )
                axs[-2, channel].set_ylabel(
                    "Target",
                    rotation=0,
                    labelpad=40,
                    va="center",
                    fontweight="bold",
                    fontsize=12,
                )
                axs[-1, channel].set_ylabel(
                    "Error",
                    rotation=0,
                    labelpad=40,
                    va="center",
                    fontweight="bold",
                    fontsize=12,
                )

            # Add two colorbars for each column
            # First colorbar for input/prediction/target (top)
            cbar_ax1 = fig.add_axes(
                [0.1 + (channel + 1) * (0.75 / C) - 0.01, 0.6, 0.01, 0.2]
            )
            cbar1 = fig.colorbar(img_input, cax=cbar_ax1, orientation="vertical")
            cbar1.set_label("Value")

            # Second colorbar for relative error (bottom)
            cbar_ax2 = fig.add_axes(
                [0.1 + (channel + 1) * (0.75 / C) - 0.01, 0.1, 0.01, 0.2]
            )
            cbar2 = fig.colorbar(img_diff, cax=cbar_ax2, orientation="vertical")
            cbar2.set_label("Error")

        # Adjust the figure layout to make room for the colorbars
        fig_dir = save_path.parent
        fig_name = save_path.stem + f"_{sample_idx}"
        fig.subplots_adjust(right=0.85)
        fig_path = fig_dir / f"{fig_name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        if svg:
            fig_path = fig_dir / f"{fig_name}.svg"
            fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)


def log_predictions_wandb(
    run,
    image_path: Path,
    name_prefix: str,
):
    """
    Log the predictions and targets to wandb.

    Parameters
    ----------
    run : wandb.wandb_run.Run
        The wandb run.
    image_path: Path
        path to the images
    name_prefix: str
        prefix for the image names
    """

    data = {}
    for image in image_path.glob("**/*.png"):
        img = Image.open(image)
        data[f"{name_prefix}/{image.name}"] = wandb.Image(
            img, file_type="png", mode="RGB"
        )
    run.log(data)
