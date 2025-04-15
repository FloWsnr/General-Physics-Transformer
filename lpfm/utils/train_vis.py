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
    svg: bool = False,
) -> None:
    """
    Visualize the inputs, predictions and targets. (batch_size, time_steps, height, width, channels)

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
    svg : bool
        Whether to (additionally) save the plots as svg.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = inputs.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    batch_idx = 0
    inputs = inputs[batch_idx, ...]
    predictions = predictions[batch_idx, ...]
    targets = targets[batch_idx, ...]

    # swap x and y axis
    inputs = np.transpose(inputs, (0, 2, 1, 3))
    predictions = np.transpose(predictions, (0, 2, 1, 3))
    targets = np.transpose(targets, (0, 2, 1, 3))

    # Calculate absolute difference
    differences = predictions - targets
    # Calculate relative error (avoiding division by zero)
    epsilon = 1e-8  # Small constant to avoid division by zero
    relative_errors = np.abs(differences) / (np.abs(targets) + epsilon)

    T, W, H, C = inputs.shape

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

    rows = T + 3  # time steps of input plus 3 rows for predictions, targets, diff
    cols = C
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 2 * rows))

    # Set column headers (channel names)
    for channel in range(C):
        axs[0, channel].set_title(
            channel_names[channel], pad=20, fontsize=14, fontweight="bold"
        )

    for channel in range(C):
        # traget and pred only have one time step
        pred_channel = predictions[0, ..., channel]
        target_channel = targets[0, ..., channel]
        diff_channel = differences[0, ..., channel]
        # Create a common normalization for all plots
        vmin = min(
            inputs[..., channel].min(),
            pred_channel.min(),
            target_channel.min(),
        )
        vmax = max(
            inputs[..., channel].max(),
            pred_channel.max(),
            target_channel.max(),
        )

        # Get the colormap for this channel
        cmap = colormaps[channel % len(colormaps)]

        for j in range(T):
            input = inputs[j, ..., channel]
            img_input = axs[j, channel].imshow(input, vmin=vmin, vmax=vmax, cmap=cmap)
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
    fig.subplots_adjust(right=0.85)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if svg:
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
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
