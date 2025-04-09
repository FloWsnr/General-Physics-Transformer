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
    show: bool = False,
) -> None:
    """
    Visualize the inputs, predictions and targets. (batch_size, time_steps, height, width, channels)

    Plots a single sample with time steps (rows) and all channels (cols).
    Inputs, predictions and targets are plotted below each other.

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
    show : bool
        Whether to show the plots.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    inputs = inputs.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # swap x and y axis
    inputs = np.transpose(inputs, (0, 1, 3, 2, 4))
    predictions = np.transpose(predictions, (0, 1, 3, 2, 4))
    targets = np.transpose(targets, (0, 1, 3, 2, 4))
    differences = predictions - targets

    B, T, W, H, C = predictions.shape

    # change the matplotlib backend to non-interactive
    plt.switch_backend("Agg")

    for channel in range(C):
        # rows = inputs, predictions, targets, diff
        # cols = time steps
        fig, axs = plt.subplots(4, T, figsize=(4 * T, 3 * 4))

        # Create a common normalization for all plots
        vmin = min(
            inputs[0, ..., channel].min(),
            predictions[0, ..., channel].min(),
            targets[0, ..., channel].min(),
        )
        vmax = max(
            inputs[0, ..., channel].max(),
            predictions[0, ..., channel].max(),
            targets[0, ..., channel].max(),
        )

        for j in range(T):
            input = inputs[0, j, :, :, channel]
            pred = predictions[0, j, :, :, channel]
            target = targets[0, j, :, :, channel]
            diff = differences[0, j, :, :, channel]

            img1 = axs[0, j].imshow(input, vmin=vmin, vmax=vmax)
            img2 = axs[1, j].imshow(pred, vmin=vmin, vmax=vmax)
            img3 = axs[2, j].imshow(target, vmin=vmin, vmax=vmax)
            img4 = axs[3, j].imshow(diff)

            # set the title only once for the first row
            if j == 0:
                axs[0, j].set_title(f"Input: Channel {channel}")
                axs[1, j].set_title(f"Pred: Channel {channel}")
                axs[2, j].set_title(f"Target: Channel {channel}")
                axs[3, j].set_title(f"Diff: Channel {channel}")

        # Add two colorbars - one for input/pred/target and one for diff
        fig.subplots_adjust(
            left=0.1, right=0.8, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3
        )

        # First colorbar for input, prediction, and target
        cbar_ax1 = fig.add_axes([0.83, 0.4, 0.03, 0.45])
        cbar1 = fig.colorbar(img1, cax=cbar_ax1)
        cbar1.set_label("Input/Pred/Target")

        # Second colorbar for difference
        cbar_ax2 = fig.add_axes([0.83, 0.15, 0.03, 0.2])
        cbar2 = fig.colorbar(img4, cax=cbar_ax2)
        cbar2.set_label("Difference")

        fig.savefig(save_path / f"channel_{channel}.png", dpi=300, bbox_inches="tight")
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
