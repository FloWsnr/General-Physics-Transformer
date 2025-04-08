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

            axs[0, j].imshow(input, vmin=vmin, vmax=vmax)
            axs[1, j].imshow(pred, vmin=vmin, vmax=vmax)
            axs[2, j].imshow(target, vmin=vmin, vmax=vmax)
            axs[3, j].imshow(diff, vmin=vmin, vmax=vmax)

            # set the title only once for the first row
            if j == 0:
                axs[0, j].set_title(f"Input: Channel {channel}")
                axs[1, j].set_title(f"Pred: Channel {channel}")
                axs[2, j].set_title(f"Target: Channel {channel}")
                axs[3, j].set_title(f"Diff: Channel {channel}")

        # Add a single colorbar for all subplots
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(axs[3, 0].imshow(diff, vmin=vmin, vmax=vmax), cax=cbar_ax)

        # plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig.savefig(save_path / f"channel_{channel}.png")
        plt.close(fig)


def log_predictions_wandb(
    run,
    image_path: Path,
    name_prefix: str | None = None,
):
    """
    Log the predictions and targets to wandb.

    Parameters
    ----------
    run : wandb.wandb_run.Run
        The wandb run.
    image_path: Path
        path to the images
    name_prefix : str | None
        The prefix to add to the names of the images.
    """

    data = {}
    for image in image_path.glob("**/*.png"):
        img = Image.open(image)
        data[f"{name_prefix}/{image.name}"] = wandb.Image(
            img, file_type="png", mode="RGB"
        )
    run.log(data)
