"""
Visualize the training process (show predictions and targets)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_predictions(
    save_path: Path,
    predictions: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Visualize the predictions and targets. (batch_size, time_steps, height, width, channels)

    Plots a single sample with time steps (rows) and all channels (cols).
    Predictions and targets are plotted below each other.

    Parameters
    ----------
    save_path : Path
        The path to save the plot.
    predictions : torch.Tensor
        The predictions.
    targets : torch.Tensor
        The targets.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    B, T, H, W, C = predictions.shape
    fig, axs = plt.subplots(2 * T, C, figsize=(3 * T, 2 * 3 * C))

    for j in range(T):
        for k in range(C):
            pred = predictions[0, j, :, :, k]
            target = targets[0, j, :, :, k]
            axs[2 * j, k].imshow(pred)
            axs[2 * j, k].set_title(f"Pred: Channel {k}, Time {j}")
            axs[2 * j + 1, k].imshow(target)
            axs[2 * j + 1, k].set_title(f"Target: Channel {k}, Time {j}")

    # save the plot
    fig.savefig(save_path)
    plt.close(fig)
