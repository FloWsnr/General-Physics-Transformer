"""
Visualize the training process (show predictions and targets)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import wandb

def visualize_predictions(
    save_path: Path,
    inputs: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Visualize the inputs, predictions and targets. (batch_size, time_steps, height, width, channels)

    Plots a single sample with time steps (rows) and all channels (cols).
    Inputs, predictions and targets are plotted below each other.

    Parameters
    ----------
    save_path : Path
        The path to save the plot.
    inputs : torch.Tensor
        The inputs.
    predictions : torch.Tensor
        The predictions.
    targets : torch.Tensor
        The targets.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = inputs.detach().cpu().numpy()
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


def log_predictions_wandb(
    run: wandb.wandb_run.Run,
    input: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    name_prefix: str | None = None,
):
    """
    Log the predictions and targets to wandb.

    Parameters
    ----------
    run : wandb.wandb_run.Run
        The wandb run.
    input : torch.Tensor
        The input. (batch_size, time_steps, height, width, channels)
    predictions : torch.Tensor
        The predictions. (batch_size, time_steps, height, width, channels)
    targets : torch.Tensor
        The targets. (batch_size, time_steps, height, width, channels)
    name_prefix : str | None
        The prefix to add to the names of the images.
    """

    B, T, H, W, C = input.shape
    input = input.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    batch_idx = 0
    data = {}

    for i in range(C):
        img_input = input[batch_idx, :, :, :, i]
        img_pred = predictions[batch_idx, :, :, :, i]
        img_target = targets[batch_idx, :, :, :, i]

        min_val = min(img_input.min(), img_pred.min(), img_target.min())
        max_val = max(img_input.max(), img_pred.max(), img_target.max())

        # normalize
        # Normalize the images to [0, 1] range for visualization
        if max_val > min_val:  # Avoid division by zero
            img_input = (img_input - min_val) / (max_val - min_val)
            img_pred = (img_pred - min_val) / (max_val - min_val)
            img_target = (img_target - min_val) / (max_val - min_val)
        else:
            # If all values are the same, set to 0.5 for visualization
            img_input = np.ones_like(img_input) * 0.5
            img_pred = np.ones_like(img_pred) * 0.5
            img_target = np.ones_like(img_target) * 0.5

        for j in range(T):
            img_input_t = img_input[j, :, :]
            img_pred_t = img_pred[j, :, :]
            img_target_t = img_target[j, :, :]

            # convert to PIL image
            img_input_t = Image.fromarray((img_input_t * 255).astype(np.uint8))
            img_pred_t = Image.fromarray((img_pred_t * 255).astype(np.uint8))
            img_target_t = Image.fromarray((img_target_t * 255).astype(np.uint8))

            data[f"{name_prefix}/input_t{j}_c{i}"] = wandb.Image(img_input_t, file_type="png")
            data[f"{name_prefix}/predictions_t{j}_c{i}"] = wandb.Image(img_pred_t, file_type="png")
            data[f"{name_prefix}/targets_t{j}_c{i}"] = wandb.Image(img_target_t, file_type="png")

    run.log(data)