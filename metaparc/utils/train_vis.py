"""
Visualize the training process (show predictions and targets)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_predictions(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Visualize the predictions and targets. (batch_size, time_steps, height, width, channels)

    Plots a single sample with all time steps (cols) and channels (rows).
    Parameters
    ----------
    predictions : torch.Tensor
        The predictions.
    targets : torch.Tensor
        The targets.
    """
    B, T, H, W, C = predictions.shape
    fig, axs = plt.subplots(C, T, figsize=(10, 5 * C))

    for i in range(B):
        for j in range(C):
            for k in range(T):
                axs[j, k].imshow(predictions[i, k, :, :, j])
                axs[j, k].set_title(f"Prediction {k}")

    plt.show()
