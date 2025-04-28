"""
Utility functions for creating visualization videos of trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio.v3 as iio
from pathlib import Path


def create_field_video(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    title: str = "Field",
    fps: int = 1,
) -> None:
    """Create a video visualization of the physical field over time.

    Parameters
    ----------
    ground_truth : np.ndarray
        Ground truth data array of shape (timesteps, height, width, channels)
    prediction : np.ndarray
        Prediction data array of shape (timesteps, height, width, channels)
    output_path : Path
        Path where the output video will be saved
    title : str, optional
        Title prefix for the video, by default "Field"
    fps : int, optional
        Frames per second for the video, by default 30
    """
    channel_names = ["p", "rho", "T", "u", "v"]
    for channel, name in enumerate(channel_names):
        gt = ground_truth[..., channel]
        pred = prediction[..., channel]

        # Create normalized colormap for consistent color scaling
        norm_gt = Normalize(vmin=gt.min(), vmax=gt.max())
        norm_pred = Normalize(vmin=pred.min(), vmax=pred.max())

        # Create frames
        frames = []
        for t in range(ground_truth.shape[0]):
            # Create figure and plot
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im_gt = ax[0].imshow(gt[t, :, :], norm=norm_gt)
            im_pred = ax[1].imshow(pred[t, :, :], norm=norm_gt)

            ax[0].set_title(f"Ground Truth - {name} (t={t})")
            ax[1].set_title(f"Prediction - {name} (t={t})")

            # add colorbar for each subplot
            fig.colorbar(im_gt, ax=ax[0])
            fig.colorbar(im_pred, ax=ax[1])

            # Convert plot to image array
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            frames.append(frame)
            plt.close()

        # Save as GIF
        vid_path = output_path / f"{title}_{name}.gif"
        print(f"Writing {len(frames)} frames to {vid_path}")
        iio.imwrite(vid_path, frames, fps=fps)
        print(f"Animation saved to {vid_path}")
