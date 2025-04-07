"""
Mock data for checking if the model can learn.

By: Florian Wiesner
Date: 2025-04-06
"""

import torch
from torch.utils.data import Dataset


class MockMovingCircleData(Dataset):
    """
    Mock data for checking if the model can learn.
    """

    def __init__(
        self,
        num_channels: int,
        num_time_steps: int,
        num_height_pixels: int,
        num_width_pixels: int,
        num_samples: int,
    ):
        self.height_pixels = num_height_pixels
        self.width_pixels = num_width_pixels
        self.num_samples = num_samples
        self.num_time_steps = num_time_steps
        self.data = torch.zeros(
            num_time_steps + 1, num_height_pixels, num_width_pixels, num_channels
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.data.clone()

        # Create a circle at a random position
        center_x = torch.randint(0, self.width_pixels // 2, (1,))
        center_y = torch.randint(0, self.height_pixels // 2, (1,))
        radius = min(center_x, center_y).clone()

        # Create a circle using the circle equation: (x-h)^2 + (y-k)^2 = r^2
        y_indices, x_indices = torch.meshgrid(
            torch.arange(self.height_pixels),
            torch.arange(self.width_pixels),
            indexing="ij",
        )
        # move the circle in the x direction
        for i in range(self.num_time_steps + 1):  # +1 to include the initial position
            circle_mask = (
                (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2
            ) <= radius**2
            x[i, circle_mask, :] = 1

            # Move the center for the next time step (after setting the current one)
            center_x += 1
            center_y += 1

        return x


class MockCircleData(Dataset):
    """
    Mock data for checking if the model can learn.
    """

    def __init__(
        self,
        num_channels: int,
        num_time_steps: int,
        num_height_pixels: int,
        num_width_pixels: int,
        num_samples: int,
    ):
        self.height_pixels = num_height_pixels
        self.width_pixels = num_width_pixels
        self.num_samples = num_samples
        self.data = torch.zeros(
            num_time_steps + 1, num_height_pixels, num_width_pixels, num_channels
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.data.clone()

        # Create a circle at a random position
        center_x = torch.randint(0, self.width_pixels, (1,))
        center_y = torch.randint(0, self.height_pixels, (1,))
        radius = min(center_x, center_y)

        # Create a circle using the circle equation: (x-h)^2 + (y-k)^2 = r^2
        y_indices, x_indices = torch.meshgrid(
            torch.arange(self.height_pixels),
            torch.arange(self.width_pixels),
            indexing="ij",
        )
        circle_mask = (
            (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2
        ) <= radius**2
        x[:, circle_mask, :] = 1

        return x
