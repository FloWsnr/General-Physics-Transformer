"""
Mock data for checking if the model can learn.

By: Florian Wiesner
Date: 2025-04-06
"""

import torch
from torch.utils.data import Dataset


class MockData(Dataset):
    """
    Mock data for checking if the model can learn.
    """

    def __init__(
        self,
        num_channels: int,
        num_time_steps: int,
        num_height_pixels: int,
        num_width_pixels: int,
        num_lines: int,
    ):
        self.time_steps = num_time_steps
        self.height_pixels = num_height_pixels

        self.data = torch.zeros(
            num_time_steps + 1,
            num_height_pixels,
            num_width_pixels,
            num_channels,
        )

        self.line_length = num_lines

    def __len__(self):
        return self.height_pixels - self.time_steps * self.line_length

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.data.clone()
        
        for i in range(self.time_steps +1):
            start = index + i * self.line_length
            end = start + self.line_length
            x[i, start:end, :, :] = 1

        # Remove last time step from x
        y = x.clone()
        x = x[:-1, :, :, :]
        y = y[1:, :, :, :]

        return x, y
