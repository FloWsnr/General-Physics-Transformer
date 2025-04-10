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
        binary: bool = False,
    ):
        self.height_pixels = num_height_pixels
        self.width_pixels = num_width_pixels
        self.num_samples = num_samples
        self.num_time_steps = num_time_steps
        self.data = torch.zeros(
            num_time_steps + 1, num_height_pixels, num_width_pixels, num_channels
        )
        self.binary = binary

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.data.clone()

        # Create a circle at a random position
        center_x = torch.randint(1, self.width_pixels // 2, (1,))
        center_y = torch.randint(1, self.height_pixels // 2, (1,))
        radius = min(center_x, center_y).clone()

        # Create a circle using the circle equation: (x-h)^2 + (y-k)^2 = r^2
        y_indices, x_indices = torch.meshgrid(
            torch.arange(self.height_pixels),
            torch.arange(self.width_pixels),
            indexing="ij",
        )

        # move the circle in the x direction
        for i in range(self.num_time_steps + 1):  # +1 to include the initial position
            # Calculate distance from each point to the center
            distance_squared = (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2

            # Create a gradual circle with intensity decreasing from center to edge
            # Values will be 1 at center, decreasing to 0 at radius
            intensity = torch.maximum(
                1 - torch.sqrt(distance_squared) / radius, torch.tensor(0.0)
            )
            intensity = intensity.unsqueeze(-1)

            # Apply the intensity values to all channels
            x[i, ..., :] = intensity
            # check for nan
            assert not torch.isnan(x[i, ..., :]).any(), "x contains nan"

            # Move the center for the next time step (after setting the current one)
            center_x += 2
            center_y += 2

        y = x[-1, ...].unsqueeze(0)
        x = x[:-1, ...]

        if self.binary:
            x = x.round()
            y = y.round()

        return x, y


class MockShrinkingCircleData(Dataset):
    """
    Mock data for checking if the model can learn.
    
    Parameters
    ----------
    num_channels : int
        Number of channels in the data.
    num_time_steps : int
        Number of time steps in the sequence.
    num_height_pixels : int
        Height of the image in pixels.
    num_width_pixels : int
        Width of the image in pixels.
    num_samples : int
        Number of samples in the dataset.
    binary : bool, optional
        Whether to binarize the output. Default is False.
    """

    def __init__(
        self,
        num_channels: int,
        num_time_steps: int,
        num_height_pixels: int,
        num_width_pixels: int,
        num_samples: int,
        binary: bool = False,
    ):
        self.height_pixels = num_height_pixels
        self.width_pixels = num_width_pixels
        self.num_samples = num_samples
        self.num_time_steps = num_time_steps
        self.data = torch.zeros(
            num_time_steps + 1, num_height_pixels, num_width_pixels, num_channels
        )
        self.binary = binary

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.data.clone()

        # Create a circle at a random position
        center_x = torch.randint(
            self.width_pixels // 4, self.width_pixels // 4 * 3, (1,)
        )
        center_y = torch.randint(
            self.height_pixels // 4, self.height_pixels // 4 * 3, (1,)
        )

        # Start with a large radius
        max_radius = min(
            center_x,
            center_y,
            self.width_pixels - center_x,
            self.height_pixels - center_y,
        ).clone()
        initial_radius = max_radius.clone()

        # Calculate radius reduction per time step
        radius_step = initial_radius / (self.num_time_steps + 1)

        # Create a circle using the circle equation: (x-h)^2 + (y-k)^2 = r^2
        y_indices, x_indices = torch.meshgrid(
            torch.arange(self.height_pixels),
            torch.arange(self.width_pixels),
            indexing="ij",
        )

        # Shrink the circle with each time step
        for i in range(self.num_time_steps + 1):  # +1 to include the initial position
            current_radius = initial_radius - i * radius_step
            
            # Calculate distance from each point to the center
            distance_squared = (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2
            
            # Create a gradual circle with intensity decreasing from center to edge
            # Values will be 1 at center, decreasing to 0 at radius
            intensity = torch.maximum(
                1 - torch.sqrt(distance_squared) / current_radius, torch.tensor(0.0)
            )
            intensity = intensity.unsqueeze(-1)
            
            # Apply the intensity values to all channels
            x[i, ..., :] = intensity
            # check for nan
            assert not torch.isnan(x[i, ..., :]).any(), "x contains nan"

        y = x[-1, ...].unsqueeze(0)
        x = x[:-1, ...]
        
        if self.binary:
            x = x.round()
            y = y.round()
            
        return x, y


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
