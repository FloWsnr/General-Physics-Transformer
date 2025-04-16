"""
Calculate derivatives of the physical fields as additional channels for the transformer model.

By: Florian Wiesner
Date: 2025-04-15    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class FiniteDifference(nn.Module):
    """
    Computes spatial derivatives using finite central difference filters.

    Parameters
    ----------
    num_channels : int
        Number of input channels.
    filter_1d : list
        1D filter for finite central difference (e.g., [-1.0, 0.0, 1.0]).
    """

    def __init__(
        self,
        num_channels=1,
        filter_1d=[-1.0, 0.0, 1.0],
    ):
        super(FiniteDifference, self).__init__()

        filter_size = len(filter_1d)
        filter_1d = torch.tensor(filter_1d, dtype=torch.float32)

        # Filters should be of shape [C_in, 1, kT, kH, kW]

        # Initialize dy_conv weights
        dy_filter = filter_1d.view(1, 1, filter_size, 1)
        dy_filter = dy_filter.repeat(num_channels, 1, 1, 1)

        self.register_buffer("dy_filter", dy_filter)  # [C,1,filter_size,1]

        # Initialize dx_conv weights
        dx_filter = filter_1d.view(1, 1, 1, filter_size)
        dx_filter = dx_filter.repeat(num_channels, 1, 1, 1)

        self.register_buffer("dx_filter", dx_filter)  # [C,1,1,filter_size]

    def forward(self, x):
        """
        Forward pass to compute spatial derivatives.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, H, W, C]

        Returns
        -------
        dy : torch.Tensor
            Derivative along y-axis, shape [B, T, H, W, C]
        dx : torch.Tensor
            Derivative along x-axis, shape [B, T, H, W, C]
        dz : torch.Tensor
            Derivative along z-axis, shape [B, T, H, W, C]
        """
        B, T, H, W, C = x.shape
        
        x = rearrange(x, "B T H W C -> B C T H W")
        dx = F.conv3d(x, self.dx_filter, padding="same", stride=1, padding_mode="zeros", groups=C)
        dy = F.conv3d(x, self.dy_filter, padding="same", stride=1, padding_mode="zeros", groups=C)
        dz = F.conv3d(x, self.dz_filter, padding="same", stride=1, padding_mode="zeros", groups=C)

        dx = rearrange(dx, "B C T H W -> B T H W C")
        dy = rearrange(dy, "B C T H W -> B T H W C")
        dz = rearrange(dz, "B C T H W -> B T H W C")
        return dy, dx, dz




