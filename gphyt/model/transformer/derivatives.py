"""
Calculate derivatives of the physical fields as additional channels for the transformer model.
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
    filter_1d : str = "2nd"
        Type of 1D filter for finite central difference.
        "2nd" : 2nd order finite central difference (e.g., [-1.0, 0.0, 1.0])
        "4th" : 4th order finite central difference (e.g., [1/12, -2/3, 0, 2/3, -1/12])
    """

    def __init__(
        self,
        num_channels=1,
        filter_1d="2nd",
    ):
        super(FiniteDifference, self).__init__()

        if filter_1d == "2nd":
            filter_1d = [-1.0, 0.0, 1.0]
        elif filter_1d == "4th":
            filter_1d = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
        else:
            raise ValueError(f"Invalid filter type: {filter_1d}")
        filter_size = len(filter_1d)
        filter_1d = torch.tensor(filter_1d, dtype=torch.float32, requires_grad=False)

        # Filters should be of shape [C_in, 1, kT, kH, kW]
        # Initialize dt_conv weights
        dt_filter = filter_1d.view(1, 1, filter_size, 1, 1)
        dt_filter = dt_filter.repeat(num_channels, 1, 1, 1, 1)
        self.register_buffer("dt_filter", dt_filter)  # [C,1,filter_size,1,1]

        # Initialize dh_conv weights
        dh_filter = filter_1d.view(1, 1, 1, filter_size, 1)
        dh_filter = dh_filter.repeat(num_channels, 1, 1, 1, 1)
        self.register_buffer("dh_filter", dh_filter)  # [C,1,1,filter_size,1]

        # Initialize dw_conv weights
        dw_filter = filter_1d.view(1, 1, 1, 1, filter_size)
        dw_filter = dw_filter.repeat(num_channels, 1, 1, 1, 1)
        self.register_buffer("dw_filter", dw_filter)  # [C,1,1,1,filter_size]

        # Calculate padding needed to maintain output size
        self.pad_t = (self.dt_filter.shape[2] - 1) // 2
        self.pad_h = (self.dh_filter.shape[3] - 1) // 2
        self.pad_w = (self.dw_filter.shape[4] - 1) // 2

    @torch.no_grad()
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute spatial derivatives.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, H, W, C]

        Returns
        -------
        dt : torch.Tensor
            Derivative along t-axis, shape [B, T, H, W, C]
        dh : torch.Tensor
            Derivative along h-axis, shape [B, T, H, W, C]
        dw : torch.Tensor
            Derivative along w-axis, shape [B, T, H, W, C]
        """
        B, T, H, W, C = x.shape

        x = rearrange(x, "B T H W C -> B C T H W")

        # Pad input tensor for temporal derivative
        x_t = F.pad(x, (0, 0, 0, 0, self.pad_t, self.pad_t), mode="replicate")
        dt = F.conv3d(x_t, self.dt_filter, padding=0, stride=1, groups=C)

        # Pad input tensor for height derivative
        x_h = F.pad(x, (0, 0, self.pad_h, self.pad_h, 0, 0), mode="replicate")
        dh = F.conv3d(x_h, self.dh_filter, padding=0, stride=1, groups=C)

        # Pad input tensor for width derivative
        x_w = F.pad(x, (self.pad_w, self.pad_w, 0, 0, 0, 0), mode="replicate")
        dw = F.conv3d(x_w, self.dw_filter, padding=0, stride=1, groups=C)

        dt = rearrange(dt, "B C T H W -> B T H W C")
        dh = rearrange(dh, "B C T H W -> B T H W C")
        dw = rearrange(dw, "B C T H W -> B T H W C")
        return dt, dh, dw
