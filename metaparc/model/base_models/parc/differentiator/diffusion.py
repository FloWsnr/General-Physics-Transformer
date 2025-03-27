import torch
import torch.nn as nn

from metaparc.model.base_models.parc.differentiator.num_differentiation import (
    FiniteDifference,
)


class Diffusion(nn.Module):
    """
    Computes the Laplacian of the state variable using finite difference filters.

    Parameters
    ----------
    num_differentiator : FiniteDifference
        Numerical method to calculate spatial deriviatives
    """

    def __init__(self, num_differentiator: FiniteDifference):
        super(Diffusion, self).__init__()
        self.num_diff = num_differentiator

    def forward(self, state_variable: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the Laplacian of the state variable.

        Parameters
        ----------
        state_variable : torch.Tensor
            Tensor of shape [N, C, H, W]

        Returns:
            laplacian (torch.Tensor): Laplacian of shape [N, C, H, W]
        """
        dy, dx = self.num_diff(state_variable)  # First derivatives
        dyy, _ = self.num_diff(dy)  # Second derivative w.r.t y
        _, dxx = self.num_diff(dx)  # Second derivative w.r.t x
        laplacian = dyy + dxx  # Sum of second derivatives
        return laplacian
