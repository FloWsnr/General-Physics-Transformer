import torch
import torch.nn as nn

from metaparc.model.base_models.parc.differentiator.num_differentiation import (
    FiniteDifference,
)


class Advection(nn.Module):
    """
    Computes the advection term based on the state variable and velocity field.

    Args:
        finite_difference_method (nn.Module): Numerical method to calculate spatial deriviatives
    """

    def __init__(self, num_differentiator: FiniteDifference):
        super(Advection, self).__init__()
        self.num_diff = num_differentiator

    def forward(
        self, state_variable: torch.Tensor, velocity_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute advection.
        Parameters
        ----------
        state_variable : torch.Tensor
            Tensor of shape [N, C, H, W]
        velocity_field : torch.Tensor
            Tensor of shape [N, 2C, H, W]

        Returns
        -------
        advect : torch.Tensor
            Advection term of shape [N, 1, H, W]
        """
        dy, dx = self.num_diff(state_variable)  # Each of shape [N, 1, H, W]
        spatial_deriv = torch.cat([dx, dy], dim=1)  # [N, 2, H, W]
        advect = torch.sum(
            spatial_deriv * velocity_field, dim=1, keepdim=True
        )  # [N, 1, H, W]
        return advect
