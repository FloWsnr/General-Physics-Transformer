import torch
import torch.nn as nn

from metaparc.model.base_models.parc.differentiator.advection import Advection
from metaparc.model.base_models.parc.differentiator.diffusion import Diffusion
from metaparc.model.base_models.parc.differentiator.num_differentiation import (
    FiniteDifference,
)
from metaparc.model.base_models.unet import Unet
from metaparc.model.base_models.parc.differentiator.transform_net import (
    TransformNet,
)


class Differentiator(nn.Module):
    """ """

    def __init__(self, num_input_channels: int):
        super(Differentiator, self).__init__()

        # Numerical calculation of advection and diffusion
        # No learnable parameters
        self.advection = Advection(FiniteDifference())
        self.diffusion = Diffusion(FiniteDifference())

        # Reaction network (basically feature extraction of all input features)
        # Learnable parameters
        self.reaction = Unet(
            block_dimensions=[
                64,
                64 * 2,
                64 * 4,
                64 * 8,
                64 * 16,
            ],  # Channel dimensions at different levels of the UNet
            input_channels=num_input_channels,  # Number of input channels: p, re, u, v
            output_channels=128,  # Number of output features (feature extraction capacity)
            kernel_size=3,
            padding="same",
            padding_mode="zeros",
        )

        #
        self.transform_net_states = TransformNet()
        self.transform_net_velocities = TransformNet()

    def forward(self, states: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, velocities], dim=1)

        # The reaction gets all features
        reaction = self.reaction(x)

        # Compute advection and diffusion, only numerical, no ML
        advection = self.advection(states, velocities)
        diffusion = self.diffusion(states)

        # Transform all features into final states derivatives wrt time
        states_dot = self.transform_net_states(reaction, mask=[advection, diffusion])
        vel_dot = self.transform_net_velocities(reaction, mask=[advection, diffusion])

        return torch.cat([states_dot, vel_dot], dim=1)
