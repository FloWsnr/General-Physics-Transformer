import torch
import torch.nn as nn

from metaparc.model.base_models.parc.advection import Advection
from metaparc.model.base_models.parc.diffusion import Diffusion
from metaparc.model.base_models.parc.num_differentiation import FiniteDifference
from metaparc.model.base_models.parc.reaction import ReactionNet
from metaparc.model.base_models.parc.transform import TransformNet


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
        self.reaction = ReactionNet(
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
            padding_mode="zeros",
            up_block_use_concat=[
                False,
                True,
                False,
                True,
            ],  # Control whether to use concatenation in upsampling blocks
            skip_connection_indices=[
                2,
                0,
            ],  # Indices of skip connections for UNet (residual connections)
        )

        self.transform_net = TransformNet()

    def forward(self, states: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, velocities], dim=1)

        # The reaction gets all features
        reaction = self.reaction(x)

        # Compute advection and diffusion
        advection = self.advection(states, velocities)
        diffusion = self.diffusion(states)

        # Transform all features into final states derivatives wrt time
        states_dot = self.transform_net(advection, diffusion, reaction)

        return states_dot
