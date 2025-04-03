"""
A collection of individual "hardcoded" datasets.
This is necessary, because each dataset has different field names

All final outputs should include the following fields in that order:
- pressure
- density
- temperature
- velocity_x
- velocity_y

By: Florian Wiesner
Date: 2025-04-03
"""

import torch
from metaparc.data.phys_dataset import PhysicsDataset


class RayleighBenardDataset(PhysicsDataset):
    def __init__(self, *args, **kwargs):
        include_field_names = {
            "t0_fields": ["pressure", "buoyancy"],
            "t1_fields": ["velocity"],
        }

        super().__init__(include_field_names=include_field_names, *args, **kwargs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to get

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of input and target tensors
        """
        x, y = super().__getitem__(index)  # (time, h, w, c)

        # density and temperature are represented by buoyancy
        buoyancy = x[:, :, :, 1]
        density = buoyancy.unsqueeze(-1)
        # use ideal gas law to get temperature
        pressure = x[:, :, :, 0]
        temperature = pressure.unsqueeze(-1) / (density)

        # so we need to add density at position 1 (override buoyancy)
        # and then add temperature at position 2
        x = torch.cat(
            [x[:, :, :, :1], density, temperature, x[:, :, :, 2:]],
            dim=-1,
        )
        y = torch.cat(
            [y[:, :, :, :1], density, temperature, y[:, :, :, 2:]],
            dim=-1,
        )

        return x, y


class ShearFlowDataset(PhysicsDataset):
    def __init__(self, *args, **kwargs):
        include_field_names = {
            "t0_fields": ["pressure"],
            "t1_fields": ["velocity"],
        }

        super().__init__(include_field_names=include_field_names, *args, **kwargs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = super().__getitem__(index)  # (time, h, w, c)

        # in this dataset, the x axis is smaller than the y axis
        # so we need to swap the axes
        x = x.permute(0, 2, 1, 3)
        y = y.permute(0, 2, 1, 3)

        # then we need to swap the velocity components
        x_vel_x = x[:, :, :, 1]
        x_vel_y = x[:, :, :, 2]
        x[:, :, :, 1], x[:, :, :, 2] = x_vel_y, x_vel_x

        y_vel_x = y[:, :, :, 1]
        y_vel_y = y[:, :, :, 2]
        y[:, :, :, 1], y[:, :, :, 2] = y_vel_y, y_vel_x

        # create a zero tensor with the same shape as the input
        zero_tensor = torch.zeros_like(x[:, :, :, 0])
        zero_tensor = zero_tensor.unsqueeze(-1)

        # density and temperature are not present in the input
        # so we need to add them at the correct position (1, 2)

        x = torch.cat(
            [x[:, :, :, :1], zero_tensor, zero_tensor, x[:, :, :, 1:]], dim=-1
        )
        y = torch.cat(
            [y[:, :, :, :1], zero_tensor, zero_tensor, y[:, :, :, 1:]], dim=-1
        )

        return x, y


class TurbulentRadiativeDataset(PhysicsDataset):
    def __init__(self, *args, **kwargs):
        include_field_names = {
            "t0_fields": ["pressure", "density"],
            "t1_fields": ["velocity"],
        }

        super().__init__(include_field_names=include_field_names, *args, **kwargs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = super().__getitem__(index)  # (time, h, w, c)

        # in this dataset, the x axis is smaller than the y axis
        # so we need to swap the axes
        x = x.permute(0, 2, 1, 3)
        y = y.permute(0, 2, 1, 3)

        # then we need to swap the velocity components
        x_vel_x = x[:, :, :, 1]
        x_vel_y = x[:, :, :, 2]
        x[:, :, :, 1], x[:, :, :, 2] = x_vel_y, x_vel_x

        y_vel_x = y[:, :, :, 1]
        y_vel_y = y[:, :, :, 2]
        y[:, :, :, 1], y[:, :, :, 2] = y_vel_y, y_vel_x

        zero_tensor = torch.zeros_like(x[:, :, :, 1])
        zero_tensor = zero_tensor.unsqueeze(-1)

        # temperature is not present in the input
        # so we need to add it
        x = torch.cat([x[:, :, :, :2], zero_tensor, x[:, :, :, 2:]], dim=-1)
        y = torch.cat([y[:, :, :, :2], zero_tensor, y[:, :, :, 2:]], dim=-1)
        return x, y


class EulerDataset(PhysicsDataset):
    def __init__(self, *args, **kwargs):
        include_field_names = {
            "t0_fields": ["pressure", "density"],
            "t1_fields": ["momentum"],
        }

        super().__init__(include_field_names=include_field_names, *args, **kwargs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = super().__getitem__(index)  # (time, h, w, c)

        mom_x = x[:, :, :, 2]
        mom_y = x[:, :, :, 3]
        density = x[:, :, :, 1]

        # Convert momentum to velocity
        vel_x = mom_x / density
        vel_y = mom_y / density

        # Replace momentum with velocity
        x[:, :, :, 2], x[:, :, :, 3] = vel_x, vel_y
        y[:, :, :, 2], y[:, :, :, 3] = vel_x, vel_y

        # no temperature is available, so we add it
        zero_tensor = torch.zeros_like(x[:, :, :, 0])
        zero_tensor = zero_tensor.unsqueeze(-1)
        x = torch.cat([x[:, :, :, :2], zero_tensor, x[:, :, :, 2:]], dim=-1)
        y = torch.cat([y[:, :, :, :2], zero_tensor, y[:, :, :, 2:]], dim=-1)
        return x, y
