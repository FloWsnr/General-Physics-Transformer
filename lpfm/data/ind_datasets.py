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
from lpfm.data.phys_dataset import PhysicsDataset


class ComsolIncompressibleFlowDataset(PhysicsDataset):
    def __init__(self, *args, **kwargs):
        include_field_names = {
            "t0_fields": ["pressure"],
            "t1_fields": ["velocity"],
        }

        super().__init__(include_field_names=include_field_names, *args, **kwargs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = super().__getitem__(index)  # (time, h, w, c)

        # create a zero tensor with the same shape as the input
        zero_tensor_x = torch.zeros_like(x[:, :, :, 0])
        zero_tensor_y = torch.zeros_like(y[:, :, :, 0])
        zero_tensor_x = zero_tensor_x.unsqueeze(-1)
        zero_tensor_y = zero_tensor_y.unsqueeze(-1)

        # density and temperature are not present in the input
        # so we need to add them at the correct position (1, 2)
        pressure_x = x[:, :, :, 0].unsqueeze(-1)
        x = torch.cat(
            [pressure_x, zero_tensor_x, zero_tensor_x, x[:, :, :, 1:]], dim=-1
        )

        pressure_y = y[:, :, :, 0].unsqueeze(-1)
        y = torch.cat(
            [pressure_y, zero_tensor_y, zero_tensor_y, y[:, :, :, 1:]], dim=-1
        )

        return x, y


class ComsolHeatedFlowDataset(PhysicsDataset):
    def __init__(self, *args, **kwargs):
        include_field_names = {
            "t0_fields": ["pressure", "density", "temperature"],
            "t1_fields": ["velocity"],
        }

        super().__init__(include_field_names=include_field_names, *args, **kwargs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = super().__getitem__(index)  # (time, h, w, c)
        return x, y


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
        buoyancy_x = x[:, :, :, 1]
        density_x = buoyancy_x.unsqueeze(-1)
        # use ideal gas law to get temperature
        pressure_x = x[:, :, :, 0].unsqueeze(-1)
        temperature_x = pressure_x / (density_x)

        # so we need to add density at position 1 (override buoyancy)
        # and then add temperature at position 2
        x = torch.cat(
            [pressure_x, density_x, temperature_x, x[:, :, :, 2:]],
            dim=-1,
        )

        buoyancy_y = y[:, :, :, 1]
        density_y = buoyancy_y.unsqueeze(-1)
        # use ideal gas law to get temperature
        pressure_y = y[:, :, :, 0].unsqueeze(-1)
        temperature_y = pressure_y / (density_y)

        y = torch.cat(
            [pressure_y, density_y, temperature_y, y[:, :, :, 2:]],
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
        zero_tensor_x = torch.zeros_like(x[:, :, :, 0])
        zero_tensor_y = torch.zeros_like(y[:, :, :, 0])
        zero_tensor_x = zero_tensor_x.unsqueeze(-1)
        zero_tensor_y = zero_tensor_y.unsqueeze(-1)

        # density and temperature are not present in the input
        # so we need to add them at the correct position (1, 2)
        pressure_x = x[:, :, :, 0].unsqueeze(-1)
        x = torch.cat(
            [pressure_x, zero_tensor_x, zero_tensor_x, x[:, :, :, 1:]], dim=-1
        )

        pressure_y = y[:, :, :, 0].unsqueeze(-1)
        y = torch.cat(
            [pressure_y, zero_tensor_y, zero_tensor_y, y[:, :, :, 1:]], dim=-1
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

        zero_tensor_x = torch.zeros_like(x[:, :, :, 1])
        zero_tensor_y = torch.zeros_like(y[:, :, :, 1])
        zero_tensor_x = zero_tensor_x.unsqueeze(-1)
        zero_tensor_y = zero_tensor_y.unsqueeze(-1)

        # temperature is not present in the input
        # so we need to add it

        x = torch.cat([x[:, :, :, :2], zero_tensor_x, x[:, :, :, 2:]], dim=-1)
        y = torch.cat([y[:, :, :, :2], zero_tensor_y, y[:, :, :, 2:]], dim=-1)
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
        zero_tensor_x = torch.zeros_like(x[:, :, :, 0])
        zero_tensor_y = torch.zeros_like(y[:, :, :, 0])
        zero_tensor_x = zero_tensor_x.unsqueeze(-1)
        zero_tensor_y = zero_tensor_y.unsqueeze(-1)

        x = torch.cat([x[:, :, :, :2], zero_tensor_x, x[:, :, :, 2:]], dim=-1)
        y = torch.cat([y[:, :, :, :2], zero_tensor_y, y[:, :, :, 2:]], dim=-1)
        return x, y
