"""
Test the individual datasets
These tests only work if the datasets are present.
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from metaparc.data.ind_datasets import (
    RayleighBenardDataset,
    ShearFlowDataset,
    TurbulentRadiativeDataset,
    EulerDataset,
    ComsolIncompressibleFlowDataset,
    ComsolHeatedFlowDataset,
)


def test_rayleigh_benard_dataset():
    """Test RayleighBenardDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/rayleigh_benard/data/train")
    dataset = RayleighBenardDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, buoyancy]
    x, y = dataset[1]

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocities)
    assert x.shape == (1, 512, 128, 5)
    assert y.shape == (1, 512, 128, 5)

    # Verify we don't have NaN values (which could happen in temperature calculation)
    assert not torch.isnan(x).any()
    assert not torch.isnan(y).any()


def test_shear_flow_dataset():
    """Test ShearFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/shear_flow/data/train")
    dataset = ShearFlowDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, velocity]
    x, y = dataset[0]

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
    assert x.shape == (1, 512, 256, 5)
    assert y.shape == (1, 512, 256, 5)

    # check that the density and temperature are zero
    assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
    assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_turbulent_radiative_dataset():
    """Test TurbulentRadiativeDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/turbulent_radiative_layer_2D/data/train")
    dataset = TurbulentRadiativeDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, density, velocity]
    x, y = dataset[0]

    # Check shapes - should now have 4 channels (pressure, density, temperature, velocity)
    assert x.shape == (1, 384, 128, 5)
    assert y.shape == (1, 384, 128, 5)

    # check that the temperature is zero
    assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))
    assert torch.allclose(y[:, :, :, 2], torch.zeros_like(y[:, :, :, 2]))


def test_euler_dataset():
    """Test EulerDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/euler_multi_quadrants_periodicBC/data/train")
    dataset = EulerDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, density, velocity]
    x, y = dataset[0]

    # Check shapes - should now have 4 channels (pressure, density, temperature, velocity)
    assert x.shape == (1, 512, 512, 5)
    assert y.shape == (1, 512, 512, 5)

    # check that the temperature is zero
    assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))
    assert torch.allclose(y[:, :, :, 2], torch.zeros_like(y[:, :, :, 2]))


def test_cylinder_pipe_flow_water_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/cylinder_pipe_flow_water/data/train")
    dataset = ComsolIncompressibleFlowDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, velocity]
    x, y = dataset[0]

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
    assert x.shape == (1, 336, 128, 5)
    assert y.shape == (1, 336, 128, 5)

    # check that the density and temperature are zero
    assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
    assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_cylinder_pipe_flow_water_dataset_with_dataloader():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/cylinder_pipe_flow_water/data/train")
    dataset = ComsolIncompressibleFlowDataset(data_dir=path)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        x, y = batch
        break

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
    assert x.shape == (2, 1, 336, 128, 5)
    assert y.shape == (2, 1, 336, 128, 5)

    # check that the density and temperature are zero
    assert torch.allclose(x[..., 1], torch.zeros_like(x[..., 1]))
    assert torch.allclose(x[..., 2], torch.zeros_like(x[..., 2]))


def test_cylinder_sym_flow_water_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/cylinder_sym_flow_water/data/train")
    dataset = ComsolIncompressibleFlowDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, velocity]
    x, y = dataset[0]

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
    assert x.shape == (1, 336, 128, 5)
    assert y.shape == (1, 336, 128, 5)

    # check that the density and temperature are zero
    assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
    assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_object_periodic_flow_water_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/object_periodic_flow_water/data/train")
    dataset = ComsolIncompressibleFlowDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, velocity]
    x, y = dataset[0]

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
    assert x.shape == (1, 256, 128, 5)
    assert y.shape == (1, 256, 128, 5)

    # check that the density and temperature are zero
    assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
    assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_object_sym_flow_air_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/object_sym_flow_air/data/train")
    dataset = ComsolIncompressibleFlowDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, velocity]
    x, y = dataset[0]

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
    assert x.shape == (1, 256, 128, 5)
    assert y.shape == (1, 256, 128, 5)

    # check that the density and temperature are zero
    assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
    assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_object_sym_flow_water_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/object_sym_flow_water/data/train")
    dataset = ComsolIncompressibleFlowDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, velocity]
    x, y = dataset[0]

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
    assert x.shape == (1, 256, 128, 5)
    assert y.shape == (1, 256, 128, 5)

    # check that the density and temperature are zero
    assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
    assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_heated_flow_dataset():
    """Test ComsolHeatedFlowDataset returns correct tensor shapes and field order."""
    path = Path("data\datasets\heated_object_pipe_flow_air\data")
    dataset = ComsolHeatedFlowDataset(data_dir=path)

    # Call the method - parent returns (time, h, w, c) with c=[pressure, velocity]
    x, y = dataset[0]

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
    assert x.shape == (1, 256, 128, 5)
    assert y.shape == (1, 256, 128, 5)

    # check that the density and temperature are not zero
    assert not torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
    assert not torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))
