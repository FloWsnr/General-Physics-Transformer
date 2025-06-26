"""
Test the individual datasets
These tests only work if the datasets are present.
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from gphyt.data.phys_dataset import PhysicsDataset
import random


def test_rayleigh_benard_dataset():
    """Test RayleighBenardDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/rayleigh_benard/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocities)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # Verify we don't have NaN values (which could happen in temperature calculation)
        assert not torch.isnan(x).any()
        assert not torch.isnan(y).any()


def test_shear_flow_dataset():
    """Test ShearFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/shear_flow/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density and temperature are zero
        assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_turbulent_radiative_dataset():
    """Test TurbulentRadiativeDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/turbulent_radiative_layer_2D/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 4 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the temperature is zero
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))
        assert torch.allclose(y[:, :, :, 2], torch.zeros_like(y[:, :, :, 2]))


def test_euler_dataset():
    """Test EulerDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/euler_multi_quadrants_periodicBC/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 4 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the temperature is zero
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))
        assert torch.allclose(y[:, :, :, 2], torch.zeros_like(y[:, :, :, 2]))


def test_cylinder_pipe_flow_water_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/cylinder_pipe_flow_water/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density and temperature are zero
        assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_cylinder_pipe_flow_water_dataset_with_dataloader():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/cylinder_pipe_flow_water/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        x, y = batch
        break

    # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
    assert x.shape == (2, 1, 256, 128, 5)
    assert y.shape == (2, 1, 256, 128, 5)

    # check that the density and temperature are zero
    assert torch.allclose(x[..., 1], torch.zeros_like(x[..., 1]))
    assert torch.allclose(x[..., 2], torch.zeros_like(x[..., 2]))


def test_cylinder_sym_flow_water_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/cylinder_sym_flow_water/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density and temperature are zero
        assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_object_periodic_flow_water_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/object_periodic_flow_water/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density and temperature are zero
        assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_object_sym_flow_air_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/object_sym_flow_air/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density and temperature are zero
        assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_object_sym_flow_water_dataset():
    """Test ComsolIncompressibleFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/object_sym_flow_water/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density and temperature are zero
        assert torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_heated_flow_dataset():
    """Test ComsolHeatedFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/heated_object_pipe_flow_air/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density and temperature are not zero
        assert not torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert not torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_cooled_flow_dataset():
    """Test ComsolHeatedFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/cooled_object_pipe_flow_air/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density and temperature are not zero
        assert not torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert not torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_rayleigh_benard_obs_dataset():
    """Test RayleighBenardDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/rayleigh_benard_obstacle/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density and temperature are not zero
        assert not torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert not torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


def test_twophase_flow_dataset():
    """Test TwophaseFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/twophase_flow/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the density is not zero and temperature is zero
        assert not torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


@pytest.mark.skip(reason="Takes too long to run")
def test_twophase_flow_dataset_full_integrity():
    """Test TwophaseFlowDataset returns correct tensor shapes and field order."""
    path = Path("data/datasets/twophase_flow/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for i in range(len(dataset)):
        print(f"Testing index {i}")
        x, y = dataset[i]
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)


def test_supersonic_flow_dataset():
    """Test SupersonicFlowDataset returns correct tensor shapes and field order."""
    path = Path("/scratch/zsa8rk/datasets/supersonic_flow/data/train")
    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }
    dataset = PhysicsDataset(data_dir=path, include_field_names=include_field_names)

    # Test 4 random indices
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]

        # Check shapes - should now have 5 channels (pressure, density, temperature, velocity)
        assert x.shape == (1, 256, 128, 5)
        assert y.shape == (1, 256, 128, 5)

        # check that the pressure and density is not zero
        assert not torch.allclose(x[:, :, :, 0], torch.zeros_like(x[:, :, :, 0]))
        assert not torch.allclose(x[:, :, :, 1], torch.zeros_like(x[:, :, :, 1]))
        # check that the temperature is zero
        assert torch.allclose(x[:, :, :, 2], torch.zeros_like(x[:, :, :, 2]))


if __name__ == "__main__":
    test_twophase_flow_dataset_full_integrity()
