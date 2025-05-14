import torch
import pytest
from pathlib import Path
from typing import Callable

from lpfm.data.dataset_utils import (
    collate_fn,
    get_rng_transforms,
    get_datasets,
    get_dt_datasets,
    get_dataloader,
)
from lpfm.data.phys_dataset import PhysicsDataset


def test_physics_dataset_collate_fn(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, n_steps_input=4, n_steps_output=4)
    batch = [dataset[0], dataset[1]]
    collated = collate_fn(batch)
    assert collated[0].shape == (2, 4, 32, 32, 6)
    assert collated[1].shape == (2, 4, 32, 32, 6)


def test_rng_transforms(dummy_datapath: Path):
    # Create a dataset
    dataset_t = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=1,
        n_steps_output=1,
        transform=get_rng_transforms(p_flip=0.9),
    )

    dataset_nt = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=1,
        n_steps_output=1,
        transform=None,
    )

    # Get a sample from the dataset
    sample_t = dataset_t[0]
    sample_nt = dataset_nt[0]
    input_fields_t = sample_t[0]
    input_fields_nt = sample_nt[0]

    # Check that the transformed data has the same shape
    assert input_fields_t.shape == input_fields_nt.shape

    # Check that the transform did something (data should be different)
    # Note: There's a very small chance this could fail if the random transform
    # happens to return the exact same data
    assert not torch.allclose(input_fields_t, input_fields_nt)


def test_get_datasets(tmp_path: Path, write_dummy_data: Callable[[Path], None]):
    """Test the get_datasets function.

    Parameters
    ----------
    tmp_path : Path
        Temporary path for test data
    write_dummy_data : Callable[[Path], None]
        Factory function for writing dummy data
    """
    # Create test data in train and valid directories
    write_dummy_data(tmp_path / "dummy_1/data/train/dummy_dataset.hdf5")
    write_dummy_data(tmp_path / "dummy_2/data/train/dummy_dataset.hdf5")

    data_config = {
        "data_dir": tmp_path,
        "n_steps_input": 4,
        "n_steps_output": 1,
        "dt_stride": [1, 2],
        "full_trajectory_mode": False,
        "max_rollout_steps": 10000,
        "zero_field_value": 0.0,
        "use_normalization": False,
        "nan_to_zero": True,
        "datasets": ["dummy_1", "dummy_2"],
    }

    # Test getting datasets for different splits
    train_datasets = get_datasets(data_config, split="train")

    # Check that we get the expected number of datasets
    assert len(train_datasets) == 2

    # Check that the datasets are PhysicsDataset instances
    assert isinstance(train_datasets["dummy_1"], PhysicsDataset)
    assert isinstance(train_datasets["dummy_2"], PhysicsDataset)


def test_get_dt_datasets(tmp_path: Path, write_dummy_data: Callable):
    """Test the get_dt_datasets function."""
    # Create test data in train and valid directories
    write_dummy_data(tmp_path / "dummy_1/data/train/dummy_dataset.hdf5")
    write_dummy_data(tmp_path / "dummy_2/data/train/dummy_dataset.hdf5")

    data_config = {
        "data_dir": tmp_path,
        "n_steps_input": 4,
        "n_steps_output": 1,
        "dt_stride": [1, 2],
        "full_trajectory_mode": False,
        "max_rollout_steps": 10000,
        "zero_field_value": 0.0,
        "use_normalization": False,
        "nan_to_zero": True,
        "datasets": ["dummy_1", "dummy_2"],
    }

    dt_datasets = get_dt_datasets(data_config, split="train")
    assert len(dt_datasets) == 2
    assert isinstance(dt_datasets[1]["dummy_1"], PhysicsDataset)
    assert isinstance(dt_datasets[2]["dummy_1"], PhysicsDataset)


@pytest.mark.parametrize("shuffle", [True, False])
def test_get_dataloader(tmp_path: Path, write_dummy_data: Callable, shuffle: bool):
    """Test the get_dataloader function."""
    # Create test data in train and valid directories
    write_dummy_data(tmp_path / "dummy_1/data/train/dummy_dataset.hdf5")
    write_dummy_data(tmp_path / "dummy_2/data/train/dummy_dataset.hdf5")

    data_config = {
        "data_dir": tmp_path,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "dt_stride": [1, 2],
        "full_trajectory_mode": False,
        "max_rollout_steps": 10000,
        "zero_field_value": 0.0,
        "use_normalization": False,
        "nan_to_zero": True,
        "datasets": ["dummy_1", "dummy_2"],
    }
    dataloader = get_dataloader(
        data_config,
        seed=42,
        batch_size=1,
        num_workers=0,
        prefetch_factor=None,
        split="train",
        shuffle=shuffle,
    )
    assert len(dataloader) == 68

@pytest.mark.parametrize("shuffle", [True, False])
def test_get_dataloader_data_fraction(tmp_path: Path, write_dummy_data: Callable, shuffle: bool):
    """Test the get_dataloader function."""
    # Create test data in train and valid directories
    write_dummy_data(tmp_path / "dummy_1/data/valid/dummy_dataset.hdf5")
    write_dummy_data(tmp_path / "dummy_2/data/valid/dummy_dataset.hdf5")

    data_config = {
        "data_dir": tmp_path,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "dt_stride": [1, 2],
        "full_trajectory_mode": False,
        "max_rollout_steps": 10000,
        "zero_field_value": 0.0,
        "use_normalization": False,
        "nan_to_zero": True,
        "datasets": ["dummy_1", "dummy_2"],
    }
    dataloader = get_dataloader(
        data_config,
        seed=42,
        batch_size=1,
        num_workers=0,
        prefetch_factor=None,
        split="val",
        shuffle=shuffle,
        data_fraction=0.5,
    )
    assert len(dataloader) == 34
