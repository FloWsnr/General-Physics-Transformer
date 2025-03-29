import pytest
from pathlib import Path

import torch

from metaparc.data.datasets import (
    PhysicsDataset,
    collate_fn,
    get_dataloader,
    SuperDataloader,
    SuperDataset,
    get_rng_transforms,
)


def test_physics_dataset(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent)
    assert len(dataset) == 18
    x, y = dataset[0]
    assert x.shape == (1, 2, 32, 32)
    assert y.shape == (1, 2, 32, 32)


def test_physics_dataset_more_fields(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, n_steps_input=2, n_steps_output=2)
    x, y = dataset[0]
    assert x.shape == (2, 2, 32, 32)
    assert y.shape == (2, 2, 32, 32)


def test_physics_dataset_collate_fn(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, n_steps_input=2, n_steps_output=2)
    batch = [dataset[0], dataset[1]]
    collated = collate_fn(batch)
    assert collated[0].shape == (2, 4, 32, 32)
    assert collated[1].shape == (2, 4, 32, 32)


def test_dataloader_with_collate_fn(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, n_steps_input=2, n_steps_output=2)
    dataloader = get_dataloader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(dataloader))
    assert batch[0].shape == (2, 4, 32, 32)
    assert batch[1].shape == (2, 4, 32, 32)


def test_super_dataloader(dummy_datapath: Path):
    # Create two datasets with different parameters
    dataset1 = PhysicsDataset(dummy_datapath.parent, n_steps_input=1, n_steps_output=1)
    dataset2 = PhysicsDataset(dummy_datapath.parent, n_steps_input=2, n_steps_output=2)

    # Create dataloaders for each dataset
    dataloader1 = get_dataloader(dataset1, batch_size=2, num_workers=0)
    dataloader2 = get_dataloader(dataset2, batch_size=2, num_workers=0)

    # Create a SuperDataloader with both dataloaders
    super_dataloader = SuperDataloader([dataloader1, dataloader2])

    # Check the length is the sum of the individual dataloader lengths
    assert len(super_dataloader) == len(dataloader1) + len(dataloader2)

    # Iterate through the SuperDataloader and check we get batches
    batch_count = 0
    for batch in super_dataloader:
        batch_count += 1

        # The shape should be either from dataloader1 or dataloader2
        input_shape = batch[0].shape
        assert input_shape[0] == 2  # batch size
        assert input_shape[1] == 2 or input_shape[1] == 4  # channels
        assert input_shape[2:] == (32, 32)  # spatial dimensions

    # Verify we got the expected number of batches
    assert batch_count == len(super_dataloader)


def test_super_dataset(dummy_datapath: Path):
    dataset1 = PhysicsDataset(dummy_datapath.parent, n_steps_input=1, n_steps_output=1)
    dataset2 = PhysicsDataset(dummy_datapath.parent, n_steps_input=1, n_steps_output=1)
    super_dataset = SuperDataset([dataset1, dataset2], (32, 16), n_channels=2)
    assert len(super_dataset) == len(dataset1) + len(dataset2)
    x, y = super_dataset[0]
    assert x.shape == (1, 2, 32, 16)
    assert y.shape == (1, 2, 32, 16)


def test_super_dataset_with_n_channels(dummy_datapath: Path):
    dataset1 = PhysicsDataset(dummy_datapath.parent, n_steps_input=1, n_steps_output=1)
    dataset2 = PhysicsDataset(dummy_datapath.parent, n_steps_input=1, n_steps_output=1)
    super_dataset = SuperDataset([dataset1, dataset2], (32, 16), n_channels=3)

    x, y = super_dataset[0]
    assert x.shape == (1, 3, 32, 16)
    assert y.shape == (1, 3, 32, 16)


def test_super_dataset_with_dataloader(dummy_datapath: Path):
    dataset1 = PhysicsDataset(dummy_datapath.parent, n_steps_input=1, n_steps_output=1)
    dataset2 = PhysicsDataset(dummy_datapath.parent, n_steps_input=1, n_steps_output=1)
    super_dataset = SuperDataset([dataset1, dataset2], (32, 16), n_channels=3)
    dataloader = get_dataloader(super_dataset, batch_size=2, num_workers=0)
    batch = next(iter(dataloader))
    assert batch[0].shape == (2, 3, 32, 16)
    assert batch[1].shape == (2, 3, 32, 16)


def test_rng_transforms(dummy_datapath: Path):
    # Create a dataset
    dataset_t = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=1,
        n_steps_output=1,
        transform=get_rng_transforms(p_flip=1.0),
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
