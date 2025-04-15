import pytest
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lpfm.data.phys_dataset import PhysicsDataset, SuperDataset


def test_physics_dataset(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent)
    assert len(dataset) == 18
    x, y = dataset[0]
    assert x.shape == (1, 32, 32, 6)
    assert y.shape == (1, 32, 32, 6)


def test_physics_dataset_more_fields(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, n_steps_input=2, n_steps_output=2)
    x, y = dataset[0]
    assert x.shape == (2, 32, 32, 6)  # (time, height, width, channels)
    assert y.shape == (2, 32, 32, 6)


def test_physics_dataset_custom_field_selection(dummy_datapath: Path):
    dataset = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=2,
        n_steps_output=2,
        include_field_names={"t0_fields": ["variable_field1"]},
    )
    x, y = dataset[0]
    assert x.shape == (2, 32, 32, 1)
    assert y.shape == (2, 32, 32, 1)


def test_physics_dataset_custom_field_selection2(dummy_datapath: Path):
    dataset = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=2,
        n_steps_output=2,
        include_field_names={"t0_fields": ["variable_field1"], "t1_fields": ["field1"]},
    )
    x, y = dataset[0]
    assert x.shape == (2, 32, 32, 3)
    assert y.shape == (2, 32, 32, 3)


def test_physics_dataset_length_limit(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, length_limit=10)
    assert len(dataset) == 10
    x, y = dataset[0]
    assert x.shape == (1, 32, 32, 6)
    assert y.shape == (1, 32, 32, 6)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    assert len(dataloader) == 10


def test_super_dataset(dummy_datapath: Path):
    dataset1 = PhysicsDataset(dummy_datapath.parent, n_steps_input=1, n_steps_output=1)
    dataset2 = PhysicsDataset(dummy_datapath.parent, n_steps_input=1, n_steps_output=1)
    super_dataset = SuperDataset([dataset1, dataset2], (32, 16))
    assert len(super_dataset) == len(dataset1) + len(dataset2)
    x, y = super_dataset[0]
    assert x.shape == (1, 32, 16, 6)
    assert y.shape == (1, 32, 16, 6)
