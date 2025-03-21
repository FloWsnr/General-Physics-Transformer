import pytest
from pathlib import Path
from torch.utils.data import DataLoader
from metaparc.data.datasets import PhysicsDataset, collate_fn, get_dataloader


def test_physics_dataset(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent)
    assert len(dataset) == 18
    assert dataset[0]["input_fields"].shape == (1, 32, 32, 2)
    assert dataset[0]["output_fields"].shape == (1, 32, 32, 2)


def test_physics_dataset_more_fields(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, n_steps_input=2, n_steps_output=2)
    assert dataset[0]["input_fields"].shape == (2, 32, 32, 2)
    assert dataset[0]["output_fields"].shape == (2, 32, 32, 2)


def test_physics_dataset_collate_fn(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, n_steps_input=2, n_steps_output=2)
    batch = [dataset[0], dataset[1]]
    collated = collate_fn(batch)
    assert collated["input_fields"].shape == (2, 4, 32, 32)
    assert collated["output_fields"].shape == (2, 4, 32, 32)


def test_dataloader_with_collate_fn(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, n_steps_input=2, n_steps_output=2)
    dataloader = get_dataloader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(dataloader))
    assert batch["input_fields"].shape == (2, 4, 32, 32)
    assert batch["output_fields"].shape == (2, 4, 32, 32)
