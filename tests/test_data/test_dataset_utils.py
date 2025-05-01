import torch
from pathlib import Path

from lpfm.data.dataset_utils import collate_fn, get_rng_transforms
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
