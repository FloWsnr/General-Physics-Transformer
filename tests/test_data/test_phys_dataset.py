import pytest
from pathlib import Path

import torch
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


def test_physics_dataset_variable_dT_stride(dummy_datapath: Path):
    dataset = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=2,
        n_steps_output=2,
        dt_stride=[1, 4],
    )
    x, y = dataset[0]
    assert x.shape == (2, 32, 32, 6)
    assert y.shape == (2, 32, 32, 6)


class TestSuperDataset:
    """Tests for the SuperDataset class."""

    def test_super_dataset(self, dummy_datapath: Path):
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        super_dataset = SuperDataset(datasets)
        assert len(super_dataset) == len(dataset1) + len(dataset2)
        x, y = super_dataset[0]
        assert x.shape == (1, 32, 32, 6)
        assert y.shape == (1, 32, 32, 6)

    def test_random_sampling_length(self, dummy_datapath: Path):
        """Test that SuperDataset has correct length when using max_samples_per_ds."""
        # Create two identical datasets
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        # Create SuperDataset with max_samples_per_ds
        max_samples = 5
        super_dataset = SuperDataset(
            datasets,
            max_samples_per_ds=max_samples,
            seed=42,
        )

        # Check total length is correct
        assert len(super_dataset) == max_samples * 2

    def test_random_sampling_indices(self, dummy_datapath: Path):
        """Test that SuperDataset samples random indices rather than sequential ones."""
        # Create two identical datasets
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        # Create SuperDataset with max_samples_per_ds
        max_samples = 5
        super_dataset = SuperDataset(
            datasets,
            max_samples_per_ds=max_samples,
            seed=42,
        )

        # Get all samples from first dataset
        samples1 = []
        for i in range(max_samples):
            x, y = super_dataset[i]
            samples1.append(x)

        # Get all samples from second dataset
        samples2 = []
        for i in range(max_samples, max_samples * 2):
            x, y = super_dataset[i]
            samples2.append(x)

        # Verify we got different samples (not just first N)
        # Compare with original dataset's first N samples
        first_n_samples = [dataset1[i][0] for i in range(max_samples)]
        assert not all(torch.equal(s1, s2) for s1, s2 in zip(samples1, first_n_samples))

    def test_random_sampling_reproducibility(self, dummy_datapath: Path):
        """Test that SuperDataset produces the same samples with the same seed."""
        # Create two identical datasets
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )
        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        # Create SuperDataset with max_samples_per_ds
        max_samples = 5
        super_dataset = SuperDataset(
            datasets,
            max_samples_per_ds=max_samples,
            seed=42,
        )

        # Create another SuperDataset with the same seed
        super_dataset2 = SuperDataset(
            datasets,
            max_samples_per_ds=max_samples,
            seed=42,
        )

        # Verify reproducibility with same seed
        for i in range(len(super_dataset)):
            x1, y1 = super_dataset[i]
            x2, y2 = super_dataset2[i]
            assert torch.equal(x1, x2)
            assert torch.equal(y1, y2)

    def test_random_sampling_different_seeds(self, dummy_datapath: Path):
        """Test that SuperDataset produces different samples with different seeds."""
        # Create two identical datasets
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1
        )
        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        # Create SuperDataset with max_samples_per_ds
        max_samples = 5
        super_dataset = SuperDataset(
            datasets,
            max_samples_per_ds=max_samples,
            seed=42,
        )

        # Create another SuperDataset with a different seed
        super_dataset3 = SuperDataset(
            datasets,
            max_samples_per_ds=max_samples,
            seed=43,
        )

        # Verify different samples with different seed
        assert not all(
            torch.equal(super_dataset[i][0], super_dataset3[i][0])
            for i in range(len(super_dataset))
        )
