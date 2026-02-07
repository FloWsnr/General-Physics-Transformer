import pytest
from pathlib import Path

import torch
from gphyt.data.phys_dataset import PhysicsDataset, SuperDataset, zero_field_to_value


def test_zero_field_to_value():
    x = torch.zeros(2, 32, 32, 6)

    # make some channels non-zero
    x[..., :3] = 2.0
    x = zero_field_to_value(x, 1.0)
    # check that the non-zero channels are unchanged
    assert torch.all(x[..., :3] == 2.0)
    # check that the zero channels are replaced with 1.0
    assert torch.all(x[..., 3:] == 1.0)


def test_physics_dataset(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, use_normalization=False)
    assert len(dataset) == 18
    x, y = dataset[0]
    assert x.shape == (1, 32, 32, 6)
    assert y.shape == (1, 32, 32, 6)


def test_physics_dataset_more_fields(dummy_datapath: Path):
    dataset = PhysicsDataset(dummy_datapath.parent, n_steps_input=2, n_steps_output=2, use_normalization=False)
    x, y = dataset[0]
    assert x.shape == (2, 32, 32, 6)  # (time, height, width, channels)
    assert y.shape == (2, 32, 32, 6)


def test_physics_dataset_custom_field_selection(dummy_datapath: Path):
    dataset = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=2,
        n_steps_output=2,
        use_normalization=False,
    )
    x, y = dataset[0]
    # Default returns all fields
    assert x.ndim == 4  # (T, H, W, C)
    assert y.ndim == 4


def test_physics_dataset_custom_field_selection2(dummy_datapath: Path):
    dataset = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=2,
        n_steps_output=2,
        use_normalization=False,
    )
    x, y = dataset[0]
    assert x.ndim == 4
    assert y.ndim == 4


def test_physics_dataset_variable_dT_stride(dummy_datapath: Path):
    dataset = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=2,
        n_steps_output=2,
        dt_stride=[1, 4],
        use_normalization=False,
    )
    x, y = dataset[0]
    assert x.shape == (2, 32, 32, 6)
    assert y.shape == (2, 32, 32, 6)


def test_physics_dataset_nan_to_zero(dummy_datapath: Path):
    dataset = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=2,
        n_steps_output=2,
        nan_to_zero=True,
        use_normalization=False,
    )

    x, y = dataset[0]
    assert not torch.any(torch.isnan(x))
    assert not torch.any(torch.isnan(y))


def test_physics_dataset_copy(dummy_datapath: Path):
    """Test that PhysicsDataset.copy() creates a new dataset with correct parameters."""
    # Create original dataset
    original_dataset = PhysicsDataset(
        dummy_datapath.parent,
        n_steps_input=2,
        n_steps_output=2,
        dt_stride=[1, 4],
        nan_to_zero=True,
        use_normalization=False,
    )

    # Test copying without changes
    copied_dataset = original_dataset.copy()
    assert copied_dataset.config == original_dataset.config
    assert copied_dataset.data_path == original_dataset.data_path
    assert copied_dataset.nan_to_zero == original_dataset.nan_to_zero

    # Test copying with overwrites
    overwrites = {
        "n_steps_input": 3,
        "n_steps_output": 1,
        "nan_to_zero": False,
    }
    modified_dataset = original_dataset.copy(overwrites)
    assert modified_dataset.config["n_steps_input"] == 3
    assert modified_dataset.config["n_steps_output"] == 1
    assert modified_dataset.nan_to_zero is False
    assert modified_dataset.data_path == original_dataset.data_path

    # Verify the original dataset was not modified
    assert original_dataset.config["n_steps_input"] == 2
    assert original_dataset.config["n_steps_output"] == 2
    assert original_dataset.nan_to_zero is True


def test_normalize_data_basic():
    """Test basic normalization functionality."""
    dataset = PhysicsDataset.__new__(PhysicsDataset)

    # Create test tensors with known statistics
    x = torch.randn(2, 32, 32, 3)  # (time, height, width, channels)
    y = torch.randn(2, 32, 32, 3)

    # Store original statistics
    original_mean = x.mean(dim=(0, 1, 2), keepdim=True)
    original_std = x.std(dim=(0, 1, 2), keepdim=True)

    # Apply normalization
    x_norm, y_norm = dataset.normalize_data(x, y)

    # Check that x is properly normalized (should have mean ~0 and std ~1)
    assert torch.allclose(x_norm.mean(dim=(0, 1, 2)), torch.zeros(3), atol=1e-6)
    assert torch.allclose(x_norm.std(dim=(0, 1, 2)), torch.ones(3), atol=1e-6)

    # Check that y is normalized using the same statistics as x
    expected_y_norm = (y - original_mean) / (original_std + 1e-6)
    assert torch.allclose(y_norm, expected_y_norm, atol=1e-6)


def test_normalize_data_different_shapes():
    """Test normalization with different tensor shapes."""
    dataset = PhysicsDataset.__new__(PhysicsDataset)

    # Test with different time steps
    x = torch.randn(5, 16, 16, 4)  # (time, height, width, channels)
    y = torch.randn(5, 16, 16, 4)

    x_norm, y_norm = dataset.normalize_data(x, y)

    assert x_norm.shape == x.shape
    assert y_norm.shape == y.shape
    assert torch.allclose(x_norm.mean(dim=(0, 1, 2)), torch.zeros(4), atol=1e-6)
    assert torch.allclose(x_norm.std(dim=(0, 1, 2)), torch.ones(4), atol=1e-6)


def test_normalize_data_preserves_relative_relationships():
    """Test that normalization preserves relative relationships between x and y."""
    dataset = PhysicsDataset.__new__(PhysicsDataset)

    # Create x and y with a known relationship
    x = torch.randn(2, 32, 32, 3)
    y = x + 5.0  # y is just x shifted by 5

    x_norm, y_norm = dataset.normalize_data(x, y)

    # The difference between normalized x and y should be consistent
    diff = y_norm - x_norm
    # All differences should be the same (since y = x + constant)
    assert torch.allclose(diff, diff[0, 0, 0, :], atol=1e-6)


class TestSuperDataset:
    """Tests for the SuperDataset class."""

    def test_super_dataset(self, dummy_datapath: Path):
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
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
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
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
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
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
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
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
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
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

    def test_reshuffle_changes_samples(self, dummy_datapath: Path):
        """Test that reshuffle() changes the samples being returned."""
        # Create two identical datasets
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        max_samples = 5
        super_dataset = SuperDataset(
            datasets,
            max_samples_per_ds=max_samples,
            seed=42,
        )

        # Get initial samples
        initial_samples = []
        for i in range(len(super_dataset)):
            x, _ = super_dataset[i]
            initial_samples.append(x)

        # Reshuffle and get new samples
        super_dataset.reshuffle()
        new_samples = []
        for i in range(len(super_dataset)):
            x, _ = super_dataset[i]
            new_samples.append(x)

        # Verify we got different samples after reshuffle
        assert not all(
            torch.equal(s1, s2) for s1, s2 in zip(initial_samples, new_samples)
        )

    def test_reshuffle_with_same_seed(self, dummy_datapath: Path):
        """Test that reshuffle() with same seed produces same sequence."""
        # Create two identical datasets
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        max_samples = 5
        super_dataset = SuperDataset(
            datasets,
            max_samples_per_ds=max_samples,
            seed=42,
        )

        # Get samples after first reshuffle
        super_dataset.reshuffle()
        first_reshuffle_samples = []
        for i in range(len(super_dataset)):
            x, _ = super_dataset[i]
            first_reshuffle_samples.append(x)

        # Get samples after second reshuffle
        super_dataset.reshuffle()
        second_reshuffle_samples = []
        for i in range(len(super_dataset)):
            x, _ = super_dataset[i]
            second_reshuffle_samples.append(x)

        # Verify we got different samples between reshuffles
        assert not all(
            torch.equal(s1, s2)
            for s1, s2 in zip(first_reshuffle_samples, second_reshuffle_samples)
        )

    def test_reshuffle_without_max_samples(self, dummy_datapath: Path):
        """Test that reshuffle() works correctly when max_samples_per_ds is None."""
        # Create two identical datasets
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        super_dataset = SuperDataset(
            datasets,
            max_samples_per_ds=None,
            seed=42,
        )

        # Get initial samples
        initial_samples = []
        for i in range(len(super_dataset)):
            x, _ = super_dataset[i]
            initial_samples.append(x)

        # Reshuffle and get new samples
        super_dataset.reshuffle()
        new_samples = []
        for i in range(len(super_dataset)):
            x, _ = super_dataset[i]
            new_samples.append(x)

        # Verify we got the same samples (since max_samples_per_ds is None)
        assert all(torch.equal(s1, s2) for s1, s2 in zip(initial_samples, new_samples))

    def test_super_dataset_max_samples_per_ds_list(self, dummy_datapath: Path):
        """Test that SuperDataset works with a list of max_samples_per_ds."""
        # Create two identical datasets
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        max_samples_per_ds = [5, 10]
        super_dataset = SuperDataset(datasets, max_samples_per_ds=max_samples_per_ds)
        assert len(super_dataset) == sum(max_samples_per_ds)

        for i in range(len(super_dataset)):
            x, _ = super_dataset[i]
            assert x.shape == (1, 32, 32, 6)

    def test_return_ds_idx_false(self, dummy_datapath: Path):
        """Test that SuperDataset returns only x, y when return_ds_idx=False."""
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        super_dataset = SuperDataset(datasets, return_ds_idx=False)

        # Get a sample and verify it returns only x, y (2 elements)
        result = super_dataset[0]
        assert len(result) == 2
        x, y = result
        assert x.shape == (1, 32, 32, 6)
        assert y.shape == (1, 32, 32, 6)

    def test_return_ds_idx_true(self, dummy_datapath: Path):
        """Test that SuperDataset returns x, y, ds_idx when return_ds_idx=True."""
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        super_dataset = SuperDataset(datasets, return_ds_idx=True)

        # Test samples from first dataset (indices 0 to len(dataset1)-1)
        dataset1_len = len(dataset1)
        for i in range(min(5, dataset1_len)):  # Test first 5 or all if less than 5
            result = super_dataset[i]
            assert len(result) == 3
            x, y, ds_idx = result
            assert x.shape == (1, 32, 32, 6)
            assert y.shape == (1, 32, 32, 6)
            assert ds_idx == 0  # Should be dataset index 0 (first dataset)

        # Test samples from second dataset (indices len(dataset1) to end)
        dataset2_start_idx = dataset1_len
        for i in range(
            dataset2_start_idx, min(dataset2_start_idx + 5, len(super_dataset))
        ):
            result = super_dataset[i]
            assert len(result) == 3
            x, y, ds_idx = result
            assert x.shape == (1, 32, 32, 6)
            assert y.shape == (1, 32, 32, 6)
            assert ds_idx == 1  # Should be dataset index 1 (second dataset)

    def test_return_ds_idx_with_max_samples(self, dummy_datapath: Path):
        """Test that return_ds_idx works correctly with max_samples_per_ds."""
        dataset1 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )
        dataset2 = PhysicsDataset(
            dummy_datapath.parent, n_steps_input=1, n_steps_output=1, use_normalization=False
        )

        datasets = {"dataset1": dataset1, "dataset2": dataset2}
        max_samples = 3
        super_dataset = SuperDataset(
            datasets, max_samples_per_ds=max_samples, return_ds_idx=True, seed=42
        )

        # Test first max_samples indices should come from dataset1 (ds_idx=0)
        for i in range(max_samples):
            result = super_dataset[i]
            assert len(result) == 3
            x, y, ds_idx = result
            assert ds_idx == 0

        # Test next max_samples indices should come from dataset2 (ds_idx=1)
        for i in range(max_samples, max_samples * 2):
            result = super_dataset[i]
            assert len(result) == 3
            x, y, ds_idx = result
            assert ds_idx == 1
