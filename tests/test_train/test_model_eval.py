"""Tests for the model evaluation module."""

from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import torch
from gphyt.train.model_eval import Evaluator


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Identity()

        def forward(self, x):
            return self.net(x)[:, -1, ...].unsqueeze(1)

    return MockModel()


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 10):
        super().__init__()
        self.length = length
        self.dataset_name = "cylinder_sym_flow_water"  # Use a valid dataset name from DATASET_FIELDS

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # T, H, W, C
        return (
            torch.randn(4, 10, 10, 5),
            torch.randn(1, 10, 10, 5),
        )

    def copy(self, overwrites: dict = None):
        return MockDatasetFullTrajectory(timesteps=self.length)


class MockDatasetFullTrajectory(torch.utils.data.Dataset):
    def __init__(self, length: int = 10, timesteps: int = 30):
        super().__init__()
        self.length = length
        self.timesteps = timesteps
        self.dataset_name = "cylinder_sym_flow_water"  # Use a valid dataset name from DATASET_FIELDS

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # T, H, W, C
        return (
            torch.randn(4, 10, 10, 5),
            torch.randn(self.timesteps, 10, 10, 5),
        )


def test_model_eval(tmp_path: Path, mock_model: torch.nn.Module):
    dataset = MockDataset()
    evaluator = Evaluator(mock_model, {"test": dataset}, tmp_path)
    losses_dict = evaluator._eval_on_dataset(dataset)
    assert isinstance(losses_dict, dict)
    assert "MSE" in losses_dict
    assert "VMSE" in losses_dict
    assert losses_dict["MSE"].shape == (10,)
    assert losses_dict["VMSE"].shape == (10,)


def test_eval_all(tmp_path: Path, mock_model: torch.nn.Module):
    """Test the eval_all method of the Evaluator class."""
    datasets = {
        "test": MockDataset(length=10),
        "test2": MockDataset(length=12),
    }
    evaluator = Evaluator(mock_model, datasets, tmp_path)
    result_dfs = evaluator.eval_all(datasets)

    # Check that we get a dictionary with both MSE and RVMSE DataFrames
    assert isinstance(result_dfs, dict)
    assert "MSE" in result_dfs
    assert "VMSE" in result_dfs

    # Check DataFrame shape and columns for each criterion
    for criterion_name, df in result_dfs.items():
        assert df.shape == (12, 2)
        assert df.columns.tolist() == ["test", "test2"]
        # Check that all values are non-negative (since they are losses), ignore nan
        assert np.all(df.fillna(0) >= 0)


def test_model_rollout(tmp_path: Path, mock_model: torch.nn.Module):
    dataset = MockDatasetFullTrajectory(timesteps=30)
    evaluator = Evaluator(mock_model, {"test": dataset}, tmp_path)
    outputs, full_traj, losses_dict = evaluator._rollout(dataset)
    assert outputs.shape == (30, 10, 10, 5)
    assert full_traj.shape == (30, 10, 10, 5)
    assert isinstance(losses_dict, dict)
    assert "MSE" in losses_dict
    assert "VMSE" in losses_dict
    # cylinder_sym_flow_water has fields (0, 3, 4) = 3 channels
    assert losses_dict["MSE"].shape == (30, 3)  # (T, C)
    assert losses_dict["VMSE"].shape == (30, 3)  # (T, C)


def test_model_rollout_all(tmp_path: Path, mock_model: torch.nn.Module):
    datasets = {
        "test": MockDataset(length=10),
        "test2": MockDataset(length=12),
        "test3": MockDataset(length=14),
    }
    evaluator = Evaluator(mock_model, datasets, tmp_path)
    result_dfs = evaluator.rollout_all(datasets)

    # Check that we get a dictionary with both MSE and RVMSE DataFrames
    assert isinstance(result_dfs, dict)
    assert "MSE" in result_dfs
    assert "VMSE" in result_dfs

    # Check DataFrame shape for each criterion - 3 metrics x 3 datasets x 3 channels
    for criterion_name, df in result_dfs.items():
        assert df.shape == (14, len(datasets) * 3 * 3)  # 3 metrics x 3 datasets x 3 channels


def test_model_eval_main(tmp_path: Path, mock_model: torch.nn.Module):
    datasets = {
        "test": MockDataset(length=10),
        "test2": MockDataset(length=12),
        "test3": MockDataset(length=14),
    }
    evaluator = Evaluator(mock_model, datasets, tmp_path)
    evaluator.main()

    # Check that loss files are created for each criterion
    assert (tmp_path / "mse_losses.csv").exists()
    assert (tmp_path / "vmse_losses.csv").exists()
    assert (tmp_path / "nmse_losses.csv").exists()

    df = pd.read_csv(tmp_path / "mse_losses.csv", header=0)
    assert df.shape == (14, len(datasets))
    assert df.columns.tolist() == ["test", "test2", "test3"]


def test_model_eval_visualize_rollout(tmp_path: Path, mock_model: torch.nn.Module):
    datasets = {
        "test": MockDataset(length=10),
        "test2": MockDataset(length=12),
        "test3": MockDataset(length=14),
    }
    evaluator = Evaluator(mock_model, datasets, tmp_path)
    evaluator.visualize_rollout(
        datasets["test"], num_timesteps=50, save_path=tmp_path / "images"
    )

    assert (tmp_path / "images" / "pressure_pred_t0.png").exists()
    assert (tmp_path / "images" / "pressure_gt_t0.png").exists()

    assert (tmp_path / "images" / "velocity_x_pred_t3.png").exists()
    assert (tmp_path / "images" / "velocity_x_gt_t3.png").exists()
