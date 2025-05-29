"""Tests for the model evaluation module."""

from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import torch
from lpfm.run.model_eval import Evaluator


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
    losses = evaluator._eval_on_dataset(dataset)
    assert losses.shape == (10,)


def test_eval_all(tmp_path: Path, mock_model: torch.nn.Module):
    """Test the eval_all method of the Evaluator class."""
    datasets = {
        "test": MockDataset(length=10),
        "test2": MockDataset(length=12),
    }
    evaluator = Evaluator(mock_model, datasets, tmp_path)
    df = evaluator.eval_all(datasets)

    # Check DataFrame shape and columns
    assert df.shape == (12, 2)
    assert df.columns.tolist() == ["test", "test2"]

    # Check that all values are non-negative (since they are MSE losses), ignore nan
    assert np.all(df.fillna(0) >= 0)


def test_model_rollout(tmp_path: Path, mock_model: torch.nn.Module):
    dataset = MockDatasetFullTrajectory(timesteps=30)
    evaluator = Evaluator(mock_model, {"test": dataset}, tmp_path)
    outputs, full_traj, loss = evaluator._rollout(dataset)
    assert outputs.shape == (30, 10, 10, 5)
    assert full_traj.shape == (30, 10, 10, 5)
    assert loss.shape == (30, 5)


def test_model_rollout_all(tmp_path: Path, mock_model: torch.nn.Module):
    datasets = {
        "test": MockDataset(length=10),
        "test2": MockDataset(length=12),
        "test3": MockDataset(length=14),
    }
    evaluator = Evaluator(mock_model, datasets, tmp_path)
    df = evaluator.rollout_all(datasets)
    assert df.shape == (14, len(datasets) * 3 * 5)  # 3 metrics, 5 channels


def test_model_eval_main(tmp_path: Path, mock_model: torch.nn.Module):
    datasets = {
        "test": MockDataset(length=10),
        "test2": MockDataset(length=12),
        "test3": MockDataset(length=14),
    }
    evaluator = Evaluator(mock_model, datasets, tmp_path)
    evaluator.main()

    df = pd.read_csv(tmp_path / "losses.csv", header=0)
    assert df.shape == (14, len(datasets))
    assert df.columns.tolist() == ["test", "test2", "test3"]

    df = pd.read_csv(tmp_path / "rollout_losses.csv", header=[0, 1, 2])
    assert df.shape == (14, len(datasets) * 3 * 5)


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
