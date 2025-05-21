"""Tests for the model evaluation module."""

from pathlib import Path

import pandas as pd
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
            return self.net(x)[:, -1, ...]

    return MockModel()


class MockDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 10

    def __getitem__(self, index):
        # T, H, W, C
        return (
            torch.randn(4, 10, 10, 5),
            torch.randn(1, 10, 10, 5),
        )


def test_model_eval(tmp_path: Path, mock_model: torch.nn.Module):
    dataset = MockDataset()
    evaluator = Evaluator(mock_model, {"test": dataset}, tmp_path)
    losses = evaluator._eval_on_dataset(dataset)
    assert losses.shape == (10,)


def test_model_eval_main(tmp_path: Path, mock_model: torch.nn.Module):
    datasets = {
        "test": MockDataset(),
        "test2": MockDataset(),
    }
    evaluator = Evaluator(mock_model, datasets, tmp_path)
    evaluator.main()

    df = pd.read_csv(tmp_path / "losses.csv", header=0)
    assert df.shape == (10, 2)
    assert df.columns.tolist() == ["test", "test2"]
