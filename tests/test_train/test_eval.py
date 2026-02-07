from pathlib import Path
import pytest

import torch
from torch.utils.data import DataLoader

from gphyt.train.eval import Evaluator


@pytest.fixture
def model():
    return torch.nn.Identity()


@pytest.fixture
def metrics():
    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    return {
        "mse": mse,
        "mae": mae,
    }


@pytest.fixture
def real_dataloader() -> DataLoader:
    """Create a real PyTorch DataLoader for testing."""
    # Create dummy data in the format expected by trainer
    input_data = torch.randn(4, 10, 10)
    target_data = torch.randn(4, 10, 10)

    # Create dataset with proper format
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return (self.inputs[idx], self.targets[idx])

    dataset = TestDataset(input_data, target_data)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_eval(
    real_dataloader: DataLoader,
    model: torch.nn.Module,
    tmp_path: Path,
    metrics: dict[str, torch.nn.Module],
):
    evaluator = Evaluator(
        model=model,
        dataloader=real_dataloader,
        metrics=metrics,
        eval_dir=tmp_path,
    )
    losses = evaluator.eval()

    for metric_name, metric_value in losses.items():
        assert metric_value.item() != 0.0
