import pytest
import torch

from lpfm.run.train import get_lr_scheduler


@pytest.fixture
def model():
    return torch.nn.Linear(10, 10)


@pytest.fixture
def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=0.001)


def test_get_lr_scheduler_linear(optimizer: torch.optim.Optimizer):
    total_updates = 1000
    total_updates_trained = 0
    lrs_config = {
        "first_stage": {
            "name": "LinearLR",
            "start_factor": 0.001,
            "end_factor": 1.0,
            "num_updates": 100,
        },
    }
    scheduler = get_lr_scheduler(
        optimizer, lrs_config, total_updates, total_updates_trained
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)


def test_get_lr_scheduler_cosine(optimizer: torch.optim.Optimizer):
    total_updates = 1000
    total_updates_trained = 0
    lrs_config = {
        "first_stage": {
            "name": "CosineAnnealingLR",
            "end_factor": 0.01,
            "num_updates": 100,
        },
    }
    scheduler = get_lr_scheduler(
        optimizer, lrs_config, total_updates, total_updates_trained
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)


def test_get_lr_scheduler_two_stages(optimizer: torch.optim.Optimizer):
    total_updates = 1000
    total_updates_trained = 0
    lrs_config = {
        "first_stage": {
            "name": "LinearLR",
            "start_factor": 0.001,
            "end_factor": 1.0,
            "num_updates": 100,
        },
        "second_stage": {
            "name": "CosineAnnealingLR",
            "end_factor": 0.01,
            "num_updates": -1,
        },
    }
    scheduler = get_lr_scheduler(
        optimizer, lrs_config, total_updates, total_updates_trained
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
