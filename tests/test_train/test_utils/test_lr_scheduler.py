import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from gphyt.train.utils.lr_scheduler import get_lr_scheduler


class TestGetLrScheduler:
    def test_get_lr_scheduler_single_linear_stage(self):
        model = nn.Linear(10, 5)
        optimizer = optim.AdamW(model.parameters(), lr=0.01)

        lrs_config = {
            "first_stage": {
                "name": "LinearLR",
                "num_updates": 100,
                "start_factor": 0.1,
                "end_factor": 1.0
            }
        }

        scheduler = get_lr_scheduler(optimizer, lrs_config, total_batches=100)

        assert isinstance(scheduler, optim.lr_scheduler.SequentialLR)

    def test_get_lr_scheduler_two_stages(self):
        model = nn.Linear(10, 5)
        optimizer = optim.AdamW(model.parameters(), lr=0.01)

        lrs_config = {
            "first_stage": {
                "name": "LinearLR",
                "num_updates": 50,
                "start_factor": 0.1,
                "end_factor": 1.0
            },
            "second_stage": {
                "name": "CosineAnnealingLR",
                "num_updates": 50,
                "end_factor": 0.1
            }
        }

        scheduler = get_lr_scheduler(optimizer, lrs_config, total_batches=100)

        assert isinstance(scheduler, optim.lr_scheduler.SequentialLR)

    def test_get_lr_scheduler_with_minus_one_milestone(self):
        model = nn.Linear(10, 5)
        optimizer = optim.AdamW(model.parameters(), lr=0.01)

        lrs_config = {
            "first_stage": {
                "name": "LinearLR",
                "num_updates": 20,
                "start_factor": 0.1,
                "end_factor": 1.0
            },
            "second_stage": {
                "name": "CosineAnnealingLR",
                "num_updates": -1,  # Should use remaining updates
                "end_factor": 0.1
            }
        }

        scheduler = get_lr_scheduler(optimizer, lrs_config, total_batches=100)

        assert isinstance(scheduler, optim.lr_scheduler.SequentialLR)

    def test_get_lr_scheduler_multiple_minus_one_raises_error(self):
        model = nn.Linear(10, 5)
        optimizer = optim.AdamW(model.parameters(), lr=0.01)

        lrs_config = {
            "first_stage": {
                "name": "LinearLR",
                "num_updates": -1,
                "start_factor": 0.1,
                "end_factor": 1.0
            },
            "second_stage": {
                "name": "CosineAnnealingLR",
                "num_updates": -1,
                "end_factor": 0.1
            }
        }

        with pytest.raises(ValueError, match="Only one milestone can be -1"):
            get_lr_scheduler(optimizer, lrs_config, total_batches=100)