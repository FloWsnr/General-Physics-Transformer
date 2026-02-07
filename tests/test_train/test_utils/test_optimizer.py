import pytest
import torch
import torch.nn as nn
from gphyt.train.utils.optimizer import get_optimizer


class TestGetOptimizer:
    def test_get_optimizer_adamw_basic(self):
        model = nn.Linear(10, 5)
        config = {
            "name": "AdamW",
            "learning_rate": 0.001
        }

        optimizer = get_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == 0
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)

    def test_get_optimizer_adamw_with_custom_params(self):
        model = nn.Linear(10, 5)
        config = {
            "name": "AdamW",
            "learning_rate": 0.01,
            "weight_decay": 0.1,
            "betas": (0.8, 0.99)
        }

        optimizer = get_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        assert optimizer.param_groups[0]["betas"] == (0.8, 0.99)

    def test_get_optimizer_sgd_basic(self):
        model = nn.Linear(10, 5)
        config = {
            "name": "SGD",
            "learning_rate": 0.001
        }

        optimizer = get_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["momentum"] == 0
        assert optimizer.param_groups[0]["weight_decay"] == 0

    def test_get_optimizer_adam_basic(self):
        model = nn.Linear(10, 5)
        config = {
            "name": "Adam",
            "learning_rate": 0.001
        }

        optimizer = get_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_get_optimizer_unsupported_name(self):
        model = nn.Linear(10, 5)
        config = {
            "name": "RMSprop",
            "learning_rate": 0.001
        }

        with pytest.raises(ValueError, match="Optimizer RMSprop not supported"):
            get_optimizer(model, config)