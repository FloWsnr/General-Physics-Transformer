import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from gphyt.train.utils.run_utils import human_format, reduce_all_losses, compute_metrics


class TestHumanFormat:
    def test_human_format_small_numbers(self):
        assert human_format(123) == "123.00"
        assert human_format(999) == "999.00"

    def test_human_format_with_prefixes(self):
        assert human_format(1500) == "1.50K"
        assert human_format(1_500_000) == "1.50M"
        assert human_format(1_500_000_000) == "1.50B"
        assert human_format(1_500_000_000_000) == "1.50T"


class TestReduceAllLosses:
    @patch('torch.distributed.is_initialized')
    def test_reduce_all_losses_no_distributed(self, mock_dist_init):
        mock_dist_init.return_value = False

        losses = {
            'loss1': torch.tensor(1.0),
            'loss2': torch.tensor(2.0)
        }

        result = reduce_all_losses(losses)

        assert torch.equal(result['loss1'], torch.tensor(1.0))
        assert torch.equal(result['loss2'], torch.tensor(2.0))

    @patch('torch.distributed.all_reduce')
    @patch('torch.distributed.is_initialized')
    def test_reduce_all_losses_with_distributed(self, mock_dist_init, mock_all_reduce):
        mock_dist_init.return_value = True

        losses = {
            'loss1': torch.tensor(1.0),
            'loss2': torch.tensor(2.0)
        }

        result = reduce_all_losses(losses)

        assert mock_all_reduce.call_count == 2
        assert 'loss1' in result
        assert 'loss2' in result


class TestComputeMetrics:
    def test_compute_metrics_basic(self):
        x = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)

        mock_metric1 = MagicMock(return_value=torch.tensor(0.5))
        mock_metric2 = MagicMock(return_value=torch.tensor(0.8))

        metrics = {
            'metric1': mock_metric1,
            'metric2': mock_metric2
        }

        result = compute_metrics(x, target, metrics)

        assert 'metric1' in result
        assert 'metric2' in result
        assert torch.equal(result['metric1'], torch.tensor(0.5))
        assert torch.equal(result['metric2'], torch.tensor(0.8))
        mock_metric1.assert_called_once_with(x, target)
        mock_metric2.assert_called_once_with(x, target)