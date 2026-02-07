import pytest
import time
from unittest.mock import patch
from gphyt.train.utils.time_keeper import TimeKeeper, TimeEstimates


class TestTimeEstimates:
    def test_initialization(self):
        estimates = TimeEstimates(
            avg_sec_per_epoch=10.0,
            avg_sec_per_update_loop=1.0,
            avg_sec_per_val_loop=5.0,
        )
        assert estimates.avg_sec_per_epoch == 10.0
        assert estimates.avg_sec_per_update_loop == 1.0
        assert estimates.avg_sec_per_val_loop == 5.0


class TestTimeKeeper:
    def test_initialization_with_time_limit(self):
        keeper = TimeKeeper(time_limit=3600.0, global_rank=1)
        assert keeper.time_limit == 3600.0
        assert keeper.global_rank == 1
        assert isinstance(keeper.time_start, float)
        assert keeper.estimates.avg_sec_per_epoch == 0.0
        assert keeper.estimates.avg_sec_per_update_loop == 0.0
        assert keeper.estimates.avg_sec_per_val_loop == 0.0

    def test_initialization_without_time_limit(self):
        keeper = TimeKeeper(time_limit=None)
        assert keeper.time_limit is None
        assert keeper.global_rank == 0

    def test_update_estimate_epoch(self):
        keeper = TimeKeeper(time_limit=3600.0)
        keeper.update_estimate(duration=10.0, state_key="avg_sec_per_epoch", n_phases=0)
        assert keeper.estimates.avg_sec_per_epoch == 10.0

        keeper.update_estimate(duration=20.0, state_key="avg_sec_per_epoch", n_phases=1)
        assert keeper.estimates.avg_sec_per_epoch == 15.0

    def test_update_estimate_update_loop(self):
        keeper = TimeKeeper(time_limit=3600.0)
        keeper.update_estimate(
            duration=2.0, state_key="avg_sec_per_update_loop", n_phases=0
        )
        assert keeper.estimates.avg_sec_per_update_loop == 2.0

    def test_update_estimate_validation(self):
        keeper = TimeKeeper(time_limit=3600.0)
        keeper.update_estimate(
            duration=5.0, state_key="avg_sec_per_val_loop", n_phases=0
        )
        assert keeper.estimates.avg_sec_per_val_loop == 5.0

    def test_get_remaining_time_no_limit(self):
        keeper = TimeKeeper(time_limit=None)
        assert keeper.get_remaining_time() == float("inf")

    def test_get_remaining_time_with_limit(self):
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            keeper = TimeKeeper(time_limit=3600.0)
            keeper.time_start = 1000.0

            mock_time.return_value = 1010.0
            remaining = keeper.get_remaining_time()
            assert remaining == 3590.0

    def test_should_stop_training_no_limit(self):
        keeper = TimeKeeper(time_limit=None)
        assert not keeper.should_stop_training()

    def test_should_stop_training_sufficient_time(self):
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            keeper = TimeKeeper(time_limit=3600.0)
            keeper.time_start = 1000.0
            keeper.estimates.avg_sec_per_epoch = 100.0

            mock_time.return_value = 1010.0
            assert not keeper.should_stop_training()

    def test_should_stop_training_insufficient_time(self):
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            keeper = TimeKeeper(time_limit=200.0)
            keeper.time_start = 1000.0
            keeper.estimates.avg_sec_per_epoch = 100.0

            mock_time.return_value = 1100.0
            assert keeper.should_stop_training()

    def test_should_stop_update_loop_no_limit(self):
        keeper = TimeKeeper(time_limit=None)
        assert not keeper.should_stop_update_loop()

    def test_should_stop_update_loop_sufficient_time(self):
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            keeper = TimeKeeper(time_limit=3600.0)
            keeper.time_start = 1000.0
            keeper.estimates.avg_sec_per_update_loop = 10.0

            mock_time.return_value = 1010.0
            assert not keeper.should_stop_update_loop()

    def test_should_stop_update_loop_insufficient_time(self):
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            keeper = TimeKeeper(time_limit=50.0)
            keeper.time_start = 1000.0
            keeper.estimates.avg_sec_per_update_loop = 10.0

            mock_time.return_value = 1040.0
            assert keeper.should_stop_update_loop()

    def test_should_stop_validation_no_limit(self):
        keeper = TimeKeeper(time_limit=None)
        assert not keeper.should_stop_validation()

    def test_should_stop_validation_sufficient_time(self):
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            keeper = TimeKeeper(time_limit=3600.0)
            keeper.time_start = 1000.0
            keeper.estimates.avg_sec_per_val_loop = 30.0

            mock_time.return_value = 1010.0
            assert not keeper.should_stop_validation()

    def test_should_stop_validation_insufficient_time(self):
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            keeper = TimeKeeper(time_limit=100.0)
            keeper.time_start = 1000.0
            keeper.estimates.avg_sec_per_val_loop = 30.0

            mock_time.return_value = 1070.0
            assert keeper.should_stop_validation()

    def test_safety_margin_applied(self):
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            keeper = TimeKeeper(time_limit=100.0)
            keeper.time_start = 1000.0
            keeper.estimates.avg_sec_per_epoch = 100.0

            mock_time.return_value = 1000.0
            remaining_time = 100.0
            estimated_with_margin = 100.0 * 1.1
            assert remaining_time < estimated_with_margin
            assert keeper.should_stop_training()

    def test_multiple_update_estimate_calls(self):
        keeper = TimeKeeper(time_limit=3600.0)

        keeper.update_estimate(duration=10.0, state_key="avg_sec_per_epoch", n_phases=0)
        assert keeper.estimates.avg_sec_per_epoch == 10.0

        keeper.update_estimate(duration=12.0, state_key="avg_sec_per_epoch", n_phases=1)
        assert keeper.estimates.avg_sec_per_epoch == 11.0

        keeper.update_estimate(duration=8.0, state_key="avg_sec_per_epoch", n_phases=2)
        assert keeper.estimates.avg_sec_per_epoch == 10.0
