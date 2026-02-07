from unittest.mock import patch, MagicMock
from gphyt.train.utils.wandb_logger import WandbLogger


class TestWandbLogger:
    @patch('gphyt.train.utils.wandb_logger.wandb')
    def test_wandb_logger_initialization_success(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        config = {
            "enabled": True,
            "project": "test_project",
            "entity": "test_entity",
            "id": "test_id",
            "tags": ["test"],
            "notes": "test notes"
        }

        logger = WandbLogger(config)

        assert logger.run is mock_run
        mock_wandb.login.assert_called_once()
        mock_wandb.init.assert_called_once()

    @patch('gphyt.train.utils.wandb_logger.wandb')
    def test_wandb_logger_initialization_failure(self, mock_wandb):
        mock_wandb.init.side_effect = Exception("Connection failed")

        config = {"enabled": True, "project": "test_project", "entity": "test_entity"}

        logger = WandbLogger(config)

        assert logger.run is None

    @patch('gphyt.train.utils.wandb_logger.wandb')
    def test_log_data_success(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger({"enabled": True, "project": "test", "entity": "test_entity"})

        data = {"metric1": 1.0, "metric2": 2.0}
        logger.log(data, folder="train")

        expected_data = {"train/metric1": 1.0, "train/metric2": 2.0}
        mock_run.log.assert_called_once_with(expected_data, commit=True)

    @patch('gphyt.train.utils.wandb_logger.wandb')
    def test_log_data_no_run(self, mock_wandb):
        mock_wandb.init.side_effect = Exception("Failed")

        logger = WandbLogger({"enabled": True, "project": "test", "entity": "test_entity"})
        logger.log({"metric": 1.0}, "train")

        # Should not raise error when run is None