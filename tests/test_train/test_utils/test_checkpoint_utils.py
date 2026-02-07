import pytest
import torch
import torch.nn as nn
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from gphyt.train.utils.checkpoint_utils import sanitize_model_dict, load_checkpoint, save_checkpoint, find_checkpoint


class TestSanitizeModelDict:
    def test_sanitize_model_dict_basic(self):
        model = nn.Linear(5, 3)
        original_dict = model.state_dict()

        sanitized = sanitize_model_dict(model)

        assert len(sanitized) == len(original_dict)
        for key in original_dict.keys():
            assert key in sanitized

    def test_sanitize_model_dict_with_module_prefix(self):
        model = nn.Linear(5, 3)
        # Simulate what happens with nn.DataParallel
        modified_state = {}
        for key, value in model.state_dict().items():
            modified_state[f"module.{key}"] = value

        # Mock the state_dict to return modified keys
        with patch.object(model, 'state_dict', return_value=modified_state):
            sanitized = sanitize_model_dict(model)

            for key in sanitized.keys():
                assert not key.startswith("module.")

    def test_sanitize_model_dict_with_orig_mod_prefix(self):
        model = nn.Linear(5, 3)
        # Simulate what happens with torch.compile
        modified_state = {}
        for key, value in model.state_dict().items():
            modified_state[f"_orig_mod.{key}"] = value

        with patch.object(model, 'state_dict', return_value=modified_state):
            sanitized = sanitize_model_dict(model)

            for key in sanitized.keys():
                assert not key.startswith("_orig_mod.")


class TestLoadCheckpoint:
    def test_load_checkpoint(self):
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            test_data = {"epoch": 5, "loss": 0.1}
            torch.save(test_data, checkpoint_path)

            loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))

            assert loaded["epoch"] == 5
            assert loaded["loss"] == 0.1


class TestSaveCheckpoint:
    def test_save_checkpoint(self):
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"

            model = nn.Linear(5, 3)
            optimizer = torch.optim.Adam(model.parameters())
            grad_scaler = torch.amp.grad_scaler.GradScaler()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

            save_checkpoint(
                global_rank=0,
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                samples_trained=1000,
                batches_trained=100,
                epoch=5,
                grad_scaler=grad_scaler,
                scheduler=scheduler
            )

            assert checkpoint_path.exists()

            loaded = torch.load(checkpoint_path, weights_only=False)
            assert loaded["samples_trained"] == 1000
            assert loaded["batches_trained"] == 100
            assert loaded["epoch"] == 5
            assert "model_state_dict" in loaded
            assert "optimizer_state_dict" in loaded


class TestFindCheckpoint:
    def test_find_checkpoint_last_checkpoint(self, tmp_path):
        """Test finding the last checkpoint when it exists."""
        # Create a last checkpoint file
        last_checkpoint = tmp_path / "last_checkpoint.pth"
        last_checkpoint.touch()

        result = find_checkpoint(tmp_path, "epoch")
        assert result == last_checkpoint

    def test_find_checkpoint_best_model(self, tmp_path):
        """Test finding the best model checkpoint."""
        # Create a best model file
        best_model = tmp_path / "best_model.pth"
        best_model.touch()

        result = find_checkpoint(tmp_path, "epoch", specific_checkpoint="best_model")
        assert result == best_model

    def test_find_checkpoint_specific_epoch(self, tmp_path):
        """Test finding a specific epoch checkpoint."""
        # Create an epoch directory with checkpoint
        epoch_dir = tmp_path / "epoch_42"
        epoch_dir.mkdir()
        checkpoint = epoch_dir / "checkpoint.pth"
        with open(checkpoint, "w") as f:
            f.write("test")

        result = find_checkpoint(tmp_path, "epoch_", specific_checkpoint="42")
        assert result == checkpoint

    def test_find_checkpoint_in_epoch_dirs(self, tmp_path):
        """Test finding the latest checkpoint in epoch directories."""
        # Create multiple epoch directories
        for i in range(1, 4):
            epoch_dir = tmp_path / f"epoch_{i}"
            epoch_dir.mkdir()
            checkpoint = epoch_dir / "checkpoint.pth"
            checkpoint.touch()

        result = find_checkpoint(tmp_path, "epoch")
        assert result == tmp_path / "epoch_3" / "checkpoint.pth"

    def test_find_checkpoint_nonexistent_dir(self):
        """Test finding checkpoint in non-existent directory."""
        with pytest.raises(FileNotFoundError):
            find_checkpoint(Path("/nonexistent/path"), "epoch")

    def test_find_checkpoint_no_checkpoints(self, tmp_path):
        """Test finding checkpoint when no checkpoints exist."""
        result = find_checkpoint(tmp_path, "epoch")
        assert result is None

    def test_find_checkpoint_nonexistent_specific_epoch(self, tmp_path):
        """Test finding non-existent specific epoch checkpoint."""
        result = find_checkpoint(tmp_path, "epoch", specific_checkpoint="999")
        assert result is None

    def test_find_checkpoint_nonexistent_best_model(self, tmp_path):
        """Test finding non-existent best model checkpoint."""
        result = find_checkpoint(tmp_path, "epoch", specific_checkpoint="best_model")
        assert result is None