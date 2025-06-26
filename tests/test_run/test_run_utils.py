import pytest
from pathlib import Path
from gphyt.run.run_utils import find_checkpoint


def test_find_checkpoint_last_checkpoint(tmp_path):
    """Test finding the last checkpoint when it exists."""
    # Create a last checkpoint file
    last_checkpoint = tmp_path / "last_checkpoint.pth"
    last_checkpoint.touch()

    result = find_checkpoint(tmp_path, "epoch")
    assert result == last_checkpoint


def test_find_checkpoint_best_model(tmp_path):
    """Test finding the best model checkpoint."""
    # Create a best model file
    best_model = tmp_path / "best_model.pth"
    best_model.touch()

    result = find_checkpoint(tmp_path, "epoch", specific_checkpoint="best_model")
    assert result == best_model


def test_find_checkpoint_specific_epoch(tmp_path):
    """Test finding a specific epoch checkpoint."""
    # Create an epoch directory with checkpoint
    epoch_dir = tmp_path / "epoch_42"
    epoch_dir.mkdir()
    checkpoint = epoch_dir / "checkpoint.pth"
    with open(checkpoint, "w") as f:
        f.write("test")

    result = find_checkpoint(tmp_path, "epoch_", specific_checkpoint="42")
    assert result == checkpoint


def test_find_checkpoint_in_epoch_dirs(tmp_path):
    """Test finding the latest checkpoint in epoch directories."""
    # Create multiple epoch directories
    for i in range(1, 4):
        epoch_dir = tmp_path / f"epoch_{i}"
        epoch_dir.mkdir()
        checkpoint = epoch_dir / "checkpoint.pth"
        checkpoint.touch()

    result = find_checkpoint(tmp_path, "epoch")
    assert result == tmp_path / "epoch_3" / "checkpoint.pth"


def test_find_checkpoint_nonexistent_dir():
    """Test finding checkpoint in non-existent directory."""
    with pytest.raises(FileNotFoundError):
        find_checkpoint(Path("/nonexistent/path"), "epoch")


def test_find_checkpoint_no_checkpoints(tmp_path):
    """Test finding checkpoint when no checkpoints exist."""
    result = find_checkpoint(tmp_path, "epoch")
    assert result is None


def test_find_checkpoint_nonexistent_specific_epoch(tmp_path):
    """Test finding non-existent specific epoch checkpoint."""
    result = find_checkpoint(tmp_path, "epoch", specific_checkpoint="999")
    assert result is None


def test_find_checkpoint_nonexistent_best_model(tmp_path):
    """Test finding non-existent best model checkpoint."""
    result = find_checkpoint(tmp_path, "epoch", specific_checkpoint="best_model")
    assert result is None
