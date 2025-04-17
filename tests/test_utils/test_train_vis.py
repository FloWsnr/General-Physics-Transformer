"""
Tests for the training visualization utilities.
"""

from pathlib import Path

import pytest
import torch

from lpfm.utils.train_vis import visualize_predictions


@pytest.fixture
def sample_data():
    """Create sample input, prediction and target tensors for testing.

    Returns
    -------
    tuple
        A tuple containing (inputs, predictions, targets) tensors with shape
        (batch_size=4, time_steps=4, height=256, width=128, channels=5)
    """
    # Create sample data with known values
    inputs = torch.ones((4, 8, 256, 128, 5)) * 0.5
    predictions = torch.ones((4, 1, 256, 128, 5)) * 0.7
    targets = torch.ones((4, 1, 256, 128, 5)) * 0.6

    # add some noise to the targets
    targets += torch.randn((4, 1, 256, 128, 5)) * 0.05
    predictions += torch.randn((4, 1, 256, 128, 5)) * 0.05
    inputs += torch.randn((4, 8, 256, 128, 5)) * 0.05

    return inputs, predictions, targets


def test_visualize_predictions_creates_files(tmp_path: Path, sample_data: tuple):
    """Test that visualize_predictions creates the expected output files.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest
    sample_data : tuple
        Sample input data provided by fixture
    """
    # swap the first two dimensions
    inputs, predictions, targets = sample_data
    save_path = tmp_path / "visualizations" / "test.png"

    # Call the function
    visualize_predictions(save_path, inputs, predictions, targets, svg=True)

    # find all files in the save_path.parent
    files = list(save_path.parent.glob("*.png"))
    files += list(save_path.parent.glob("*.svg"))
    assert len(files) == 2


def test_visualize_predictions_with_less_than_4_time_steps(
    tmp_path: Path, sample_data: tuple
):
    """Test that visualize_predictions creates the expected output files.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest
    sample_data : tuple
        Sample input data provided by fixture
    """
    # swap the first two dimensions
    inputs, predictions, targets = sample_data
    inputs = inputs[:, :3, :, :, :]
    save_path = tmp_path / "visualizations" / "test.png"

    # Call the function
    visualize_predictions(save_path, inputs, predictions, targets, svg=True)

    # find all files in the save_path.parent
    files = list(save_path.parent.glob("*.png"))
    files += list(save_path.parent.glob("*.svg"))
    assert len(files) == 2


def test_visualize_predictions_handles_single_timestep(
    tmp_path: Path, sample_data: tuple
):
    """Test that visualize_predictions handles single timestep correctly.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest
    sample_data : tuple
        Sample input data provided by fixture
    """
    inputs, predictions, targets = sample_data
    inputs = inputs[:, 0, :, :, :].unsqueeze(1)
    predictions = predictions[:, 0, :, :, :].unsqueeze(1)
    targets = targets[:, 0, :, :, :].unsqueeze(1)
    save_path = tmp_path / "visualizations.png"

    # Call the function
    visualize_predictions(save_path, inputs, predictions, targets, svg=True)

    # find all files in the save_path.parent
    files = list(save_path.parent.glob("*.png"))
    files += list(save_path.parent.glob("*.svg"))
    assert len(files) == 2


def test_visualize_predictions_handles_multiple_samples(
    tmp_path: Path, sample_data: tuple
):
    """Test that visualize_predictions handles multiple samples correctly.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest
    sample_data : tuple
        Sample input data provided by fixture
    """
    # swap the first two dimensions
    inputs, predictions, targets = sample_data
    save_path = tmp_path / "visualizations" / "test.png"

    # Call the function
    visualize_predictions(
        save_path, inputs, predictions, targets, num_samples=2, svg=True
    )

    # find all files in the save_path.parent
    files = list(save_path.parent.glob("*.png"))
    files += list(save_path.parent.glob("*.svg"))
    assert len(files) == 4


def test_visualize_predictions_handles_more_samples_than_available(
    tmp_path: Path, sample_data: tuple
):
    """Test that visualize_predictions handles more samples than available correctly.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest
    sample_data : tuple
        Sample input data provided by fixture
    """
    # swap the first two dimensions
    inputs, predictions, targets = sample_data
    save_path = tmp_path / "visualizations" / "test.png"

    # Call the function
    visualize_predictions(
        save_path, inputs, predictions, targets, num_samples=10, svg=True
    )

    # find all files in the save_path.parent
    files = list(save_path.parent.glob("*.png"))
    files += list(save_path.parent.glob("*.svg"))
    assert len(files) == 4 * 2  # batch size 4
