"""
Tests for the training visualization utilities.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from metaparc.utils.train_vis import visualize_predictions


@pytest.fixture
def sample_data():
    """Create sample input, prediction and target tensors for testing.

    Returns
    -------
    tuple
        A tuple containing (inputs, predictions, targets) tensors with shape
        (batch_size=4, time_steps=4, height=32, width=32, channels=5)
    """
    # Create sample data with known values
    inputs = torch.ones((4, 4, 32, 32, 5)) * 0.5
    predictions = torch.ones((4, 4, 32, 32, 5)) * 0.7
    targets = torch.ones((4, 4, 32, 32, 5)) * 0.6

    # add some noise to the targets
    targets += torch.randn((4, 4, 32, 32, 5)) * 0.05
    predictions += torch.randn((4, 4, 32, 32, 5)) * 0.05
    inputs += torch.randn((4, 4, 32, 32, 5)) * 0.05

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
    inputs, predictions, targets = sample_data
    save_path = tmp_path / "visualizations"

    # Call the function
    visualize_predictions(save_path, inputs, predictions, targets, show=True)

    # Check that files were created
    assert save_path.exists()
    assert len(list(save_path.glob("*.png"))) == 5  # One file per channel

    # Check file names
    assert (save_path / "channel_0.png").exists()
    assert (save_path / "channel_1.png").exists()
    assert (save_path / "channel_2.png").exists()
    assert (save_path / "channel_3.png").exists()
    assert (save_path / "channel_4.png").exists()
