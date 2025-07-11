"""
Tests for the FNO model.
"""

import pytest
import torch

from gphyt.model.fno import FNO, get_model
from gphyt.model.model_specs import FNO_M, FNO_S


def test_forward():
    """Test basic forward pass."""
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 32, 32, 3)

    fno = FNO(
        in_channels=3,
        out_channels=3,
        hidden_channels=16,
        n_layers=2,
        n_modes_height=8,
        n_modes_width=8,
        n_modes_time=4,
    )
    output = fno(data)
    assert output.shape == (10, 1, 32, 32, 3)


def test_forward_cuda():
    """Test forward pass on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = torch.randn(10, 8, 32, 32, 3).cuda()
    fno = FNO(
        in_channels=3,
        out_channels=3,
        hidden_channels=16,
        n_layers=2,
        n_modes_height=8,
        n_modes_width=8,
        n_modes_time=4,
    )
    fno.cuda()
    output = fno(data)
    assert output.shape == (10, 1, 32, 32, 3)

def test_get_model():
    """Test the get_model function."""
    fno_m = get_model(FNO_M())
    assert isinstance(fno_m, FNO)

    fno_s = get_model(FNO_S())
    assert isinstance(fno_s, FNO)
