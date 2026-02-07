"""
Tests for the FNO model.
"""

import pytest
import torch

from gphyt.models.fno import FNO, get_model
from gphyt.models.model_specs import FNO_M, FNO_S


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
    config_m = {"model_size": "FNO_M"}
    fno_m = get_model(config_m)
    assert isinstance(fno_m, FNO)

    config_s = {"model_size": "FNO_S"}
    fno_s = get_model(config_s)
    assert isinstance(fno_s, FNO)

    # Test default behavior (should default to FNO_M)
    config_default = {}
    fno_default = get_model(config_default)
    assert isinstance(fno_default, FNO)


def test_fno_s_config():
    """Test FNO with FNO_S configuration."""
    data = torch.randn(2, 4, 32, 32, 5)
    config = {"model_size": "FNO_S"}
    fno = get_model(config)
    output = fno(data)
    assert output.shape == (2, 1, 32, 32, 5)


def test_fno_m_config():
    """Test FNO with FNO_M configuration."""
    data = torch.randn(2, 4, 32, 32, 5)
    config = {"model_size": "FNO_M"}
    fno = get_model(config)
    output = fno(data)
    assert output.shape == (2, 1, 32, 32, 5)
