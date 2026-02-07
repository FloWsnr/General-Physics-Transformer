"""
Tests for custom loss functions.
"""

import torch
import pytest
from gphyt.models.loss_fns import MSE, MAE, NMSE, VMSE, NRMSE, VRMSE, RMSE


@pytest.fixture
def sample_tensors():
    """Create sample tensors with shape (batch, time, h, w, channels)."""
    batch_size, time, h, w, channels = 2, 4, 8, 8, 3
    pred = torch.randn(batch_size, time, h, w, channels)
    target = torch.randn(batch_size, time, h, w, channels)
    return pred, target


class TestMSE:
    """Test MSE loss function."""

    def test_mse_dims_none(self, sample_tensors):
        """Test MSE with dims=None (reduces over all dimensions)."""
        pred, target = sample_tensors
        loss_fn = MSE(dims=None)
        loss = loss_fn(pred, target)

        # Should return scalar when reducing over all dims
        assert loss.shape == torch.Size([])
        assert loss >= 0

    def test_mse_dims_123(self, sample_tensors):
        """Test MSE with dims=(1,2,3) (reduces over time, height, width)."""
        pred, target = sample_tensors
        loss_fn = MSE(dims=(1, 2, 3))
        loss = loss_fn(pred, target)

        # Should return tensor with shape (batch, channels)
        assert loss.shape == torch.Size([2, 3])
        assert torch.all(loss >= 0)


class TestMAE:
    """Test MAE loss function."""

    def test_mae_dims_none(self, sample_tensors):
        """Test MAE with dims=None (reduces over all dimensions)."""
        pred, target = sample_tensors
        loss_fn = MAE(dims=None)
        loss = loss_fn(pred, target)

        # Should return scalar when reducing over all dims
        assert loss.shape == torch.Size([])
        assert loss >= 0

    def test_mae_dims_123(self, sample_tensors):
        """Test MAE with dims=(1,2,3) (reduces over time, height, width)."""
        pred, target = sample_tensors
        loss_fn = MAE(dims=(1, 2, 3))
        loss = loss_fn(pred, target)

        # Should return tensor with shape (batch, channels)
        assert loss.shape == torch.Size([2, 3])
        assert torch.all(loss >= 0)


class TestNMSE:
    """Test NMSE loss function."""

    def test_nmse_dims_none(self, sample_tensors):
        """Test NMSE with dims=None (reduces over all dimensions)."""
        pred, target = sample_tensors
        loss_fn = NMSE(dims=None)
        loss = loss_fn(pred, target)

        # Should return scalar when reducing over all dims
        assert loss.shape == torch.Size([])
        assert loss >= 0

    def test_nmse_dims_123(self, sample_tensors):
        """Test NMSE with dims=(1,2,3) (reduces over time, height, width)."""
        pred, target = sample_tensors
        loss_fn = NMSE(dims=(1, 2, 3))
        loss = loss_fn(pred, target)

        # Should return tensor with shape (batch, channels)
        assert loss.shape == torch.Size([2, 3])
        assert torch.all(loss >= 0)


class TestVMSE:
    """Test VMSE loss function."""

    def test_vmse_dims_none(self, sample_tensors):
        """Test VMSE with dims=None (reduces over all dimensions)."""
        pred, target = sample_tensors
        loss_fn = VMSE(dims=None)
        loss = loss_fn(pred, target)

        # Should return scalar when reducing over all dims
        assert loss.shape == torch.Size([])
        assert loss >= 0

    def test_vmse_dims_123(self, sample_tensors):
        """Test VMSE with dims=(1,2,3) (reduces over time, height, width)."""
        pred, target = sample_tensors
        loss_fn = VMSE(dims=(1, 2, 3))
        loss = loss_fn(pred, target)

        # Should return tensor with shape (batch, channels)
        assert loss.shape == torch.Size([2, 3])
        assert torch.all(loss >= 0)


class TestNRMSE:
    """Test NRMSE loss function."""

    def test_nrmse_dims_none(self, sample_tensors):
        """Test NRMSE with dims=None (reduces over all dimensions)."""
        pred, target = sample_tensors
        loss_fn = NRMSE(dims=None)
        loss = loss_fn(pred, target)

        # Should return scalar when reducing over all dims
        assert loss.shape == torch.Size([])
        assert loss >= 0

    def test_nrmse_dims_123(self, sample_tensors):
        """Test NRMSE with dims=(1,2,3) (reduces over time, height, width)."""
        pred, target = sample_tensors
        loss_fn = NRMSE(dims=(1, 2, 3))
        loss = loss_fn(pred, target)

        # Should return tensor with shape (batch, channels)
        assert loss.shape == torch.Size([2, 3])
        assert torch.all(loss >= 0)


class TestVRMSE:
    """Test VRMSE loss function."""

    def test_vrmse_dims_none(self, sample_tensors):
        """Test VRMSE with dims=None (reduces over all dimensions)."""
        pred, target = sample_tensors
        loss_fn = VRMSE(dims=None)
        loss = loss_fn(pred, target)

        # Should return scalar when reducing over all dims
        assert loss.shape == torch.Size([])
        assert loss >= 0

    def test_vrmse_dims_123(self, sample_tensors):
        """Test VRMSE with dims=(1,2,3) (reduces over time, height, width)."""
        pred, target = sample_tensors
        loss_fn = VRMSE(dims=(1, 2, 3))
        loss = loss_fn(pred, target)

        # Should return tensor with shape (batch, channels)
        assert loss.shape == torch.Size([2, 3])
        assert torch.all(loss >= 0)


class TestRMSE:
    """Test RMSE loss function."""

    def test_rmse_dims_none(self, sample_tensors):
        """Test RMSE with dims=None (reduces over all dimensions)."""
        pred, target = sample_tensors
        loss_fn = RMSE(dims=None)
        loss = loss_fn(pred, target)

        # Should return scalar when reducing over all dims
        assert loss.shape == torch.Size([])
        assert loss >= 0

    def test_rmse_dims_123(self, sample_tensors):
        """Test RMSE with dims=(1,2,3) (reduces over time, height, width)."""
        pred, target = sample_tensors
        loss_fn = RMSE(dims=(1, 2, 3))
        loss = loss_fn(pred, target)

        # Should return tensor with shape (batch, channels)
        assert loss.shape == torch.Size([2, 3])
        assert torch.all(loss >= 0)