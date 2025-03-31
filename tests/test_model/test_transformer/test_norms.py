"""
Tests for normalization layers in the transformer module.

By: Florian Wiesner
Date: 2025-03-31
"""

import pytest
import torch

from metaparc.model.transformer.norms import RevIN, RevSPADE_3D, RevSPADE_2D


class Test_RevIN:
    def test_revin_initialization(self):
        """
        Test RevIN layer initialization.

        Tests if the RevIN layer is correctly initialized with the expected
        parameter shapes and default values.
        """
        num_channels = 64
        revin = RevIN(num_channels)

        assert revin.num_channels == num_channels
        assert revin.eps == 1e-5
        assert revin.affine_weight.shape == (num_channels,)
        assert revin.affine_bias.shape == (num_channels,)
        assert torch.all(revin.affine_weight == 1.0)
        assert torch.all(revin.affine_bias == 0.0)

    def test_revin_forward(self):
        """
        Test RevIN forward pass in both normalization and denormalization modes.

        Verifies that the RevIN layer correctly processes inputs in both normalization
        and denormalization modes, maintaining the expected tensor shapes.
        """
        batch_size = 4
        time_steps = 10
        num_channels = 64
        height = 32
        width = 32

        revin = RevIN(num_channels)

        # Create random input data
        x = torch.randn(batch_size, time_steps, num_channels, height, width)

        # Test normalization
        x_norm = revin(x, mode="norm")
        assert x_norm.shape == x.shape

        # Test denormalization
        x_denorm = revin(x_norm, mode="denorm")
        assert x_denorm.shape == x.shape

        # Check if denormalization approximately reconstructs the original input
        assert torch.allclose(x, x_denorm, rtol=1e-5, atol=1e-5)

    def test_revin_statistics(self):
        """
        Test if RevIN correctly computes and applies statistics.

        Checks if the RevIN layer correctly normalizes input data to have
        approximately zero mean and unit variance across specified dimensions.
        """
        batch_size = 10
        time_steps = 100
        num_channels = 128
        height = 32
        width = 32

        revin = RevIN(num_channels)

        # Create input with known statistics
        x = torch.randn(batch_size, time_steps, num_channels, height, width)
        x = x * 2.0 + 1.0  # Scale and shift

        # Apply normalization
        x_norm = revin(x, mode="norm")

        # Check if normalized data has approximately zero mean and unit variance
        mean = torch.mean(x_norm, dim=(1, 3, 4))
        var = torch.var(x_norm, dim=(1, 3, 4))

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-2)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


class Test_RevSpade_3D:
    def test_revspade_initialization(self):
        """Test RevSPADE layer initialization."""
        in_channels = 64
        mask_channels = 32
        kernel_size = 3
        revspade = RevSPADE_3D(in_channels, mask_channels, kernel_size)

        assert revspade.in_channels == in_channels
        assert revspade.mask_channels == mask_channels
        assert revspade.kernel_size == kernel_size
        assert revspade.eps == 1e-5

    def test_revspade_forward(self):
        """Test RevSPADE forward pass in both normalization and denormalization modes."""
        batch_size = 4
        time_steps = 10
        height = 32
        width = 32
        in_channels = 64
        mask_channels = 2
        revspade = RevSPADE_3D(in_channels, mask_channels)

        # Create random input data and mask
        x = torch.randn(batch_size, time_steps, in_channels, height, width)
        mask = torch.randn(batch_size, time_steps, mask_channels, height, width)

        # Test normalization
        x_norm = revspade(x, mask, mode="norm")
        assert x_norm.shape == x.shape

        # Test denormalization
        x_denorm = revspade(x_norm, mask, mode="denorm")
        assert x_denorm.shape == x.shape

        # Check if denormalization approximately reconstructs the original input
        assert torch.allclose(x, x_denorm, atol=1e-1)


class Test_RevSpade_2D:
    def test_revspade_initialization(self):
        """Test RevSPADE layer initialization."""
        in_channels = 64
        mask_channels = 32
        kernel_size = 3
        revspade = RevSPADE_2D(in_channels, mask_channels, kernel_size)

        # Check if the layer has the correct attributes
        assert revspade.in_channels == in_channels
        assert revspade.mask_channels == mask_channels
        assert revspade.kernel_size == kernel_size

        # Check if the layer has the correct components
        assert hasattr(revspade, "initial_conv")
        assert hasattr(revspade, "gamma_conv")
        assert hasattr(revspade, "beta_conv")

    def test_revspade_2d_forward_norm_mode(self):
        """Test RevSPADE_2D forward pass in normalization mode."""
        batch_size = 4
        height = 32
        width = 32
        time_steps = 10
        in_channels = 64
        mask_channels = 32

        revspade = RevSPADE_2D(in_channels * time_steps, mask_channels * time_steps)

        # Create input tensor and mask
        x = torch.randn(batch_size, time_steps, in_channels, height, width)
        mask = torch.randn(batch_size, time_steps, mask_channels, height, width)

        # Apply normalization
        output = revspade(x, mask, mode="norm")

        # Check output shape
        assert output.shape == x.shape

        # Check that gamma and beta were computed
        assert hasattr(revspade, "gamma")
        assert hasattr(revspade, "beta")
        assert hasattr(revspade, "mean")
        assert hasattr(revspade, "stdev")

    def test_revspade_2d_forward_denorm_mode(self):
        """Test RevSPADE_2D forward pass in denormalization mode."""
        batch_size = 4
        height = 32
        width = 32
        time_steps = 10
        in_channels = 64
        mask_channels = 32
        revspade = RevSPADE_2D(in_channels * time_steps, mask_channels * time_steps)

        # Create input tensor and mask
        x = torch.randn(batch_size, time_steps, in_channels, height, width)
        mask = torch.randn(batch_size, time_steps, mask_channels, height, width)

        # First apply normalization to set internal state
        x_norm = revspade(x, mask, mode="norm")

        # Then apply denormalization
        x_denorm = revspade(x_norm, mask, mode="denorm")

        # Check output shape
        assert x_denorm.shape == x.shape

        # Check if denormalization approximately reverses normalization
        assert torch.allclose(x, x_denorm, atol=1e-1)
