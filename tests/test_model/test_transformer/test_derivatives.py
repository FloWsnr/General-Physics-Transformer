"""
Tests for the FiniteDifference class.

By: Florian Wiesner
Date: 2025-04-15
"""

import torch
import pytest
from lpfm.model.transformer.derivatives import FiniteDifference


def test_finite_difference_2nd_order():
    """
    Test the FiniteDifference class with 2nd order filter on a simple input.

    Creates a 3D tensor with a linear gradient in each direction and verifies
    that the derivatives are computed correctly.
    """
    # Create a simple 3D tensor with linear gradients
    B, T, H, W, C = 1, 3, 3, 3, 1
    x = torch.zeros(B, T, H, W, C)

    # Add constant line in t-direction
    x[:, :, 1, 1, 0] = 1
    # Initialize the finite difference module
    fd = FiniteDifference(num_channels=C, filter_1d="2nd")

    # Compute derivatives
    dt, dh, dw = fd(x)

    # In T direction, the derivative should be 0
    assert torch.allclose(dt, torch.zeros_like(dt))
    # In H and W direction, the derivative should be non-zero
    assert not torch.allclose(dh, torch.zeros_like(dh))
    assert not torch.allclose(dw, torch.zeros_like(dw))