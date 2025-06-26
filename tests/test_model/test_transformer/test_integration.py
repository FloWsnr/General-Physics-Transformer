"""
Tests for the numerical integration module.
"""

import pytest
import torch
import torch.nn as nn

from gphyt.model.transformer.num_integration import Euler


class TestEuler:
    """Test suite for the Euler integration class."""

    def test_initialization(self):
        """Test initialization of Euler integrator."""
        integrator = Euler()
        assert isinstance(integrator, nn.Module)

    def test_forward_default_step_size(self):
        """Test forward pass with default step size."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensors
        dt = torch.randn(batch_size, time, height, width, channels)
        input_state = torch.randn(batch_size, time, height, width, channels)

        integrator = Euler()
        output = integrator(dt, input_state)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # Check if integration is correct (x + dt)
        expected_output = input_state + dt
        assert torch.allclose(output, expected_output)

    def test_forward_custom_step_size(self):
        """Test forward pass with custom step size."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64
        step_size = 0.5

        # Create input tensors
        dt = torch.randn(batch_size, time, height, width, channels)
        input_state = torch.randn(batch_size, time, height, width, channels)

        integrator = Euler()
        output = integrator(dt, input_state, step_size=step_size)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # Check if integration is correct (x + step_size * dt)
        expected_output = input_state + step_size * dt
        assert torch.allclose(output, expected_output)

    def test_forward_zero_derivative(self):
        """Test forward pass with zero derivative."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensors
        dt = torch.zeros(batch_size, time, height, width, channels)
        input_state = torch.randn(batch_size, time, height, width, channels)

        integrator = Euler()
        output = integrator(dt, input_state)

        # Check if output equals input when derivative is zero
        assert torch.allclose(output, input_state)

    def test_forward_gradient_flow(self):
        """Test if gradients flow properly through the integrator."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensors with requires_grad=True
        dt = torch.randn(batch_size, time, height, width, channels, requires_grad=True)
        input_state = torch.randn(
            batch_size, time, height, width, channels, requires_grad=True
        )

        integrator = Euler()
        output = integrator(dt, input_state)

        # Create a dummy loss and check if gradients flow
        loss = output.mean()
        loss.backward()

        assert dt.grad is not None
        assert input_state.grad is not None
        assert torch.all(torch.isfinite(dt.grad))
        assert torch.all(torch.isfinite(input_state.grad))
