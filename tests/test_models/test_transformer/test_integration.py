"""
Tests for the numerical integration module.
"""

import pytest
import torch
import torch.nn as nn

from gphyt.models.transformer.num_integration import Euler, RK4, Heun, Integrator


class TestIntegrator:
    """Test suite for the base Integrator class."""

    def test_abstract_class(self):
        """Test that Integrator is an abstract base class."""
        with pytest.raises(TypeError):
            Integrator()


class TestEuler:
    """Test suite for the Euler integration class."""

    def test_initialization(self):
        """Test initialization of Euler integrator."""
        integrator = Euler()
        assert isinstance(integrator, nn.Module)
        assert isinstance(integrator, Integrator)

    def test_forward_default_step_size(self):
        """Test forward pass with default step size."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define a simple derivative function
        def derivative_fn(x):
            return torch.ones_like(x) * 2.0

        integrator = Euler()
        output = integrator(derivative_fn, input_state)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # Check if integration is correct (x + step_size * derivative)
        expected_output = input_state + 1.0 * derivative_fn(input_state)
        assert torch.allclose(output, expected_output)

    def test_forward_custom_step_size(self):
        """Test forward pass with custom step size."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64
        step_size = 0.5

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define a simple derivative function
        def derivative_fn(x):
            return torch.ones_like(x) * 3.0

        integrator = Euler()
        output = integrator(derivative_fn, input_state, step_size=step_size)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # Check if integration is correct (x + step_size * derivative)
        expected_output = input_state + step_size * derivative_fn(input_state)
        assert torch.allclose(output, expected_output)

    def test_forward_zero_derivative(self):
        """Test forward pass with zero derivative function."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define zero derivative function
        def derivative_fn(x):
            return torch.zeros_like(x)

        integrator = Euler()
        output = integrator(derivative_fn, input_state)

        # Check if output equals input when derivative is zero
        assert torch.allclose(output, input_state)

    def test_forward_gradient_flow(self):
        """Test if gradients flow properly through the integrator."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensor with requires_grad=True
        input_state = torch.randn(
            batch_size, time, height, width, channels, requires_grad=True
        )

        # Define derivative function that depends on input
        def derivative_fn(x):
            return x * 2.0

        integrator = Euler()
        output = integrator(derivative_fn, input_state)

        # Create a dummy loss and check if gradients flow
        loss = output.mean()
        loss.backward()

        assert input_state.grad is not None
        assert torch.all(torch.isfinite(input_state.grad))


class TestRK4:
    """Test suite for the RK4 integration class."""

    def test_initialization(self):
        """Test initialization of RK4 integrator."""
        integrator = RK4()
        assert isinstance(integrator, nn.Module)
        assert isinstance(integrator, Integrator)

    def test_forward_default_step_size(self):
        """Test forward pass with default step size."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define a simple derivative function
        def derivative_fn(x):
            return torch.ones_like(x) * 2.0

        integrator = RK4()
        output = integrator(derivative_fn, input_state)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # For constant derivative, RK4 should give same result as Euler
        expected_output = input_state + 1.0 * derivative_fn(input_state)
        assert torch.allclose(output, expected_output, atol=1e-6)

    def test_forward_custom_step_size(self):
        """Test forward pass with custom step size."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64
        step_size = 0.5

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define a simple derivative function
        def derivative_fn(x):
            return torch.ones_like(x) * 3.0

        integrator = RK4()
        output = integrator(derivative_fn, input_state, step_size=step_size)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # For constant derivative, RK4 should give same result as Euler
        expected_output = input_state + step_size * derivative_fn(input_state)
        assert torch.allclose(output, expected_output, atol=1e-6)

    def test_forward_zero_derivative(self):
        """Test forward pass with zero derivative function."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define zero derivative function
        def derivative_fn(x):
            return torch.zeros_like(x)

        integrator = RK4()
        output = integrator(derivative_fn, input_state)

        # Check if output equals input when derivative is zero
        assert torch.allclose(output, input_state)

    def test_forward_gradient_flow(self):
        """Test if gradients flow properly through the integrator."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensor with requires_grad=True
        input_state = torch.randn(
            batch_size, time, height, width, channels, requires_grad=True
        )

        # Define derivative function that depends on input
        def derivative_fn(x):
            return x * 2.0

        integrator = RK4()
        output = integrator(derivative_fn, input_state)

        # Create a dummy loss and check if gradients flow
        loss = output.mean()
        loss.backward()

        assert input_state.grad is not None
        assert torch.all(torch.isfinite(input_state.grad))

    def test_forward_linear_derivative(self):
        """Test RK4 with linear derivative function."""
        batch_size = 1
        time = 1
        height = 4
        width = 4
        channels = 2
        step_size = 0.1

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define linear derivative function
        def derivative_fn(x):
            return x * 2.0

        integrator = RK4()
        output = integrator(derivative_fn, input_state, step_size=step_size)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # For linear derivative, RK4 should give different result than Euler
        euler_integrator = Euler()
        euler_output = euler_integrator(derivative_fn, input_state, step_size=step_size)

        # RK4 should be more accurate than Euler for non-constant derivatives
        assert not torch.allclose(output, euler_output, atol=1e-6)


class TestHeun:
    """Test suite for the Heun integration class."""

    def test_initialization(self):
        """Test initialization of Heun integrator."""
        integrator = Heun()
        assert isinstance(integrator, nn.Module)
        assert isinstance(integrator, Integrator)

    def test_forward_default_step_size(self):
        """Test forward pass with default step size."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define a simple derivative function
        def derivative_fn(x):
            return torch.ones_like(x) * 2.0

        integrator = Heun()
        output = integrator(derivative_fn, input_state)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # For constant derivative, Heun should give same result as Euler
        expected_output = input_state + 1.0 * derivative_fn(input_state)
        assert torch.allclose(output, expected_output, atol=1e-6)

    def test_forward_custom_step_size(self):
        """Test forward pass with custom step size."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64
        step_size = 0.5

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define a simple derivative function
        def derivative_fn(x):
            return torch.ones_like(x) * 3.0

        integrator = Heun()
        output = integrator(derivative_fn, input_state, step_size=step_size)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # For constant derivative, Heun should give same result as Euler
        expected_output = input_state + step_size * derivative_fn(input_state)
        assert torch.allclose(output, expected_output, atol=1e-6)

    def test_forward_zero_derivative(self):
        """Test forward pass with zero derivative function."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define zero derivative function
        def derivative_fn(x):
            return torch.zeros_like(x)

        integrator = Heun()
        output = integrator(derivative_fn, input_state)

        # Check if output equals input when derivative is zero
        assert torch.allclose(output, input_state)

    def test_forward_gradient_flow(self):
        """Test if gradients flow properly through the integrator."""
        batch_size = 2
        time = 3
        height = 16
        width = 16
        channels = 64

        # Create input tensor with requires_grad=True
        input_state = torch.randn(
            batch_size, time, height, width, channels, requires_grad=True
        )

        # Define derivative function that depends on input
        def derivative_fn(x):
            return x * 2.0

        integrator = Heun()
        output = integrator(derivative_fn, input_state)

        # Create a dummy loss and check if gradients flow
        loss = output.mean()
        loss.backward()

        assert input_state.grad is not None
        assert torch.all(torch.isfinite(input_state.grad))

    def test_forward_linear_derivative(self):
        """Test Heun with linear derivative function."""
        batch_size = 1
        time = 1
        height = 4
        width = 4
        channels = 2
        step_size = 0.1

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define linear derivative function
        def derivative_fn(x):
            return x * 2.0

        integrator = Heun()
        output = integrator(derivative_fn, input_state, step_size=step_size)

        # Check output shape matches input
        assert output.shape == (batch_size, time, height, width, channels)

        # For linear derivative, Heun should give different result than Euler
        euler_integrator = Euler()
        euler_output = euler_integrator(derivative_fn, input_state, step_size=step_size)

        # Heun should be more accurate than Euler for non-constant derivatives
        assert not torch.allclose(output, euler_output, atol=1e-6)


class TestIntegrationComparison:
    """Test suite for comparing different integration methods."""

    def test_constant_derivative_all_methods(self):
        """Test that all methods give same result for constant derivative."""
        batch_size = 1
        time = 1
        height = 4
        width = 4
        channels = 2
        step_size = 0.5

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define constant derivative function
        def derivative_fn(x):
            return torch.ones_like(x) * 3.0

        euler = Euler()
        rk4 = RK4()
        heun = Heun()

        euler_output = euler(derivative_fn, input_state, step_size=step_size)
        rk4_output = rk4(derivative_fn, input_state, step_size=step_size)
        heun_output = heun(derivative_fn, input_state, step_size=step_size)

        # All methods should give same result for constant derivative
        assert torch.allclose(euler_output, rk4_output, atol=1e-6)
        assert torch.allclose(euler_output, heun_output, atol=1e-6)
        assert torch.allclose(rk4_output, heun_output, atol=1e-6)

    def test_linear_derivative_accuracy_comparison(self):
        """Test accuracy comparison for linear derivative function."""
        batch_size = 1
        time = 1
        height = 4
        width = 4
        channels = 2
        step_size = 0.1

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define linear derivative function
        def derivative_fn(x):
            return x * 2.0

        euler = Euler()
        rk4 = RK4()
        heun = Heun()

        euler_output = euler(derivative_fn, input_state, step_size=step_size)
        rk4_output = rk4(derivative_fn, input_state, step_size=step_size)
        heun_output = heun(derivative_fn, input_state, step_size=step_size)

        # All outputs should be different for non-constant derivative
        assert not torch.allclose(euler_output, rk4_output, atol=1e-6)
        assert not torch.allclose(euler_output, heun_output, atol=1e-6)
        assert not torch.allclose(rk4_output, heun_output, atol=1e-6)

    def test_step_size_zero(self):
        """Test behavior when step size is zero."""
        batch_size = 1
        time = 1
        height = 4
        width = 4
        channels = 2
        step_size = 0.0

        # Create input tensor
        input_state = torch.randn(batch_size, time, height, width, channels)

        # Define derivative function
        def derivative_fn(x):
            return torch.ones_like(x) * 2.0

        euler = Euler()
        rk4 = RK4()
        heun = Heun()

        euler_output = euler(derivative_fn, input_state, step_size=step_size)
        rk4_output = rk4(derivative_fn, input_state, step_size=step_size)
        heun_output = heun(derivative_fn, input_state, step_size=step_size)

        # All methods should return input unchanged when step_size is zero
        assert torch.allclose(euler_output, input_state)
        assert torch.allclose(rk4_output, input_state)
        assert torch.allclose(heun_output, input_state)
