"""
Tests for the PhysicsTransformer model.
"""

import pytest
import torch
from typing import Optional

from gphyt.models.transformer.model import PhysicsTransformer


def test_forward():
    """Test basic forward pass."""
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_cuda():
    """Test forward pass on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = torch.randn(10, 8, 128, 128, 3).cuda()
    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        patch_size=(4, 16, 16),
        num_layers=4,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
    )
    transformer.cuda()
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_absolute_pos_enc():
    """Test forward pass with absolute positional encoding."""
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="absolute",
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_conv_net_tokenizer():
    """Test forward pass with conv_net tokenizer."""
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="conv_net",
        detokenizer_mode="conv_net",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="absolute",
        tokenizer_net_channels=[16, 32, 64, 128],
        detokenizer_net_channels=[128, 64, 32, 16],
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_linear_tokenizer_overlap():
    """Test forward pass with overlapping tokenizer."""
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="absolute",
        tokenizer_overlap=2,
        detokenizer_overlap=2,
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_derivatives():
    """Test forward pass with derivatives enabled."""
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        use_derivatives=True,
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_euler_integrator():
    """Test forward pass with Euler integrator."""
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="Euler",
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_rk4_integrator():
    """Test forward pass with RK4 integrator."""
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="RK4",
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_heun_integrator():
    """Test forward pass with Heun integrator."""
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="Heun",
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_no_integrator():
    """Test forward pass without integrator."""
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="InvalidIntegrator",  # This will set integrator to None
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_with_step_size():
    """Test forward pass with custom step size."""
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="Euler",
    )
    output = transformer(data, step_size=0.5)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_derivatives_plus_integrator():
    """Test forward pass with both derivatives and integrator."""
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        use_derivatives=True,
        integrator="RK4",
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_invalid_integrator_sets_to_none():
    """Test that invalid integrator sets integrator to None."""
    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="InvalidIntegrator",
    )
    # The model should still work, just without an integrator
    assert transformer.integrator is None


def test_invalid_pos_enc_mode():
    """Test that invalid positional encoding mode raises error."""
    with pytest.raises(ValueError):
        PhysicsTransformer(
            num_fields=3,
            hidden_dim=96,
            mlp_dim=256,
            num_heads=4,
            dropout=0.0,
            tokenizer_mode="linear",
            detokenizer_mode="linear",
            img_size=(8, 128, 128),
            patch_size=(4, 16, 16),
            num_layers=4,
            stochastic_depth_rate=0.0,
            pos_enc_mode="invalid",  # type: ignore
        )


def test_gradient_flow():
    """Test that gradients flow through the model."""
    data = torch.randn(10, 8, 128, 128, 3, requires_grad=True)

    transformer = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="Euler",
    )

    output = transformer(data)
    loss = output.mean()
    loss.backward()

    assert data.grad is not None
    assert torch.all(torch.isfinite(data.grad))


def test_different_integrators_give_different_outputs():
    """Test that different integrators give different outputs."""
    data = torch.randn(5, 8, 64, 64, 3)

    # Create models with different integrators
    euler_model = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 64, 64),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="Euler",
    )

    rk4_model = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 64, 64),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="RK4",
    )

    heun_model = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 64, 64),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="Heun",
    )

    euler_output = euler_model(data)
    rk4_output = rk4_model(data)
    heun_output = heun_model(data)

    # All outputs should have the same shape
    assert euler_output.shape == rk4_output.shape == heun_output.shape

    # Outputs should be different (due to different integration methods)
    assert not torch.allclose(euler_output, rk4_output, atol=1e-6)
    assert not torch.allclose(euler_output, heun_output, atol=1e-6)
    assert not torch.allclose(rk4_output, heun_output, atol=1e-6)


def test_model_without_integrator_vs_with_integrator():
    """Test that model without integrator gives different output than with integrator."""
    data = torch.randn(5, 8, 64, 64, 3)

    # Model without integrator
    no_integrator_model = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 64, 64),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="InvalidIntegrator",  # This will set integrator to None
    )

    # Model with Euler integrator
    euler_model = PhysicsTransformer(
        num_fields=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 64, 64),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
        integrator="Euler",
    )

    no_integrator_output = no_integrator_model(data)
    euler_output = euler_model(data)

    # Both should have the same shape
    assert no_integrator_output.shape == euler_output.shape

    # Outputs should be different
    assert not torch.allclose(no_integrator_output, euler_output, atol=1e-6)
