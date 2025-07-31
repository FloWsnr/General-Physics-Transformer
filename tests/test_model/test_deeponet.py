import pytest
import torch
from pathlib import Path

from gphyt.model.deeponet import DeepONet, get_model
from gphyt.model.model_specs import DeepONet_S


class TestDeepONet:
    """Test DeepONet model implementation."""

    def test_deeponet_init(self):
        """Test DeepONet initialization."""
        model = DeepONet(
            input_channels=4,
            branch_n_down_blocks=2,
            trunk_n_layers=2,
            latent_dim=128,
            img_size=(32, 32),
            n_steps_input=2,
        )

        assert model.input_channels == 4
        assert model.img_size == (32, 32)
        assert model.n_steps_input == 2
        assert model.spatial_size == 32 * 32
        assert model.latent_dim == 128

    def test_deeponet_forward(self):
        """Test DeepONet forward pass with synthetic data."""
        batch_size = 2
        n_steps = 4
        height, width = 16, 16
        channels = 3

        model = DeepONet(
            input_channels=channels,
            branch_n_down_blocks=2,
            trunk_n_layers=2,
            latent_dim=66,  # 66 is divisible by 3
            img_size=(height, width),
            n_steps_input=n_steps,
        )

        # Create synthetic input matching PhysicsDataset format
        x = torch.randn(batch_size, n_steps, height, width, channels)

        # Forward pass
        output = model(x)

        # Check output shape matches expected format
        assert output.shape == (batch_size, 1, height, width, channels)
        assert not torch.isnan(output).any()

    def test_get_deeponet_model_factory(self):
        """Test DeepONet factory function."""
        config = DeepONet_S(
            branch_down_blocks=2,
            trunk_n_layers=2,
            latent_dim=125,  # 125 is divisible by 5
        )

        model = get_model(config, input_channels=5, img_size=(24, 24), n_steps_input=3)

        assert isinstance(model, DeepONet)
        assert model.input_channels == 5
        assert model.img_size == (24, 24)
        assert model.n_steps_input == 3
        assert model.latent_dim == 125

    def test_deeponet_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = DeepONet(
            input_channels=2,
            branch_n_down_blocks=1,
            trunk_n_layers=1,
            latent_dim=32,  # 32 is divisible by 2
            img_size=(8, 8),
            n_steps_input=1,
        )

        x = torch.randn(1, 1, 8, 8, 2, requires_grad=True)
        target = torch.randn(1, 1, 8, 8, 2)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None


class TestDeepONetDataCompatibility:
    """Test DeepONet compatibility with PhysicsDataset."""

    def test_deeponet_with_physics_dataset_shapes(self):
        """Test DeepONet works with expected PhysicsDataset output shapes."""
        # Test with known expected shapes from PhysicsDataset
        batch_size = 4
        n_steps_input = 2
        height, width = 64, 64
        channels = 4

        model = DeepONet(
            input_channels=channels,
            branch_n_down_blocks=3,
            trunk_n_layers=3,
            latent_dim=64,  # 64 is divisible by 4
            img_size=(height, width),
            n_steps_input=n_steps_input,
        )

        # Simulate PhysicsDataset output format
        x = torch.randn(batch_size, n_steps_input, height, width, channels)
        y_target = torch.randn(batch_size, 1, height, width, channels)

        output = model(x)

        # Verify shapes match between model output and expected target
        assert output.shape == y_target.shape

        # Test that we can compute loss
        loss = torch.nn.functional.mse_loss(output, y_target)
        assert loss.item() >= 0

    def test_deeponet_matches_transformer_io_format(self):
        """Test that DeepONet I/O format matches PhysicsTransformer."""
        # Both models should handle the same input/output formats
        batch_size = 2
        n_steps = 1
        height, width = 32, 32
        channels = 5

        deeponet = DeepONet(
            input_channels=channels,
            branch_n_down_blocks=2,
            trunk_n_layers=2,
            latent_dim=65,  # 65 is divisible by 5
            img_size=(height, width),
            n_steps_input=n_steps,
        )

        # Input format: (batch_size, n_steps, height, width, channels)
        x = torch.randn(batch_size, n_steps, height, width, channels)

        deeponet_output = deeponet(x)

        # Both models should output: (batch_size, 1, height, width, channels)
        expected_shape = (batch_size, 1, height, width, channels)
        assert deeponet_output.shape == expected_shape

        # Test that the output is reasonable (not all zeros/nans/infs)
        assert not torch.isnan(deeponet_output).any()
        assert not torch.isinf(deeponet_output).any()
        assert deeponet_output.abs().max() > 0  # Not all zeros
