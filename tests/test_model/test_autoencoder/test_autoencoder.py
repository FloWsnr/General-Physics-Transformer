import torch
import pytest
from lpfm.model.autoencoder.autoencoder import Encoder, Decoder


class TestAutoencoder:
    @pytest.mark.parametrize("patch_size", [(2, 4, 4), (4, 16, 16), (8, 32, 32)])
    def test_encoder_forward_patch_sizes(self, patch_size):
        """Test the forward function of the Encoder class."""
        # Define test parameters
        batch_size = 3
        time = 8
        height = 64
        width = 64
        in_channels = 3
        out_channels = 128

        hidden_channels = [in_channels, 16, 32, out_channels]

        # Create input tensor
        x = torch.randn(batch_size, time, height, width, in_channels)

        # Initialize encoder
        encoder = Encoder(
            channels=hidden_channels,
            patch_size=patch_size,
        )

        # Forward pass
        output = encoder(x)

        # Calculate expected output shape
        expected_time = time // patch_size[0]
        expected_height = height // patch_size[1]
        expected_width = width // patch_size[2]

        # Check output shape
        assert output.shape == (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            out_channels,
        )

        # Check output is not all zeros or NaNs
        assert not torch.isnan(output).any()
        assert not (output == 0).all()

    @pytest.mark.parametrize(
        "hidden_channels",
        [
            [3, 16, 128],
            [3, 16, 32, 128],
            [3, 16, 32, 64, 128],
        ],
    )
    def test_encoder_forward_hidden_channels(self, hidden_channels):
        """Test the forward function of the Encoder class."""
        # Define test parameters
        batch_size = 3
        time = 8
        height = 64
        width = 64
        in_channels = 3

        out_channels = 128

        patch_size = (4, 16, 16)

        # Create input tensor
        x = torch.randn(batch_size, time, height, width, in_channels)

        # Initialize encoder
        encoder = Encoder(
            channels=hidden_channels,
            patch_size=patch_size,
        )

        # Forward pass
        output = encoder(x)

        # Calculate expected output shape
        expected_time = time // patch_size[0]
        expected_height = height // patch_size[1]
        expected_width = width // patch_size[2]

        # Check output shape
        assert output.shape == (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            out_channels,
        )


class TestDecoder:
    @pytest.mark.parametrize("patch_size", [(2, 4, 4), (4, 16, 16), (8, 32, 32)])
    def test_decoder_forward_patch_sizes(self, patch_size):
        """
        Test the forward function of the Decoder class.

        Parameters
        ----------
        patch_size : tuple
            Tuple of (time, height, width) indicating the total downsampling factor to reverse.
        """

        # Define test parameters
        batch_size = 3
        time = 2
        height = 4
        width = 4
        in_channels = 128
        out_channels = 3

        hidden_channels = [in_channels, 64, 32, out_channels]

        # Create input tensor (latent representation)
        x = torch.randn(batch_size, time, height, width, in_channels)

        # Initialize decoder
        decoder = Decoder(
            channels=hidden_channels,
            patch_size=patch_size,
        )

        # Forward pass
        output = decoder(x)

        # Calculate expected output shape
        expected_time = time * patch_size[0]
        expected_height = height * patch_size[1]
        expected_width = width * patch_size[2]

        # Check output shape
        assert output.shape == (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            out_channels,
        )

        # Check output values
        assert not torch.isnan(output).any()
        assert not (output == 0).all()

    @pytest.mark.parametrize(
        "hidden_channels",
        [
            [128, 64, 3],
            [128, 64, 32, 3],
            [128, 64, 32, 16, 3],
        ],
    )
    def test_decoder_forward_hidden_channels(self, hidden_channels):
        """
        Test the forward function of the Decoder class.

        Parameters
        ----------
        hidden_channels : list
            List of channel dimensions for each layer of the decoder.
        """

        # Define test parameters
        batch_size = 3
        time = 2
        height = 4
        width = 4
        in_channels = 128
        out_channels = 3

        patch_size = (4, 16, 16)

        # Create input tensor (latent representation)
        x = torch.randn(batch_size, time, height, width, in_channels)

        # Initialize decoder
        decoder = Decoder(
            channels=hidden_channels,
            patch_size=patch_size,
        )

        # Forward pass
        output = decoder(x)

        # Calculate expected output shape
        expected_time = time * patch_size[0]
        expected_height = height * patch_size[1]
        expected_width = width * patch_size[2]

        # Check output shape
        assert output.shape == (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            out_channels,
        )

        # Check output values
        assert not torch.isnan(output).any()
        assert not (output == 0).all()
