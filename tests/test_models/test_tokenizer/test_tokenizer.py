import pytest
import torch
from gphyt.models.tokenizer.tokenizer import (
    ConvNetDetokenizer,
    ConvNetTokenizer,
    LinearTokenizer,
    LinearDetokenizer,
    Tokenizer,
    Detokenizer,
)


class TestTokenizer:
    """
    Tests for the Tokenizer class.
    """

    @pytest.mark.parametrize("mode", ["linear", "conv_net"])
    def test_tokenizer_forward_pass(self, mode):
        """
        Test that the Tokenizer class forward pass is correct.
        """
        img_size = (4, 64, 32)
        patch_size = (2, 4, 4)
        in_channels = 5
        dim_embed = 128
        batch_size = 2

        conv_net_channels = [16, 32, 64]

        tokenizer = Tokenizer(
            patch_size, in_channels, dim_embed, mode, conv_net_channels
        )
        x = torch.randn(batch_size, *img_size, in_channels)
        output = tokenizer(x)

        num_t_patches = img_size[0] // patch_size[0]
        num_h_patches = img_size[1] // patch_size[1]
        num_w_patches = img_size[2] // patch_size[2]

        assert output.shape == (
            batch_size,
            num_t_patches,
            num_h_patches,
            num_w_patches,
            dim_embed,
        )


class TestDetokenizer:
    """
    Tests for the Detokenizer class.
    """

    @pytest.mark.parametrize("mode", ["linear", "conv_net"])
    def test_detokenizer_forward_pass(self, mode):
        """
        Test that the Detokenizer class forward pass is correct.
        """
        img_size = (4, 64, 32)
        patch_size = (2, 4, 4)
        out_channels = 5
        dim_embed = 128
        batch_size = 2

        conv_net_channels = [64, 32, 16]

        num_t_patches = img_size[0] // patch_size[0]
        num_h_patches = img_size[1] // patch_size[1]
        num_w_patches = img_size[2] // patch_size[2]
        x = torch.randn(
            batch_size, num_t_patches, num_h_patches, num_w_patches, dim_embed
        )

        detokenizer = Detokenizer(
            patch_size=patch_size,
            dim_embed=dim_embed,
            out_channels=out_channels,
            mode=mode,
            conv_net_channels=conv_net_channels,
        )
        output = detokenizer(x)

        assert output.shape == (batch_size, *img_size, out_channels)


class TestLinearTokenizer:
    """
    Tests for the LinearTokenizer class.
    """

    def test_linear_tokenizer_forward_pass(self):
        """
        Test that the LinearTokenizer class initializes correctly.
        """
        batch_size = 2
        img_size = (4, 256, 128)
        patch_size = (2, 4, 4)
        in_channels = 5
        dim_embed = 256

        tokenizer = LinearTokenizer(patch_size, in_channels, dim_embed)
        x = torch.randn(batch_size, *img_size, in_channels)
        output = tokenizer(x)

        num_t_patches = img_size[0] // patch_size[0]
        num_h_patches = img_size[1] // patch_size[1]
        num_w_patches = img_size[2] // patch_size[2]

        assert output.shape == (
            batch_size,
            num_t_patches,
            num_h_patches,
            num_w_patches,
            dim_embed,
        )

    @pytest.mark.parametrize("overlap", [0, 2, 4, 6])
    def test_linear_tokenizer_overlap(self, overlap):
        """
        Test that the LinearTokenizer class initializes correctly.
        """
        batch_size = 2
        img_size = (4, 256, 128)
        patch_size = (2, 16, 16)
        in_channels = 5
        dim_embed = 256
        overlap = 4

        tokenizer = LinearTokenizer(patch_size, in_channels, dim_embed, overlap)
        x = torch.randn(batch_size, *img_size, in_channels)
        output = tokenizer(x)

        num_t_patches = img_size[0] // patch_size[0]
        num_h_patches = img_size[1] // patch_size[1]
        num_w_patches = img_size[2] // patch_size[2]

        assert output.shape == (
            batch_size,
            num_t_patches,
            num_h_patches,
            num_w_patches,
            dim_embed,
        )


class TestLinearDetokenizer:
    """
    Tests for the LinearDetokenizer class.
    """

    def test_linear_detokenizer_forward_pass(self):
        """
        Test that the LinearDetokenizer class shape is correct.
        """
        batch_size = 2
        img_size = (4, 256, 128)
        patch_size = (2, 4, 4)
        out_channels = 5
        dim_embed = 256

        num_t_patches = img_size[0] // patch_size[0]
        num_h_patches = img_size[1] // patch_size[1]
        num_w_patches = img_size[2] // patch_size[2]
        x = torch.randn(
            batch_size, num_t_patches, num_h_patches, num_w_patches, dim_embed
        )

        detokenizer = LinearDetokenizer(patch_size, out_channels, dim_embed)
        output = detokenizer(x)

        assert output.shape == (batch_size, *img_size, out_channels)

    @pytest.mark.parametrize("overlap", [2, 4, 6])
    def test_linear_detokenizer_overlap(self, overlap):
        """
        Test that the LinearDetokenizer class works correctly with overlap.
        """
        batch_size = 2
        img_size = (4, 256, 128)
        patch_size = (2, 16, 16)
        out_channels = 5
        dim_embed = 256

        num_t_patches = img_size[0] // patch_size[0]
        num_h_patches = img_size[1] // patch_size[1]
        num_w_patches = img_size[2] // patch_size[2]

        x = torch.randn(
            batch_size, num_t_patches, num_h_patches, num_w_patches, dim_embed
        )

        detokenizer = LinearDetokenizer(
            patch_size=patch_size,
            out_channels=out_channels,
            dim_embed=dim_embed,
            overlap=overlap,
        )
        output = detokenizer(x)

        assert output.shape == (batch_size, *img_size, out_channels)


class TestConvNetTokenizer:
    """
    Tests for the ConvNetTokenizer class.
    """

    def test_conv_net_tokenizer_forward_pass(self):
        """
        Test the forward pass of the ConvNetTokenizer class.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        batch_size = 2
        time_steps = 8
        height = 64
        width = 64
        in_channels = 3
        dim_embed = 256
        patch_size = (4, 16, 16)  # This will be split into two conv layers
        conv_net_channels = [16, 32, 64]

        tokenizer = ConvNetTokenizer(
            channels=[in_channels, *conv_net_channels, dim_embed],
            patch_size=patch_size,
        )

        x = torch.randn(batch_size, time_steps, height, width, in_channels)
        output = tokenizer(x)

        # Expected shape after tokenization
        expected_time = time_steps // patch_size[0]
        expected_height = height // patch_size[1]
        expected_width = width // patch_size[2]

        expected_shape = (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            dim_embed,
        )
        assert output.shape == expected_shape

    def test_conv_net_pure_spatial_tokenization(self):
        """
        Test ConvNetTokenizer with time size of 1 for pure spatial tokenization.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        batch_size = 2
        time_steps = 8
        height = 64
        width = 64
        in_channels = 3
        dim_embed = 256
        # Using time size of 1 for pure spatial tokenization
        patch_size = (1, 16, 16)
        conv_net_channels = [16, 32, 64]

        tokenizer = ConvNetTokenizer(
            channels=[in_channels, *conv_net_channels, dim_embed],
            patch_size=patch_size,
        )

        x = torch.randn(batch_size, time_steps, height, width, in_channels)
        output = tokenizer(x)

        # Expected shape after tokenization
        expected_height = height // patch_size[1]
        expected_width = width // patch_size[2]
        expected_time = time_steps  # Time dimension should be preserved

        expected_shape = (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            dim_embed,
        )
        assert output.shape == expected_shape


class TestConvNetDetokenizer:
    """
    Tests for the ConvNetDetokenizer tokenizer class.
    """

    def test_forward_pass(self):
        """
        Test the forward pass of the ConvNetDetokenizer class.
        """
        batch_size = 2
        time_steps = 3
        channels = 3
        height = 2
        width = 2
        dim_embed = 256
        patch_size = (4, 16, 16)
        conv_net_channels = [dim_embed, 128, 64, 32, channels]

        tokenizer = ConvNetDetokenizer(
            channels=conv_net_channels,
            patch_size=patch_size,
        )
        x = torch.randn(batch_size, time_steps, height, width, dim_embed)
        output = tokenizer(x)

        expected_time = time_steps * patch_size[0]
        expected_height = height * patch_size[1]
        expected_width = width * patch_size[2]
        expected_shape = (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            channels,
        )

        assert output.shape == expected_shape

    def test_conv_net_pure_spatial_detokenization(self):
        """
        Test ConvNetDetokenizer with time size of 1 for pure spatial detokenization.
        """
        batch_size = 2
        time_steps = 3
        channels = 3
        height = 2
        width = 2
        dim_embed = 256
        patch_size = (1, 16, 16)  # Time size of 1 for pure spatial detokenization
        conv_net_channels = [dim_embed, 128, 64, 32, channels]
        tokenizer = ConvNetDetokenizer(
            channels=conv_net_channels,
            patch_size=patch_size,
        )

        x = torch.randn(batch_size, time_steps, height, width, dim_embed)
        output = tokenizer(x)

        # Time dimension should remain unchanged since patch size has time=1
        expected_time = time_steps
        expected_height = height * patch_size[1]
        expected_width = width * patch_size[2]
        expected_shape = (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            channels,
        )

        assert output.shape == expected_shape
