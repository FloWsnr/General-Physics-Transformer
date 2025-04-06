import pytest
import torch
import torch.nn as nn
from metaparc.model.transformer.tokenizer import (
    Conv3D_Detokenizer,
    Conv3D_Tokenizer,
    LinearTokenizer,
    LinearDetokenizer,
    Tokenizer,
    Detokenizer,
)


class TestTokenizer:
    """
    Tests for the Tokenizer class.
    """

    @pytest.mark.parametrize("mode", ["linear", "conv3d"])
    def test_tokenizer_forward_pass(self, mode):
        """
        Test that the Tokenizer class forward pass is correct.
        """
        img_size = (4, 64, 32)
        patch_size = (2, 4, 4)
        in_channels = 5
        dim_embed = 128
        batch_size = 2

        tokenizer = Tokenizer(img_size, patch_size, in_channels, dim_embed, mode)
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

    @pytest.mark.parametrize("mode", ["linear", "conv3d"])
    def test_detokenizer_forward_pass(self, mode):
        """
        Test that the Detokenizer class forward pass is correct.
        """
        img_size = (4, 64, 32)
        patch_size = (2, 4, 4)
        out_channels = 5
        dim_embed = 128
        batch_size = 2

        num_t_patches = img_size[0] // patch_size[0]
        num_h_patches = img_size[1] // patch_size[1]
        num_w_patches = img_size[2] // patch_size[2]
        x = torch.randn(
            batch_size, num_t_patches, num_h_patches, num_w_patches, dim_embed
        )

        detokenizer = Detokenizer(
            img_size=img_size,
            patch_size=patch_size,
            dim_embed=dim_embed,
            out_channels=out_channels,
            mode=mode,
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

        tokenizer = LinearTokenizer(img_size, patch_size, in_channels, dim_embed)
        x = torch.randn(batch_size, *img_size, in_channels)
        output = tokenizer(x)

        num_t_patches = img_size[0] // patch_size[0]
        num_h_patches = img_size[1] // patch_size[1]
        num_w_patches = img_size[2] // patch_size[2]
        patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]

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
        in_channels = 5
        dim_embed = 256

        num_t_patches = img_size[0] // patch_size[0]
        num_h_patches = img_size[1] // patch_size[1]
        num_w_patches = img_size[2] // patch_size[2]
        x = torch.randn(
            batch_size, num_t_patches, num_h_patches, num_w_patches, dim_embed
        )

        detokenizer = LinearDetokenizer(
            img_size, patch_size, out_channels=in_channels, dim_embed=dim_embed
        )
        output = detokenizer(x)

        assert output.shape == (batch_size, *img_size, in_channels)


class TestConv3DTokenizer:
    """
    Tests for the Conv3DTokenizer class.
    """

    def test_conv3d_tokenizer_forward_pass(self):
        """
        Test the forward pass of the Conv3DTokenizer class.

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

        tokenizer = Conv3D_Tokenizer(
            in_channels=in_channels,
            dim_embed=dim_embed,
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

    def test_conv3d_pure_spatial_tokenization(self):
        """
        Test Conv3DTokenizer with time size of 1 for pure spatial tokenization.

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

        tokenizer = Conv3D_Tokenizer(
            in_channels=in_channels,
            dim_embed=dim_embed,
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


class TestConv3D_Detokenizer:
    """
    Tests for the Conv3D_Detokenizer tokenizer class.
    """

    def test_forward_pass(self):
        """
        Test the forward pass of the Conv3D_Detokenizer class.
        """
        batch_size = 2
        time_steps = 3
        channels = 3
        height = 2
        width = 2
        dim_embed = 256
        patch_size = (4, 16, 16)

        tokenizer = Conv3D_Detokenizer(
            dim_embed=dim_embed,
            out_channels=channels,
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

    def test_conv3d_pure_spatial_detokenization(self):
        """
        Test Conv3D_Detokenizer with time size of 1 for pure spatial detokenization.
        """
        batch_size = 2
        time_steps = 3
        channels = 3
        height = 2
        width = 2
        dim_embed = 256
        patch_size = (1, 16, 16)  # Time size of 1 for pure spatial detokenization

        tokenizer = Conv3D_Detokenizer(
            dim_embed=dim_embed,
            out_channels=channels,
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
