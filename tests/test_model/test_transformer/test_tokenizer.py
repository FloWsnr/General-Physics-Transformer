import pytest
import torch
import torch.nn as nn
from metaparc.model.transformer.tokenizer import (
    Conv3D_Detokenizer,
    Conv3D_Tokenizer,
    LinearTokenizer,
    LinearDetokenizer,
)


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

        detokenizer = LinearDetokenizer(img_size, patch_size, out_channels=in_channels, dim_embed=dim_embed)
        x = torch.randn(batch_size, num_t_patches, num_h_patches, num_w_patches, dim_embed)
        output = detokenizer(x)

        assert output.shape == (batch_size, *img_size, in_channels)

class TestConv3DTokenizer:
    """
    Tests for the Conv3DTokenizer class.
    """

    def test_conv3d_tokenizer_initialization(self):
        """
        Test that the Conv3DTokenizer class initializes correctly.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        in_channels = 3
        dim_embed = 256
        conv1_size = (2, 4, 4)
        conv2_size = (2, 4, 4)

        tokenizer = Conv3D_Tokenizer(
            in_channels=in_channels,
            dim_embed=dim_embed,
            conv1_size=conv1_size,
            conv2_size=conv2_size,
        )

        # Check that the tokenizer has the expected attributes
        assert isinstance(tokenizer.token_net, nn.Sequential)
        assert len(tokenizer.token_net) == 6  # 2 conv layers, 2 norm layers, 2 GELU

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
        conv1_size = (2, 4, 4)
        conv2_size = (2, 4, 4)

        tokenizer = Conv3D_Tokenizer(
            in_channels=in_channels,
            dim_embed=dim_embed,
            conv1_size=conv1_size,
            conv2_size=conv2_size,
        )

        x = torch.randn(batch_size, time_steps, height, width, in_channels)
        output = tokenizer(x)

        # Expected shape after tokenization
        expected_time = time_steps // (conv1_size[0] * conv2_size[0])
        # The spatial dimensions are reduced by the product of conv strides
        expected_height = height // (conv1_size[1] * conv2_size[1])
        expected_width = width // (conv1_size[2] * conv2_size[2])

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
        conv1_size = (1, 4, 4)
        conv2_size = (1, 4, 4)

        tokenizer = Conv3D_Tokenizer(
            in_channels=in_channels,
            dim_embed=dim_embed,
            conv1_size=conv1_size,
            conv2_size=conv2_size,
        )

        x = torch.randn(batch_size, time_steps, height, width, in_channels)
        output = tokenizer(x)

        # Expected shape after tokenization
        expected_height = height // (conv1_size[1] * conv2_size[1])
        expected_width = width // (conv1_size[2] * conv2_size[2])
        expected_time = time_steps  # Time dimension should be preserved

        expected_shape = (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            dim_embed,
        )
        assert output.shape == expected_shape


class TestSpatioTemporalDetokenization:
    """
    Tests for the Conv3D_Detokenizer tokenizer class.
    """

    def test_initialization(self):
        """
        Test that the Conv3D_Detokenizer class initializes with correct default parameters.
        """
        dim_embed = 256
        channels = 3
        conv1_size = (2, 4, 4)
        conv2_size = (2, 4, 4)
        tokenizer = Conv3D_Detokenizer(
            dim_embed=dim_embed,
            out_channels=channels,
            conv1_size=conv1_size,
            conv2_size=conv2_size,
        )
        assert tokenizer.in_channels == dim_embed
        assert tokenizer.out_channels == channels

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
        conv1_size = (2, 4, 4)
        conv2_size = (2, 4, 4)

        tokenizer = Conv3D_Detokenizer(
            dim_embed=dim_embed,
            out_channels=channels,
            conv1_size=conv1_size,
            conv2_size=conv2_size,
        )
        x = torch.randn(batch_size, time_steps, height, width, dim_embed)
        output = tokenizer(x)

        expected_time = time_steps * (conv1_size[0] * conv2_size[0])
        expected_height = height * (conv1_size[1] * conv2_size[1])
        expected_width = width * (conv1_size[2] * conv2_size[2])
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
        conv1_size = (1, 4, 4)  # Time size of 1 for pure spatial detokenization
        conv2_size = (1, 4, 4)  # Time size of 1 for pure spatial detokenization

        tokenizer = Conv3D_Detokenizer(
            dim_embed=dim_embed,
            out_channels=channels,
            conv1_size=conv1_size,
            conv2_size=conv2_size,
        )

        x = torch.randn(batch_size, time_steps, height, width, dim_embed)
        output = tokenizer(x)

        # Time dimension should remain unchanged since conv sizes have time=1
        expected_time = time_steps
        expected_height = height * (conv1_size[1] * conv2_size[1])
        expected_width = width * (conv1_size[2] * conv2_size[2])
        expected_shape = (
            batch_size,
            expected_time,
            expected_height,
            expected_width,
            channels,
        )

        assert output.shape == expected_shape
