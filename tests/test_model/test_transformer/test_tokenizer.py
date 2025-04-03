import pytest
import torch
import torch.nn as nn
from metaparc.model.transformer.tokenizer import (
    SpatioTemporalDetokenization,
    SpatioTemporalTokenization,
)


class TestSpatioTemporalTokenization:
    """
    Tests for the SpatioTemporalTokenization class.
    """

    def test_spatio_temporal_initialization(self):
        """
        Test that the SpatioTemporalTokenization class initializes correctly.

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

        tokenizer = SpatioTemporalTokenization(
            in_channels=in_channels,
            dim_embed=dim_embed,
            conv1_size=conv1_size,
            conv2_size=conv2_size,
        )

        # Check that the tokenizer has the expected attributes
        assert isinstance(tokenizer.token_net, nn.Sequential)
        assert len(tokenizer.token_net) == 6  # 2 conv layers, 2 norm layers, 2 GELU

    def test_spatio_temporal_forward_pass(self):
        """
        Test the forward pass of the SpatioTemporalTokenization class.

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

        tokenizer = SpatioTemporalTokenization(
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

    def test_spatio_temporal_pure_spatial_tokenization(self):
        """
        Test SpatioTemporalTokenization with time size of 1 for pure spatial tokenization.

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

        tokenizer = SpatioTemporalTokenization(
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
    Tests for the SpatioTemporalDetokenization tokenizer class.
    """

    def test_initialization(self):
        """
        Test that the SpatioTemporalDetokenization class initializes with correct default parameters.
        """
        dim_embed = 256
        channels = 3
        conv1_size = (2, 4, 4)
        conv2_size = (2, 4, 4)
        tokenizer = SpatioTemporalDetokenization(
            dim_embed=dim_embed,
            out_channels=channels,
            conv1_size=conv1_size,
            conv2_size=conv2_size,
        )
        assert tokenizer.in_channels == dim_embed
        assert tokenizer.out_channels == channels

    def test_forward_pass(self):
        """
        Test the forward pass of the Patch2Image class.
        """
        batch_size = 2
        time_steps = 3
        channels = 3
        height = 2
        width = 2
        dim_embed = 256
        conv1_size = (2, 4, 4)
        conv2_size = (2, 4, 4)

        tokenizer = SpatioTemporalDetokenization(
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

    def test_spatio_temporal_pure_spatial_detokenization(self):
        """
        Test SpatioTemporalDetokenization with time size of 1 for pure spatial detokenization.
        """
        batch_size = 2
        time_steps = 3
        channels = 3
        height = 2
        width = 2
        dim_embed = 256
        conv1_size = (1, 4, 4)  # Time size of 1 for pure spatial detokenization
        conv2_size = (1, 4, 4)  # Time size of 1 for pure spatial detokenization

        tokenizer = SpatioTemporalDetokenization(
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
