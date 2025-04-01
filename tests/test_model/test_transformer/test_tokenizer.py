import pytest
import torch
import torch.nn as nn
from metaparc.model.transformer.tokenizer import Image2Patch, Patch2Image


class TestImage2Patch:
    """
    Tests for the Image2Patch tokenizer class.
    """

    def test_initialization(self):
        """
        Test that the Image2Patch class initializes with correct default parameters.
        """
        dim_embed = 256
        channels = 3
        time_steps = 4
        tokenizer = Image2Patch(
            in_channels=channels,
            time_steps=time_steps,
            dim_embed=dim_embed,
        )
        assert tokenizer.in_channels == channels * time_steps
        assert tokenizer.out_channels == dim_embed * time_steps
        assert tokenizer.time_steps == time_steps

    def test_forward_pass(self):
        """
        Test the forward pass of the Image2Patch class.
        """
        batch_size = 2
        time_steps = 3
        channels = 3
        height = 32
        width = 32
        dim_embed = 256

        tokenizer = Image2Patch(
            in_channels=channels,
            time_steps=time_steps,
            dim_embed=dim_embed,
        )
        x = torch.randn(batch_size, time_steps, channels, height, width)
        output = tokenizer(x)

        # Expected shape: (batch_size, time_steps, dim_embed, height/patch_size, width/patch_size)
        expected_shape = (
            batch_size,
            time_steps,
            dim_embed,
            height // 16,
            width // 16,
        )
        assert output.shape == expected_shape


class TestPatch2Image:
    """
    Tests for the Patch2Image tokenizer class.
    """

    def test_initialization(self):
        """
        Test that the Patch2Image class initializes with correct default parameters.
        """
        dim_embed = 256
        channels = 3
        time_steps = 4
        tokenizer = Patch2Image(
            dim_embed=dim_embed,
            out_channels=channels,
            time_steps=time_steps,
        )
        assert tokenizer.in_channels == dim_embed * time_steps
        assert tokenizer.out_channels == channels * time_steps
        assert tokenizer.time_steps == time_steps

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

        tokenizer = Patch2Image(
            dim_embed=dim_embed,
            out_channels=channels,
            time_steps=time_steps,
        )
        x = torch.randn(batch_size, time_steps, dim_embed, height, width)
        output = tokenizer(x)

        # Expected shape: (batch_size, time_steps, channels, height, width)
        expected_shape = (batch_size, time_steps, channels, height * 16, width * 16)
        assert output.shape == expected_shape
