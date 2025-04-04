import torch
import torch.nn as nn
import math

from torchvision.ops import stochastic_depth

from metaparc.model.transformer.attention import AttentionBlock
from metaparc.model.transformer.pos_encodings import RotaryPositionalEmbedding
from metaparc.model.transformer.tokenizer import (
    SpatioTemporalTokenization,
    SpatioTemporalDetokenization,
)


def get_model(model_config: dict):
    """Get the model."""
    return PhysicsTransformer(
        input_channels=model_config["input_channels"],
        hidden_dim=model_config["hidden_channels"],
        mlp_dim=model_config["mlp_dim"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        patch_size=model_config["patch_size"],
        num_layers=model_config["num_layers"],
    )


def get_patch_conv_size(patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Get the conv sizes depending on the desired patch size.
    """

    t = int(math.sqrt(patch_size[0]))
    h = int(math.sqrt(patch_size[1]))
    w = int(math.sqrt(patch_size[2]))

    return (t, h, w)


class PhysicsTransformer(nn.Module):
    """
    Physics Transformer model.

    Parameters
    ----------
    input_channels: int
        Number of input channels (physical fields).
    hidden_dim: int
        Hidden dimension inside the attention blocks. Should be divisible by 6 if rotary positional encoding is used.
    mlp_dim: int
        Hidden dimension inside the MLP.
    num_heads: int
        Number of attention heads.
    dropout: float
        Dropout rate.
    patch_size: tuple[int, int, int]
        Patch size for spatial-temporal embeddings. (time, height, width)
    num_layers: int
        Number of attention blocks.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.1,
        patch_size: tuple[int, int, int] = (4, 16, 16),
        num_layers: int = 4,
    ):
        super().__init__()
        self.stochastic_depth_rate = stochastic_depth_rate

        self.pos_encodings = RotaryPositionalEmbedding(dim=hidden_dim, base=10000)
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    att_type="full",
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    pe=self.pos_encodings,
                )
                for _ in range(num_layers)
            ]
        )

        patch_conv_size = get_patch_conv_size(patch_size)
        self.tokenizer = SpatioTemporalTokenization(
            in_channels=input_channels,
            dim_embed=hidden_dim,
            conv1_size=patch_conv_size,
            conv2_size=patch_conv_size,
        )
        self.detokenizer = SpatioTemporalDetokenization(
            dim_embed=hidden_dim,
            out_channels=input_channels,
            conv1_size=patch_conv_size,
            conv2_size=patch_conv_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into patches
        x = self.tokenizer(x)

        # Apply N attention blocks (norm, att, norm, mlp)
        for block in self.attention_blocks:
            x = block(x)
            x = stochastic_depth(
                x, p=self.stochastic_depth_rate, mode="row", training=self.training
            )

        # Apply de-patching
        x = self.detokenizer(x)

        return x
