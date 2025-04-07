import torch
import torch.nn as nn

from torchvision.ops import stochastic_depth

from metaparc.model.transformer.attention import AttentionBlock
from metaparc.model.transformer.pos_encodings import (
    RotaryPositionalEmbedding,
    AbsPositionalEmbedding,
)
from metaparc.model.transformer.tokenizer import Tokenizer, Detokenizer


def get_model(model_config: dict):
    """Get the model."""
    return PhysicsTransformer(
        input_channels=model_config["input_channels"],
        hidden_dim=model_config["hidden_channels"],
        mlp_dim=model_config["mlp_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        img_size=model_config["img_size"],
        patch_size=model_config["patch_size"],
        tokenizer_mode=model_config["tokenizer_mode"],
        dropout=model_config["dropout"],
        stochastic_depth_rate=model_config["stochastic_depth_rate"],
    )


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
    num_layers: int
        Number of attention blocks.
    patch_size: tuple[int, int, int]
        Patch size for spatial-temporal embeddings. (time, height, width)
    img_size: tuple[int, int, int]
        Incoming image size (time, height, width)
    tokenizer_mode: str
        Tokenizer mode. Can be "linear" or "conv3d".
    dropout: float
        Dropout rate.
    stochastic_depth_rate: float
        Stochastic depth rate.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int,
        img_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        tokenizer_mode: str,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        self.stochastic_depth_rate = stochastic_depth_rate

        self.pos_encodings = RotaryPositionalEmbedding(dim=hidden_dim, base=10000)
        # self.pos_encodings = AbsPositionalEmbedding(
        #     num_channels=input_channels,
        #     time=img_size[0] // patch_size[0],
        #     height=img_size[1] // patch_size[1],
        #     width=img_size[2] // patch_size[2],
        # )
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

        self.tokenizer = Tokenizer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=input_channels,
            dim_embed=hidden_dim,
            mode=tokenizer_mode,
        )
        self.detokenizer = Detokenizer(
            img_size=img_size,
            patch_size=patch_size,
            dim_embed=hidden_dim,
            out_channels=input_channels,
            mode=tokenizer_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into patches
        x = self.tokenizer(x)
        # x = self.pos_encodings(x)

        # x = self.mlp(x)
        # Apply N attention blocks (norm, att, norm, mlp)
        for block in self.attention_blocks:
            x = block(x)
            # x = stochastic_depth(
            #     x, p=self.stochastic_depth_rate, mode="row", training=self.training
            # )

        # Apply de-patching
        x = self.detokenizer(x)

        return x
