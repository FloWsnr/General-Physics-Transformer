import torch
import torch.nn as nn

from torchvision.ops import stochastic_depth

from metaparc.model.transformer.attention import AttentionBlock
from metaparc.model.transformer.pos_encodings import (
    RotaryPositionalEmbedding,
    AbsPositionalEmbedding,
)
from metaparc.model.transformer.tokenizer import Tokenizer, Detokenizer
from metaparc.model.transformer.norms import RevIN


def get_model(model_config: dict):
    """Get the model."""
    transformer_config = model_config["transformer"]
    tokenizer_config = model_config["tokenizer"]
    return PhysicsTransformer(
        input_channels=transformer_config["input_channels"],
        hidden_dim=transformer_config["hidden_channels"],
        mlp_dim=transformer_config["mlp_dim"],
        num_heads=transformer_config["num_heads"],
        num_layers=transformer_config["num_layers"],
        pos_enc_mode=transformer_config["pos_enc_mode"],
        img_size=model_config["img_size"],
        patch_size=transformer_config["patch_size"],
        tokenizer_mode=tokenizer_config["tokenizer_mode"],
        detokenizer_mode=tokenizer_config["detokenizer_mode"],
        tokenizer_net_channels=tokenizer_config["tokenizer_net_channels"],
        detokenizer_net_channels=tokenizer_config["detokenizer_net_channels"],
        dropout=transformer_config["dropout"],
        stochastic_depth_rate=transformer_config["stochastic_depth_rate"],
    )


class PhysicsTransformer(nn.Module):
    """
    Physics Transformer model.

    Parameters
    ----------
    ################################################################
    ########### Transformer parameters #############################
    ################################################################

    input_channels: int
        Number of input channels (physical fields).
    hidden_dim: int
        Hidden dimension inside the attention blocks.
        Should be divisible by 6 if Rope positional encoding is used.
    mlp_dim: int
        Hidden dimension inside the MLP.
    num_heads: int
        Number of attention heads.
    num_layers: int
        Number of attention blocks.
    pos_enc_mode: str
        Position encoding mode. Can be "rope" or "absolute".
    patch_size: tuple[int, int, int]
        Patch size for spatial-temporal embeddings. (time, height, width)
    img_size: tuple[int, int, int]
        Incoming image size (time, height, width)

    ################################################################
    ########### Tokenizer parameters ###############################
    ################################################################

    tokenizer_mode: str
        Tokenizer mode. Can be "linear" or "non_linear".
    detokenizer_mode: str
        Detokenizer mode. Can be "linear" or "non_linear".
    tokenizer_net_channels: list[int]
        Number of channels in the tokenizer conv_net.
    detokenizer_net_channels: list[int]
        Number of channels in the detokenizer conv_net.

    ################################################################
    ########### Training parameters ################################
    ################################################################

    dropout: float
        Dropout rate.
    stochastic_depth_rate: float
        Stochastic depth rate.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        pos_enc_mode: str,
        patch_size: tuple[int, int, int],
        img_size: tuple[int, int, int],
        tokenizer_mode: str,
        detokenizer_mode: str,
        tokenizer_net_channels: list[int] | None = None,
        detokenizer_net_channels: list[int] | None = None,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        self.stochastic_depth_rate = stochastic_depth_rate

        if pos_enc_mode == "rope":
            att_pos_encodings = RotaryPositionalEmbedding(dim=hidden_dim, base=10000)
            self.init_pos_encodings = None
        elif pos_enc_mode == "absolute":
            self.init_pos_encodings = AbsPositionalEmbedding(
                num_channels=hidden_dim,
                time=img_size[0] // patch_size[0],
                height=img_size[1] // patch_size[1],
                width=img_size[2] // patch_size[2],
            )
            att_pos_encodings = None
        else:
            raise ValueError(f"Invalid positional encoding mode: {pos_enc_mode}")

        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    att_type="full",
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    pe=att_pos_encodings,
                )
                for _ in range(num_layers)
            ]
        )

        self.tokenizer = Tokenizer(
            patch_size=patch_size,
            in_channels=input_channels,
            dim_embed=hidden_dim,
            mode=tokenizer_mode,
            conv_net_channels=tokenizer_net_channels,
        )
        self.detokenizer = Detokenizer(
            patch_size=patch_size,
            dim_embed=hidden_dim,
            out_channels=input_channels,
            mode=detokenizer_mode,
            conv_net_channels=detokenizer_net_channels,
        )

        self.revin = RevIN(num_channels=input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.revin(x, mode="norm")
        # Split into patches
        x = self.tokenizer(x)
        # if self.init_pos_encodings is not None:
        #     x = self.init_pos_encodings(x)

        # # Apply N attention blocks (norm, att, norm, mlp)
        # for block in self.attention_blocks:
        #     x = block(x)
        #     x = stochastic_depth(
        #         x, p=self.stochastic_depth_rate, mode="row", training=self.training
        #     )

        # # Apply de-patching
        x = self.detokenizer(x)
        # x = self.revin(x, mode="denorm")

        return x
