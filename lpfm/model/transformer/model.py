from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn

from torchvision.ops import stochastic_depth

from lpfm.model.transformer.attention import AttentionBlock
from lpfm.model.transformer.pos_encodings import (
    RotaryPositionalEmbedding,
    AbsPositionalEmbedding,
)
from lpfm.model.transformer.tokenizer import Tokenizer, Detokenizer
from lpfm.model.transformer.norms import RevIN
from lpfm.model.transformer.derivatives import FiniteDifference


@dataclass
class LPFM_Ti:
    hidden_dim: int = 192
    mlp_dim: int = 768
    num_heads: int = 3
    num_layers: int = 12


class LPFM_S:
    hidden_dim: int = 384
    mlp_dim: int = 1536
    num_heads: int = 6
    num_layers: int = 12


class LPFM_M:
    hidden_dim: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    num_layers: int = 12


class LPFM_L:
    hidden_dim: int = 1024
    mlp_dim: int = 4096
    num_heads: int = 14
    num_layers: int = 24


class LPFM_XL:
    hidden_dim: int = 1280
    mlp_dim: int = 5120
    num_heads: int = 16
    num_layers: int = 32


def get_model(model_config: dict):
    """Get the model."""
    transformer_config = model_config["transformer"]
    tokenizer_config = model_config["tokenizer"]

    if model_config["model_config"] == "LPFM_Ti":
        lpfm_config = LPFM_Ti()
    elif model_config["model_config"] == "LPFM_S":
        lpfm_config = LPFM_S()
    elif model_config["model_config"] == "LPFM_M":
        lpfm_config = LPFM_M()
    elif model_config["model_config"] == "LPFM_L":
        lpfm_config = LPFM_L()

    return PhysicsTransformer(
        input_channels=transformer_config["input_channels"],
        hidden_dim=lpfm_config.hidden_dim,
        mlp_dim=lpfm_config.mlp_dim,
        num_heads=lpfm_config.num_heads,
        num_layers=lpfm_config.num_layers,
        pos_enc_mode=transformer_config["pos_enc_mode"],
        img_size=model_config["img_size"],
        patch_size=transformer_config["patch_size"],
        use_derivatives=transformer_config["use_derivatives"],
        tokenizer_mode=tokenizer_config["tokenizer_mode"],
        detokenizer_mode=tokenizer_config["detokenizer_mode"],
        tokenizer_overlap=tokenizer_config["tokenizer_overlap"],
        detokenizer_overlap=tokenizer_config["detokenizer_overlap"],
        detokenizer_squash_time=tokenizer_config["detokenizer_squash_time"],
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
    use_derivatives: bool, optional
        Whether to use derivatives in the model.

    ################################################################
    ########### Tokenizer parameters ###############################
    ################################################################

    tokenizer_mode: str = "linear"
        Tokenizer mode. Can be "linear" or "non_linear".
    detokenizer_mode: str = "linear"
        Detokenizer mode. Can be "linear" or "non_linear".
    detokenizer_squash_time: bool = False
        If True, the time dimension will be squashed into a single time step for the detokenizer.
    tokenizer_net_channels: list[int] = None
        Number of channels in the tokenizer conv_net.
    detokenizer_net_channels: list[int] = None
        Number of channels in the detokenizer conv_net.
    tokenizer_overlap: int = 0
        Number of pixels to overlap between patches for the tokenizer.
    detokenizer_overlap: int = 0
        Number of pixels to overlap between patches for the detokenizer.

    ################################################################
    ########### Training parameters ################################
    ################################################################

    dropout: float = 0.0
        Dropout rate.
    stochastic_depth_rate: float = 0.0
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
        use_derivatives: bool = False,
        tokenizer_mode: str = "linear",
        detokenizer_mode: str = "linear",
        tokenizer_overlap: int = 0,
        detokenizer_overlap: int = 0,
        detokenizer_squash_time: bool = False,
        tokenizer_net_channels: Optional[list[int]] = None,
        detokenizer_net_channels: Optional[list[int]] = None,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = input_channels
        self.squash_time = detokenizer_squash_time

        # Initialize derivatives module
        self.use_derivatives = use_derivatives
        if self.use_derivatives:
            self.derivatives = FiniteDifference(
                num_channels=input_channels, filter_1d="2nd"
            )
            # if derivatives are used, the input channels are multiplied by 4 (original, dt, dh, dw)
            # however, the output channels of the tokenizer are still the original input channels
            self.input_channels *= 4

        # Initialize revin
        self.revin = RevIN(num_channels=self.input_channels)

        self.tokenizer = Tokenizer(
            patch_size=patch_size,
            in_channels=self.input_channels,
            dim_embed=hidden_dim,
            mode=tokenizer_mode,
            conv_net_channels=tokenizer_net_channels,
            overlap=tokenizer_overlap,
        )

        # Initialize positional encodings
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

        # Initialize attention blocks
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

        # Initialize tokenizer and detokenizer

        self.detokenizer = Detokenizer(
            patch_size=patch_size,
            dim_embed=hidden_dim,
            out_channels=input_channels,
            mode=detokenizer_mode,
            conv_net_channels=detokenizer_net_channels,
            overlap=detokenizer_overlap,
            squash_time=detokenizer_squash_time,
            img_size=img_size,
        )

        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = [B, T, H, W, C]
        assert not torch.isnan(x).any(), "Input contains NaNs"

        if self.use_derivatives:
            dt, dh, dw = self.derivatives(x)
            x = torch.cat([x, dt, dh, dw], dim=-1)

        x = self.revin(x, mode="norm")
        # Split into patches
        x = self.tokenizer(x)
        if self.init_pos_encodings is not None:
            x = self.init_pos_encodings(x)

        # Apply N attention blocks (norm, att, norm, mlp)
        for block in self.attention_blocks:
            x = block(x)
            x = stochastic_depth(
                x, p=self.stochastic_depth_rate, mode="row", training=self.training
            )

        # # Apply de-patching
        x = self.detokenizer(x)
        x = self.revin(x, mode="denorm")

        # return last time step
        if not self.squash_time:
            return x[:, -1, ...].unsqueeze(1)
        else:
            return x
