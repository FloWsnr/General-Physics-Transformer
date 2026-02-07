from typing import Optional, Literal

import torch
import torch.nn as nn

from gphyt.models.transformer.attention import AttentionBlock
from gphyt.models.transformer.pos_encodings import (
    RotaryPositionalEmbedding,
    AbsPositionalEmbedding,
)
from gphyt.models.tokenizer.tokenizer import Tokenizer, Detokenizer
from gphyt.models.transformer.derivatives import FiniteDifference
from gphyt.models.transformer.num_integration import Euler, RK4, Heun
import gphyt.models.model_specs as model_specs


def get_model(model_config: dict) -> "PhysicsTransformer":
    """Get the model."""
    transformer_config: dict = model_config["transformer"]
    tokenizer_config: dict = model_config["tokenizer"]

    if transformer_config["model_size"] == "GPT_S":
        gpt_config = model_specs.GPT_S()
    elif transformer_config["model_size"] == "GPT_M":
        gpt_config = model_specs.GPT_M()
    elif transformer_config["model_size"] == "GPT_L":
        gpt_config = model_specs.GPT_L()
    elif transformer_config["model_size"] == "GPT_XL":
        gpt_config = model_specs.GPT_XL()
    else:
        raise ValueError(f"Invalid model size: {transformer_config['model_size']}")

    return PhysicsTransformer(
        num_fields=transformer_config["input_channels"],
        hidden_dim=gpt_config.hidden_dim,
        mlp_dim=gpt_config.mlp_dim,
        num_heads=gpt_config.num_heads,
        num_layers=gpt_config.num_layers,
        att_mode=transformer_config.get("att_mode", "full"),
        integrator=transformer_config.get("integrator", "Euler"),
        pos_enc_mode=transformer_config["pos_enc_mode"],
        img_size=model_config["img_size"],
        patch_size=transformer_config["patch_size"],
        use_derivatives=transformer_config["use_derivatives"],
        tokenizer_mode=tokenizer_config["tokenizer_mode"],
        detokenizer_mode=tokenizer_config["detokenizer_mode"],
        tokenizer_overlap=tokenizer_config["tokenizer_overlap"],
        detokenizer_overlap=tokenizer_config["detokenizer_overlap"],
        tokenizer_net_channels=gpt_config.conv_channels,
        detokenizer_net_channels=gpt_config.conv_channels,
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

    num_fields: int
        Number of input fields (physical fields).
    hidden_dim: int
        Hidden dimension inside the attention blocks.
        Should be divisible by 6 if Rope positional encoding is used.
    mlp_dim: int
        Hidden dimension inside the MLP.
    num_heads: int
        Number of attention heads.
    num_layers: int
        Number of attention blocks.
    pos_enc_mode: Literal["rope", "absolute"] = "rope"
        Position encoding mode. Can be "rope" or "absolute".
    patch_size: tuple[int, int, int]
        Patch size for spatial-temporal embeddings. (time, height, width)
    att_mode: Literal["full"] = "full"
        Attention mode. Can be "full".
    integrator: str
        Integrator to use
    img_size: tuple[int, int, int]
        Incoming image size (time, height, width)
    use_derivatives: bool, optional
        Whether to use derivatives in the model.

    ################################################################
    ########### Tokenizer parameters ###############################
    ################################################################

    tokenizer_mode: Literal["linear", "conv_net"] = "linear"
        Tokenizer mode. Can be "linear" or "conv_net".
    detokenizer_mode: Literal["linear", "conv_net"] = "linear"
        Detokenizer mode. Can be "linear" or "conv_net".
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
        num_fields: int,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        patch_size: tuple[int, int, int],
        img_size: tuple[int, int, int],
        use_derivatives: bool = False,
        pos_enc_mode: Literal["rope", "absolute"] = "absolute",
        att_mode: Literal["full"] = "full",
        integrator: str = "Euler",
        tokenizer_mode: Literal["linear", "conv_net"] = "linear",
        detokenizer_mode: Literal["linear", "conv_net"] = "linear",
        tokenizer_overlap: int = 0,
        detokenizer_overlap: int = 0,
        tokenizer_net_channels: Optional[list[int]] = None,
        detokenizer_net_channels: Optional[list[int]] = None,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()

        # differentiate between actual fields and input channels, which can be more due to derivatives
        num_input_channels = num_fields
        self.num_fields = num_fields
        self.att_mode = att_mode

        n_patch_t = img_size[0] // patch_size[0]
        n_patch_h = img_size[1] // patch_size[1]
        n_patch_w = img_size[2] // patch_size[2]

        # Initialize derivatives module
        self.use_derivatives = use_derivatives
        if self.use_derivatives:
            self.derivatives = FiniteDifference(
                num_channels=num_fields, filter_1d="2nd"
            )
            # if derivatives are used, the input channels are multiplied by 4 (original, dt, dh, dw)
            # however, the output channels of the tokenizer are still the original input channels
            num_input_channels *= 4

        if integrator == "Euler":
            self.integrator = Euler()
        elif integrator == "RK4":
            self.integrator = RK4()
        elif integrator == "Heun":
            self.integrator = Heun()
        else:
            self.integrator = None

        self.tokenizer = Tokenizer(
            patch_size=patch_size,
            in_channels=num_input_channels,
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
                time=n_patch_t,
                height=n_patch_h,
                width=n_patch_w,
            )
            att_pos_encodings = None
        else:
            raise ValueError(f"Invalid positional encoding mode: {pos_enc_mode}")

        # Initialize attention blocks
        self.attention_blocks = nn.Sequential(
            *[
                AttentionBlock(
                    att_type=att_mode,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    time=n_patch_t,
                    height=n_patch_h,
                    width=n_patch_w,
                    dropout=dropout,
                    stochastic_depth_rate=stochastic_depth_rate,
                    pe=att_pos_encodings,
                )
                for _ in range(num_layers)
            ]
        )

        # Initialize tokenizer and detokenizer

        self.detokenizer = Detokenizer(
            patch_size=patch_size,
            dim_embed=hidden_dim,
            out_channels=self.num_fields,  # important to set to num_fields
            mode=detokenizer_mode,
            conv_net_channels=detokenizer_net_channels,
            overlap=detokenizer_overlap,
            img_size=img_size,
        )

    def differentiate(self, x: torch.Tensor) -> torch.Tensor:
        assert not torch.isnan(x).any(), "Input contains NaNs"

        if self.use_derivatives:
            dt, dh, dw = self.derivatives(x)
            x = torch.cat([x, dt, dh, dw], dim=-1)

        # Split into patches
        x = self.tokenizer(x)
        if self.init_pos_encodings is not None:
            x = self.init_pos_encodings(x)

        # Apply N attention blocks (norm, att, norm, mlp)
        x = self.attention_blocks(x)

        # # Apply de-patching
        x = self.detokenizer(x)
        return x

    def forward(self, x: torch.Tensor, step_size: float = 1.0) -> torch.Tensor:
        out = self.differentiate(x) + x
        return out[:, -1, ...].unsqueeze(1)
