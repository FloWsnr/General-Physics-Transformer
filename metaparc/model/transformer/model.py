import torch
import torch.nn as nn

from metaparc.model.transformer.ax_attention import AttentionBlock


class PhysicsTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention_block = AttentionBlock(hidden_dim, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_block(x)
