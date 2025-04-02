from metaparc.model.transformer.model import PhysicsTransformer

import torch


def test_forward():
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        input_channels=3,
        hidden_dim=96,
        num_heads=4,
        dropout=0.0,
    )
    output = transformer(data)
    assert output.shape == (10, 8, 128, 128, 3)
