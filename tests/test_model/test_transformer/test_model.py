from lpfm.model.transformer.model import PhysicsTransformer

import torch


def test_forward():
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        input_channels=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_cuda():
    data = torch.randn(10, 8, 128, 128, 3).cuda()
    transformer = PhysicsTransformer(
        input_channels=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        patch_size=(4, 16, 16),
        num_layers=4,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        stochastic_depth_rate=0.0,
        pos_enc_mode="rope",
    )
    transformer.cuda()
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_absolute_pos_enc():
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        input_channels=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="absolute",
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)


def test_forward_conv_net_tokenizer():
    # (batch_size, time, height, width, channels)
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        input_channels=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="conv_net",
        detokenizer_mode="conv_net",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="absolute",
        tokenizer_net_channels=[16, 32, 64, 128],
        detokenizer_net_channels=[128, 64, 32, 16],
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)

def test_forward_linear_tokenizer_overlap():
    data = torch.randn(10, 8, 128, 128, 3)

    transformer = PhysicsTransformer(
        input_channels=3,
        hidden_dim=96,
        mlp_dim=256,
        num_heads=4,
        dropout=0.0,
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        img_size=(8, 128, 128),
        patch_size=(4, 16, 16),
        num_layers=4,
        stochastic_depth_rate=0.0,
        pos_enc_mode="absolute",
        tokenizer_overlap=2,
        detokenizer_overlap=2,
    )
    output = transformer(data)
    assert output.shape == (10, 1, 128, 128, 3)
