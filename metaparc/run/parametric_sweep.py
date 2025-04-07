from pathlib import Path

import yaml
import metaparc.run.train as train


def run_sweep(config: dict):
    trainer = train.Trainer(config)
    trainer.save_config()
    trainer.train()


def main():
    config_path = Path("C:/Users/zsa8rk/Coding/MetaPARC/metaparc/run/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    hidden_channels = [
        30,
        60,
        126,
        258,
    ]
    mlp_dim = [128, 256, 512, 1024]
    num_layers = [1, 2, 4, 8]
    num_heads = [1, 2, 4, 8]

    for hidden_channels, mlp_dim, num_layers, num_heads in zip(
        hidden_channels, mlp_dim, num_layers, num_heads
    ):
        config["model"]["hidden_channels"] = hidden_channels
        config["model"]["mlp_dim"] = mlp_dim
        config["model"]["num_layers"] = num_layers
        config["model"]["num_heads"] = num_heads
        config["wandb"]["id"] = (
            f"sweep_{hidden_channels}_{mlp_dim}_{num_layers}_{num_heads}"
        )
        run_sweep(config)


if __name__ == "__main__":
    main()
