from pathlib import Path

import yaml
import lpfm.run.train as train


def run_sweep(config: dict):
    trainer = train.Trainer(config)
    trainer.save_config()
    trainer.train()


def main():
    config_path = Path(
        r"C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\lpfm\run\config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    wandb_id = "ti-main-run-single-"
    epochs = [20, 30]
    idx = [2, 3]

    for epoch, id in zip(epochs, idx):
        config["wandb"]["id"] = wandb_id + f"{id:04d}"
        config["training"]["epochs"] = epoch
        run_sweep(config)


if __name__ == "__main__":
    main()
