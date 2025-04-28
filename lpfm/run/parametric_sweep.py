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
    datasets = [
        "cylinder_sym_flow_water",
        "cylinder_pipe_flow_water",
        "object_periodic_flow_water",
        "object_sym_flow_water",
        "object_sym_flow_air",
        "heated_object_pipe_flow_air",
    ]
    idx = range(8, 8 + len(datasets))

    for id, dataset in zip(idx, datasets):
        config["wandb"]["id"] = wandb_id + f"{id:04d}"
        config["data"]["datasets"] = [dataset]
        run_sweep(config)


if __name__ == "__main__":
    main()
