from pathlib import Path

import yaml
import lpfm.run.train as train


def run_sweep(config: dict):
    trainer = train.Trainer(config)
    trainer.save_config()
    trainer.train()


def main():
    config_path = Path("C:/Users/zsa8rk/Coding/MetaPARC/metaparc/run/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    datasets = [
        "cylinder_sym_flow_water",
        "cylinder_pipe_flow_water",
        "object_periodic_flow_water",
        "object_sym_flow_water",
        "object_sym_flow_air",
        "heated_object_pipe_flow_air",
        "turbulent_radiative_layer_2D",
        "rayleigh_benard",
        "shear_flow",
        # "euler",
    ]

    config["wandb"]["tags"] = config["wandb"]["tags"] + ["DatasetSweep"]

    for dataset in datasets:
        config["data"]["datasets"] = [dataset]
        config["wandb"]["id"] = f"sweep_{dataset}"
        config["wandb"]["notes"] = f"DatasetSweep - {dataset}"
        try:
            run_sweep(config)
        except Exception as e:
            print(f"Error running sweep for {dataset}: {e}")


if __name__ == "__main__":
    main()
