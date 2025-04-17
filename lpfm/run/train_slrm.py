import os
from pathlib import Path
import argparse
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from dotenv import load_dotenv

import torch.distributed as dist
from lpfm.run.train import Trainer
from lpfm.run.run_utils import find_last_checkpoint


def setup_ddp(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(
    config_path: Path,
    sim_dir: Path,
    restart: bool,
    sim_name: str,
    data_dir: Path,
    time_limit: int,
    global_rank: int,
    local_rank: int,
    world_size: int,
):
    """Main training function."""
    load_dotenv()

    if world_size > 1:
        setup_ddp(global_rank, world_size)

    if restart:
        checkpoint_path = find_last_checkpoint(sim_dir)
        if checkpoint_path is None:
            print("No checkpoint found, starting from scratch")
        else:
            print(f"Restarting from checkpoint {checkpoint_path}")
    else:
        checkpoint_path = None

    # Load config
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
        print(config)

    ####################################################################
    ########### Augment config #########################################
    ####################################################################
    config["logging"]["log_dir"] = (
        sim_dir.parent
    )  # the actual dir is set in the trainer
    config["logging"]["log_file"] = sim_dir / f"{sim_name}.log"
    config["wandb"]["id"] = sim_name
    config["data"]["data_dir"] = data_dir
    config["training"]["time_limit"] = time_limit
    ####################################################################
    ########### Initialize trainer #####################################
    ####################################################################
    trainer = Trainer(config, global_rank, local_rank, world_size)
    if restart and checkpoint_path is not None:
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)
    trainer.save_config()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--sim_dir", type=str)
    parser.add_argument("--restart", action=argparse.BooleanOptionalAction)
    parser.add_argument("--sim_name", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--time_limit", type=str)
    args = parser.parse_args()

    config_path = Path(args.config_file)
    sim_dir = Path(args.sim_dir)
    data_dir = Path(args.data_dir)
    restart = args.restart
    sim_name = args.sim_name
    time_limit = args.time_limit

    # convert time_limit to seconds
    time_limit_seconds = sum(
        x * int(t) for x, t in zip([3600, 60, 1], time_limit.split(":"))
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))

    main(
        config_path=config_path,
        sim_dir=sim_dir,
        sim_name=sim_name,
        data_dir=data_dir,
        restart=restart,
        time_limit=time_limit_seconds,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
