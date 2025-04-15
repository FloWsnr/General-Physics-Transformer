from pathlib import Path
import argparse
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from dotenv import load_dotenv

from lpfm.run.train import Trainer


def find_last_checkpoint(sim_dir: Path) -> Path:
    """Find the last epoch directory in the simulation directory.

    Parameters
    ----------
    sim_dir : Path
        Path to the simulation directory

    Returns
    -------
    Path or None
        Path to the last epoch directory if found, None otherwise
    """
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory {sim_dir} does not exist")

    # Find all directories that match the pattern "epoch_XXXX"
    epoch_dirs = [
        d for d in sim_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")
    ]

    if len(epoch_dirs) == 0:
        raise FileNotFoundError(f"No epoch directories found in {sim_dir}")

    # Sort the directories by their epoch number
    # The format is "epoch_XXXX" where XXXX is a number
    sorted_epoch_dirs = sorted(epoch_dirs, key=lambda x: int(x.name.split("_")[1]))
    last_epoch_dir = sorted_epoch_dirs[-1]

    # the checkpoint could be in the last epoch directory or the previous one
    checkpoint_path = last_epoch_dir / "checkpoint.pth"
    if not checkpoint_path.exists():
        checkpoint_path = sorted_epoch_dirs[-2] / "checkpoint.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist")

    return checkpoint_path


def main(
    config_path: Path,
    sim_dir: Path,
    restart: bool,
    sim_name: str,
    data_dir: Path,
    time_limit: int,
):
    """Main training function."""
    load_dotenv()

    if restart:
        checkpoint_path = find_last_checkpoint(sim_dir)
        print(f"Restarting from checkpoint {checkpoint_path}")
    else:
        checkpoint_path = None

    # Load config
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)

    ####################################################################
    ########### Augment config #########################################
    ####################################################################
    config["logging"]["log_dir"] = (
        sim_dir.parent
    )  # the actual dir is set in the trainer
    config["logging"]["log_file"] = None  # sim_dir / "logs" / f"{sim_name}.log"
    config["wandb"]["id"] = sim_name
    config["data"]["data_dir"] = data_dir
    config["training"]["time_limit"] = time_limit
    ####################################################################
    ########### Initialize trainer #####################################
    ####################################################################
    trainer = Trainer(config)
    if restart:
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

    main(
        config_path=config_path,
        sim_dir=sim_dir,
        restart=restart,
        sim_name=sim_name,
        data_dir=data_dir,
        time_limit=time_limit_seconds,
    )
