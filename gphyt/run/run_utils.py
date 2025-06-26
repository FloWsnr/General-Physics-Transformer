import pathlib
from pathlib import Path
from typing import Optional
import torch


def load_stored_model(
    checkpoint_path: Path, device: torch.device, remove_ddp: bool = False
) -> dict:
    """Load a checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint
    device : torch.device
        Device to load the checkpoint to
    remove_ddp : bool
        Whether to remove the DDP wrapper keys from the model

    Returns
    -------
    dict
        Checkpoint
    """
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    new_state_dict = {}
    if remove_ddp:
        for key, value in checkpoint["model_state_dict"].items():
            # Check if the key starts with 'module._orig_mod.'
            if key.startswith("module._orig_mod."):
                # Remove the prefix
                new_key = key.replace("module._orig_mod.", "")
                new_state_dict[new_key] = value
            elif key.startswith("module."):
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value
            elif key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = value
            else:
                # Keep the key as is
                new_state_dict[key] = value
        checkpoint["model_state_dict"] = new_state_dict
    return checkpoint


def find_checkpoint(
    sim_dir: Path,
    subdir_name: str,
    specific_checkpoint: str = "last_checkpoint",
) -> Path:
    """Find a specific checkpoint in the simulation directory.

    Parameters
    ----------
    sim_dir : Path
        Path to the simulation directory

    subdir_name : str
        Subdirectory name to look for in the checkpoint directory

    specific_checkpoint : str
        Specific checkpoint to look for, either "last_checkpoint",
        "best_model", or a number of a epoch directory as string

    Returns
    -------
    Path or None
        Path to the checkpoint if found, None otherwise
    """
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory {sim_dir} does not exist")

    # if no specific checkpoint is provided, try to find the last checkpoint
    if specific_checkpoint == "last_checkpoint":
        last_checkpoint_path = sim_dir / "last_checkpoint.pth"
        if last_checkpoint_path.exists():
            print(f"Found last checkpoint at {last_checkpoint_path}")
            return last_checkpoint_path
        else:
            print(f"No last checkpoint found at {last_checkpoint_path}")
            print("Checking for epoch directories")
            return _search_last_dir(sim_dir, subdir_name)
    elif specific_checkpoint == "best_model":
        best_model_path = sim_dir / "best_model.pth"
        if best_model_path.exists():
            return best_model_path
        else:
            return None
    else:
        # try to find the checkpoint in the epoch directory
        checkpoint_path = (
            sim_dir / f"{subdir_name}{specific_checkpoint}" / "checkpoint.pth"
        )
        if checkpoint_path.exists():
            return checkpoint_path
        else:
            return None


def _search_last_dir(sim_dir: Path, subdir_name: str) -> Path:
    """Find a checkpoint in an epoch directory.

    The format of the epoch directories is "epoch_XXXX" where XXXX is a number.
    The checkpoint is either in the last epoch directory or the previous one.
    If not found, None is returned.

    Parameters
    ----------
    sim_dir : Path
        Path to the simulation directory

    subdir_name : str
        Subdirectory name to look for in the checkpoint directory

    Returns
    -------
    Path or None
        Path to the checkpoint if found, None otherwise
    """
    epoch_dirs = [
        d for d in sim_dir.iterdir() if d.is_dir() and d.name.startswith(subdir_name)
    ]

    if len(epoch_dirs) == 0:
        return None

    # Sort the directories by their epoch number
    # The format is "epoch_XXXX" where XXXX is a number
    sorted_epoch_dirs = sorted(epoch_dirs, key=lambda x: int(x.name.split("_")[1]))
    last_epoch_dir = sorted_epoch_dirs[-1]

    # the checkpoint could be in the last epoch directory or the previous one
    checkpoint_path = last_epoch_dir / "checkpoint.pth"
    if not checkpoint_path.exists():
        if len(sorted_epoch_dirs) > 1:
            checkpoint_path = sorted_epoch_dirs[-2] / "checkpoint.pth"
            if not checkpoint_path.exists():
                return None
        else:
            return None

    return checkpoint_path


def path_to_string(config: dict) -> dict:
    """Convert all Path objects in the config to strings.

    Parameters
    ----------
    config : dict
        The config to convert.
    """

    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, dict):
            config[key] = path_to_string(value)
    return config


def human_format(num: int | float) -> str:
    """Format a number with SI prefixes (K, M, B).

    Parameters
    ----------
    num : int or float
        The number to format.

    Returns
    -------
    str
        Formatted string with SI prefix.
    """
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000
    return f"{num:.2f}P"
