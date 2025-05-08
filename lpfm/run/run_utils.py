import pathlib
from pathlib import Path

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
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    new_state_dict = {}
    if remove_ddp:
        for key, value in checkpoint["model_state_dict"].items():
            # Check if the key starts with 'module._orig_mod.'
            if key.startswith("module._orig_mod."):
                # Remove the prefix
                new_key = key.replace("module._orig_mod.", "")
                new_state_dict[new_key] = value
            else:
                # Keep the key as is
                new_state_dict[key] = value
        checkpoint["model_state_dict"] = new_state_dict
    return checkpoint


def find_last_checkpoint(sim_dir: Path, subdir_name: str, best_model: bool) -> Path:
    """Find the last epoch directory in the simulation directory.

    Parameters
    ----------
    sim_dir : Path
        Path to the simulation directory

    subdir_name : str
        Subdirectory name to look for in the checkpoint directory

    best_model : bool
        Whether to use the best model for potential restart or the last checkpoint

    Returns
    -------
    Path or None
        Path to the last epoch directory if found, None otherwise
    """
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory {sim_dir} does not exist")

    if best_model:
        checkpoint_path = sim_dir / "best_model.pth"
        if checkpoint_path.exists():
            return checkpoint_path
        else:
            return None

    # Find all directories that match the pattern "epoch_XXXX"
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
