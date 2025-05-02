from pathlib import Path


def find_last_checkpoint(sim_dir: Path, best_model: bool) -> Path:
    """Find the last epoch directory in the simulation directory.

    Parameters
    ----------
    sim_dir : Path
        Path to the simulation directory

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
        d for d in sim_dir.iterdir() if d.is_dir() and d.name.startswith("val_")
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
