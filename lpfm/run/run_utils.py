from pathlib import Path


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
