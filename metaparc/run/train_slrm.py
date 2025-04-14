from pathlib import Path


def find_last_checkpoint(checkpoint_dir: Path):
    """Find the last checkpoint in the checkpoint directory."""
    return sorted(checkpoint_dir.glob("*.pt"))[-1]


def main():
    """Main function."""
    checkpoint_dir = Path("checkpoints")
    checkpoint_path = find_last_checkpoint(checkpoint_dir)
    print(f"Loading checkpoint from {checkpoint_path}")


if __name__ == "__main__":
    main()  
