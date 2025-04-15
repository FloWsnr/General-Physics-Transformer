from pathlib import Path
import argparse

def find_last_checkpoint(checkpoint_dir: Path):
    """Find the last checkpoint in the checkpoint directory."""
    return sorted(checkpoint_dir.glob("*.pt"))[-1]


def main(checkpoint_dir: Path):
    """Main function."""
    print(f"Loading checkpoint from {checkpoint_dir}")
    checkpoint_path = find_last_checkpoint(checkpoint_dir)
    print(f"Loading checkpoint from {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    print(f"Loading checkpoint from {checkpoint_dir}")

    main(checkpoint_dir)
