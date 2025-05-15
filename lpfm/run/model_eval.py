"""
Detailed evaluation of the model, its predictions, and the losses.
By: Florian Wiesner
Date: 2025-05-01
"""

from typing import Optional
from pathlib import Path

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lpfm.model.transformer.model import get_model
from lpfm.data.dataset_utils import get_dt_datasets
from lpfm.data.phys_dataset import PhysicsDataset
from lpfm.utils.logger import get_logger
from lpfm.run.run_utils import load_stored_model, find_checkpoint

logger = get_logger(__name__, log_level="INFO")


def load_config(model_path: Path) -> dict:
    config_path = model_path / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def load_model(
    model_dir: Path, device: torch.device, model_config: dict, checkpoint_name: str
) -> torch.nn.Module:
    """Load a model from a checkpoint.

    Parameters
    ----------
    model_path : Path
        Path to the model checkpoint
    device : torch.device
        Device to load the model to
    model_config : dict
        Model configuration dictionary

    Returns
    -------
    torch.nn.Module
        Loaded model
    """
    subdir_name = "val"
    checkpoint_path = find_checkpoint(
        model_dir, subdir_name=subdir_name, specific_checkpoint=checkpoint_name
    )
    data = load_stored_model(checkpoint_path, device, remove_ddp=True)
    model = get_model(model_config)
    model.load_state_dict(data["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


class Evaluator:
    """Thorough evaluation of the model, its predictions, and the losses.

    Parameters
    ----------
    base_path : Path
        Path to the base directory of the model
    num_samples : Optional[int]
        Number of samples to evaluate on
        If None, all samples are evaluated
    batch_size : int
        Batch size
    num_workers : int
        Number of workers for the dataloader
    checkpoint_name : str
        Name of the checkpoint to evaluate
    """

    def __init__(
        self,
        base_path: Path,
        num_samples: Optional[int] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        checkpoint_name: str = "best_model",
    ):
        print("Setting up evaluator")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = load_config(base_path)
        self.model = load_model(
            base_path, self.device, self.config["model"], checkpoint_name
        )

        self.datasets = get_dt_datasets(self.config["data"], split="test")

        self.eval_dir = base_path / "eval"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.no_grad()
    def eval_on_dataset(self, dataset: PhysicsDataset):
        criterion = torch.nn.MSELoss(reduction="none")

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        losses = []
        for i, (x, target) in enumerate(loader):
            # shape of x is (batch_size, T, H, W, C)
            # shape of target is (batch_size, 1, H, W, C)

            print(f"   Evaluating on {i}th batch")
            x = x.to(self.device)
            target = target.to(self.device)
            y = self.model(x)

            loss = criterion(y, target).squeeze()  # remove T dimension
            # get the loss of each sample, dont average accross batches (h,w,c)
            loss = torch.mean(loss, dim=(1, 2, 3))
            losses.append(loss)

        return losses

    def main(self):
        for dataset in self.datasets:
            losses = self.eval_on_dataset(dataset)


if __name__ == "__main__":
    # import os
    # set cuda visible devices
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    base_path = Path("logs")
    model_name = "ti-main-run-0008"
    loss_evaluator = Evaluator(
        base_path=base_path / model_name,
        batch_size=256,
        num_workers=8,
    )
    loss_evaluator.main()
