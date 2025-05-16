"""
Detailed evaluation of the model, its predictions, and the losses.
By: Florian Wiesner
Date: 2025-05-01
"""

from pathlib import Path
import platform

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import pandas as pd
import matplotlib.pyplot as plt

import wandb
import wandb.wandb_run

import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from lpfm.model.transformer.model import get_model
from lpfm.data.dataset_utils import get_dt_datasets
from lpfm.data.phys_dataset import PhysicsDataset
from lpfm.utils.logger import get_logger
from lpfm.run.run_utils import load_stored_model, find_checkpoint


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
        batch_size: int = 256,
        num_workers: int = 4,
        checkpoint_name: str = "best_model",
        wandb_run: wandb.Run | None = None,
        global_rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        log_level: str = "INFO",
    ):
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.ddp_enabled = dist.is_initialized()
        self.logger = get_logger(
            "Evaluator",
            log_file=self.config["logging"]["log_file"],
            log_level=self.config["logging"]["log_level"],
        )

        self.config = load_config(base_path)
        self.model = load_model(
            base_path, self.device, self.config["model"], checkpoint_name
        )

        # print the model architecture
        self.model.to(self.device)
        torch.set_float32_matmul_precision("high")
        if not platform.system() == "Windows":
            self.log_msg("Compiling model")
            self.model = torch.compile(self.model)
        if torch.cuda.is_available():
            self.log_msg("Using AMP")
            self.use_amp = True
        else:
            self.use_amp = False

        if self.ddp_enabled:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.device,
            )
        self.model.eval()

        self.datasets = get_dt_datasets(self.config["data"], split="test")

        self.eval_dir = base_path / "eval"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_dataloader(self, dataset: PhysicsDataset, is_distributed: bool = False):
        if is_distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = SequentialSampler(dataset)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    def eval_on_dataset(self, dataset: PhysicsDataset):
        criterion = torch.nn.MSELoss(reduction="none")

        loader = self._get_dataloader(dataset, is_distributed=False)
        losses = []
        with torch.inference_mode():
            for i, (x, target) in enumerate(loader):
                x = x.to(self.device)
                target = target.to(self.device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # shape of x is (batch_size, T, H, W, C)
                    # shape of target is (batch_size, 1, H, W, C)
                    y = self.model(x)
                    loss = criterion(y, target).squeeze()  # remove T dimension
                # get the loss of each sample, dont average accross batches (h,w,c)
                loss = torch.mean(loss, dim=(1, 2, 3))
                losses.append(loss.detach())

        # send losses to all GPUs
        if self.ddp_enabled:
            # Gather losses from all GPUs to rank 0
            gathered_losses = [
                torch.zeros_like(losses[0]) for _ in range(self.world_size)
            ]
            dist.gather(losses[0], gathered_losses, dst=0)
            if self.global_rank == 0:
                # Concatenate all gathered losses
                losses = torch.cat(gathered_losses, dim=0)
            else:
                losses = None

        return losses

    def main(self):
        all_losses = {}
        for name, dataset in self.datasets.items():
            losses = self.eval_on_dataset(dataset)
            all_losses[name] = losses

        # insert losses into df, each col is a dataset, set the name of the col to the dataset name
        df = pd.DataFrame(all_losses)
        df.columns = list(self.datasets.keys())

        df.to_csv(self.eval_dir / "losses.csv", index=False)

    def _log_losses(self, df: pd.DataFrame):
        # compute mean and std of losses
        mean_losses = df.mean(axis=0)
        std_losses = df.std(axis=0)

        self.wandb_run.log({"validation-summary/eval-loss-mean": mean_losses})
        self.wandb_run.log({"validation-summary/eval-loss-std": std_losses})


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
