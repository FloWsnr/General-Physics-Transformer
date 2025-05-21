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
    model = get_model(model_config)
    model.to(device)
    data = load_stored_model(checkpoint_path, device, remove_ddp=True)
    model.load_state_dict(data["model_state_dict"], strict=True)
    return model


class Evaluator:
    """Thorough evaluation of the model, its predictions, and the losses.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    datasets : dict[str, PhysicsDataset]
        Dictionary of datasets to evaluate on
    eval_dir : Path
        Directory to save evaluation results
    batch_size : int, optional
        Batch size for evaluation, by default 256
    num_workers : int, optional
        Number of workers for dataloader, by default 4
    global_rank : int, optional
        Global rank for distributed training, by default 0
    local_rank : int, optional
        Local rank for distributed training, by default 0
    world_size : int, optional
        World size for distributed training, by default 1
    """

    def __init__(
        self,
        model: torch.nn.Module,
        datasets: dict[str, PhysicsDataset],
        eval_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
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
            log_file=None,
            log_level=log_level,
        )

        self.model = model
        self.model.eval()
        self.model.to(self.device)

        self.datasets = datasets
        self.eval_dir = eval_dir
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @classmethod
    def from_checkpoint(
        cls,
        base_path: Path,
        batch_size: int = 256,
        num_workers: int = 4,
        checkpoint_name: str = "best_model",
        global_rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> "Evaluator":
        """Create an Evaluator instance from a checkpoint.

        Parameters
        ----------
        base_path : Path
            Path to the base directory of the model
        batch_size : int, optional
            Batch size for evaluation, by default 256
        num_workers : int, optional
            Number of workers for dataloader, by default 4
        checkpoint_name : str, optional
            Name of the checkpoint to load, by default "best_model"
        global_rank : int, optional
            Global rank for distributed training, by default 0
        local_rank : int, optional
            Local rank for distributed training, by default 0
        world_size : int, optional
            World size for distributed training, by default 1

        Returns
        -------
        Evaluator
            Initialized Evaluator instance
        """
        config = load_config(base_path)
        device = (
            torch.device(f"cuda:{local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = load_model(base_path, device, config["model"], checkpoint_name)
        model.eval()
        datasets = get_dt_datasets(config["data"], split="test")

        # print the model architecture
        torch.set_float32_matmul_precision("high")
        if not platform.system() == "Windows":
            model = torch.compile(model, mode="max-autotune")

        if dist.is_initialized():
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=device,
            )

        return cls(
            model=model,
            datasets=datasets,
            eval_dir=base_path / "eval",
            batch_size=batch_size,
            num_workers=num_workers,
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
        )

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

    def _log_msg(self, msg: str):
        """Log a message."""
        if self.global_rank == 0:
            self.logger.info(msg)

    @torch.inference_mode()
    def eval_on_dataset(self, dataset: PhysicsDataset) -> torch.Tensor:
        criterion = torch.nn.MSELoss(reduction="none")
        loader = self._get_dataloader(dataset, is_distributed=self.ddp_enabled)

        losses = []
        for i, (x, target) in enumerate(loader):
            self._log_msg(f"Processing batch {i}")

            x = x.to(self.device)
            target = target.to(self.device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = self.model(x)
                loss = criterion(y, target).squeeze(1)  # remove T dimension
            # get the loss of each sample, dont average accross batches only h,w,c
            loss = torch.mean(loss, dim=(1, 2, 3))

            # gather losses from all GPUs
            if self.ddp_enabled:
                gathered_losses = [
                    torch.zeros_like(loss) for _ in range(self.world_size)
                ]
                dist.gather(loss, gathered_losses, dst=0)
                if self.global_rank == 0:
                    loss = torch.cat(gathered_losses, dim=0)

            if self.global_rank == 0:
                # Concatenate all gathered losses for this batch
                # Convert to DataFrame and append
                losses.append(loss.cpu())

        return torch.cat(losses, dim=0)

    def main(self):
        all_losses = {}
        for name, dataset in self.datasets.items():
            self._log_msg(f"Evaluating on dataset {name}")
            losses = self.eval_on_dataset(dataset)
            all_losses[name] = losses

        # insert losses into df, each col is a dataset, set the name of the col to the dataset name
        df = pd.DataFrame(all_losses)
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
