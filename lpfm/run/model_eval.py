"""
Detailed evaluation of the model, its predictions, and the losses.
By: Florian Wiesner
Date: 2025-05-01
"""

from pathlib import Path
import platform
import argparse
import os
import numpy as np

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import pandas as pd

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
    def _eval_on_dataset(self, dataset: PhysicsDataset) -> torch.Tensor:
        criterion = torch.nn.MSELoss(reduction="none")
        loader = self._get_dataloader(dataset, is_distributed=self.ddp_enabled)

        losses = []
        for i, (x, target) in enumerate(loader):
            self._log_msg(f"  Batch {i}/{len(loader)}")

            x = x.to(self.device)
            target = target.to(self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
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

    def eval_all(self, datasets: dict[str, PhysicsDataset]) -> pd.DataFrame:
        all_losses = {}
        max_timesteps = 0
        for name, dataset in datasets.items():
            self._log_msg(f"Evaluating on dataset {name}")
            losses = self._eval_on_dataset(dataset)
            max_timesteps = max(max_timesteps, losses.shape[0])
            all_losses[name] = losses.cpu().numpy()

        # pad all losses to max timesteps
        for name, losses in all_losses.items():
            losses = np.pad(
                losses,
                (0, max_timesteps - losses.shape[0]),
                mode="constant",
                constant_values=np.nan,
            )
            all_losses[name] = losses
        df = pd.DataFrame(all_losses)
        return df

    @torch.inference_mode()
    def _rollout(
        self, dataset: PhysicsDataset, traj_idx: int = 0, rollout: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout the model on a trajectory.

        Parameters
        ----------
        dataset : PhysicsDataset
            The dataset to evaluate on
        traj_idx : int, optional
            The index of the trajectory to evaluate on, by default 0
        rollout : bool, optional
            Whether to rollout the full trajectory, by default False

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing the predicted outputs,
            the ground truth, and the loss at each timestep
        """
        criterion = torch.nn.MSELoss(reduction="none")

        # get first trajectory
        traj_idx = min(traj_idx, len(dataset) - 1)
        input, full_traj = dataset[traj_idx]

        input = input.to(self.device)
        full_traj = full_traj.to(self.device)

        # add batch dimension
        input = input.unsqueeze(0)
        full_traj = full_traj.unsqueeze(0)

        B, T, H, W, C = full_traj.shape

        outputs = []
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            for i in range(T):  # T-1 because we predict the next step
                # Predict next timestep
                output = self.model(input)  # (B, 1T, H, W, C)
                # if the output is nan, stop the rollout
                if torch.isnan(output).any() or torch.isinf(output).any():
                    break

                outputs.append(output)
                # Update input
                if rollout:
                    input = torch.cat([input[:, 1:, ...], output], dim=1)
                else:
                    input = torch.cat(
                        [input[:, 1:, ...], full_traj[:, i, ...].unsqueeze(1)], dim=1
                    )

        # remove batch dimension
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.squeeze(0)
        full_traj = full_traj.squeeze(0)

        # pad outputs to T timesteps to make sure all trajectories have the same length
        pad = torch.full(
            (
                T - outputs.shape[0],
                outputs.shape[1],
                outputs.shape[2],
                outputs.shape[3],
            ),
            float("nan"),
            device=outputs.device,
        )
        outputs = torch.cat([outputs, pad], dim=0)

        # loss is still a tensor of shape (T, H, W, C)
        loss = criterion(outputs, full_traj)
        # reduce over H and W
        loss = torch.mean(loss, dim=(1, 2))  # (T, C)

        # Return predictions and ground truth (excluding first timestep)
        return outputs, full_traj, loss

    def rollout_all(
        self, datasets: dict[str, PhysicsDataset], num_samples: int = 10
    ) -> pd.DataFrame:
        all_losses = {}
        max_timesteps = 0
        for name, dataset in datasets.items():
            # copy the dataset with max rollout steps and full trajectory mode
            dataset = dataset.copy(
                overwrites={
                    "max_rollout_steps": 1000,
                    "full_trajectory_mode": True,
                }
            )
            self._log_msg(f"Rolling out on dataset {name}")

            # random trajectory indices
            indices = np.arange(len(dataset))
            if num_samples > len(indices):
                num_samples = len(indices)
            traj_idxs = np.random.choice(indices, size=num_samples, replace=False)

            traj_losses = []
            for traj_idx in traj_idxs:
                self._log_msg(f"  Trajectory {traj_idx}/{num_samples}")
                _, _, loss = self._rollout(dataset, traj_idx)  # loss is (T, C)
                max_timesteps = max(max_timesteps, loss.shape[0])
                traj_losses.append(loss)

            # (samples, T, C)
            traj_losses = torch.stack(traj_losses, dim=0)
            traj_losses = traj_losses.cpu().numpy()
            # mean, std and median over the samples (T, C)
            mean_loss = np.nanmean(traj_losses, axis=0)
            std_loss = np.nanstd(traj_losses, axis=0)
            median_loss = np.nanmedian(traj_losses, axis=0)

            all_losses[name] = {
                "mean": mean_loss,
                "std": std_loss,
                "median": median_loss,
            }

        # pad all losses to max timesteps
        for _, metrics in all_losses.items():
            metrics["mean"] = np.pad(
                metrics["mean"],
                ((0, max_timesteps - metrics["mean"].shape[0]), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )
            metrics["std"] = np.pad(
                metrics["std"],
                ((0, max_timesteps - metrics["std"].shape[0]), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )
            metrics["median"] = np.pad(
                metrics["median"],
                ((0, max_timesteps - metrics["median"].shape[0]), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )

        # Create multi-level index DataFrame
        # First create a list of tuples for the multi-index
        index_tuples = []
        data = []

        for dataset_name, metrics in all_losses.items():
            for metric_name, array in metrics.items():
                for channel in range(array.shape[1]):
                    index_tuples.append((dataset_name, metric_name, channel))
                    data.append(array[:, channel])

        # Create the multi-index
        index = pd.MultiIndex.from_tuples(
            index_tuples, names=["dataset", "metric", "channel"]
        )

        # Create DataFrame with multi-index columns
        df = pd.DataFrame(data, index=index).T

        return df

    def main(self):
        # # Evaluate on all datasets
        # df = self.eval_all(self.datasets)
        # df.to_csv(self.eval_dir / "losses.csv", index=False)

        # Rollout on all datasets
        df = self.rollout_all(self.datasets)
        df.to_csv(self.eval_dir / "rollout_losses.csv", index=False)


def setup_ddp():
    """Initialize distributed data parallel training."""
    dist.init_process_group()


def main(
    config_path: Path,
    log_dir: Path | None,
    checkpoint_name: str,
    sim_name: str | None,
    data_dir: Path | None,
    global_rank: int,
    local_rank: int,
    world_size: int,
):
    """Main evaluation function.

    Parameters
    ----------
    config_path : Path
        Path to the config file
    log_dir : Path | None
        Path to the log directory
    checkpoint_name : str
        Name of the checkpoint to load
    sim_name : str | None
        Name of the simulation
    data_dir : Path | None
        Path to the data directory
    global_rank : int
        Global rank for distributed training
    local_rank : int
        Local rank for distributed training
    world_size : int
        World size for distributed training
    """
    # Set cuda device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Load config
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)

    ####################################################################
    ########### Augment config #########################################
    ####################################################################

    if log_dir is not None:
        log_dir = Path(log_dir)
        config["logging"]["log_dir"] = log_dir

    if data_dir is not None:
        data_dir = Path(data_dir)
        config["data"]["data_dir"] = data_dir

    if sim_name is not None:
        config["wandb"]["id"] = sim_name

    ####################################################################
    ########### Initialize evaluator ###################################
    ####################################################################
    if world_size > 1:
        setup_ddp()

    evaluator = Evaluator.from_checkpoint(
        base_path=log_dir / sim_name,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        checkpoint_name=checkpoint_name,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    evaluator.main()


if __name__ == "__main__":
    ############################################################
    ########### Parse arguments ################################
    ############################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--checkpoint_name", type=str)
    parser.add_argument("--sim_name", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))

    config_path = args.config_file
    log_dir = args.log_dir
    sim_name = args.sim_name
    data_dir = args.data_dir
    checkpoint_name = args.checkpoint_name

    main(
        config_path=config_path,
        log_dir=log_dir,
        sim_name=sim_name,
        data_dir=data_dir,
        checkpoint_name=checkpoint_name,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
