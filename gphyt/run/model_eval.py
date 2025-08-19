"""
Detailed evaluation of the model, its predictions, and the losses.
By: Florian Wiesner
Date: 2025-05-01
"""

from pathlib import Path
import platform
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

from gphyt.model.transformer.model import get_model
from gphyt.model.transformer.loss_fns import RVMSELoss
from gphyt.data.dataset_utils import get_dt_datasets
from gphyt.data.phys_dataset import PhysicsDataset
from gphyt.utils.logger import get_logger
from gphyt.run.run_utils import load_stored_model, find_checkpoint
from gphyt.utils.train_vis import visualize_predictions


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


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
        criteria: dict[str, torch.nn.Module] = None,
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
        
        # Initialize evaluation criteria
        if criteria is None:
            self.criteria = {
                "MSE": torch.nn.MSELoss(reduction="none"),
                "RVMSE": RVMSELoss(dims=(1, 2, 3), return_scalar=False)
            }
        else:
            self.criteria = criteria

    @classmethod
    def from_checkpoint(
        cls,
        base_path: Path,
        data_config: dict,
        model_config: dict,
        batch_size: int = 64,
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
        data_config : dict
            Data configuration dictionary
        model_config : dict
            Model configuration dictionary
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
        device = (
            torch.device(f"cuda:{local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model, checkpoint_info = cls._load_checkpoint(
            base_path, device, model_config, checkpoint_name
        )
        model.eval()
        datasets = get_dt_datasets(data_config, split="test")

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

        eval_dir = base_path / "eval" / checkpoint_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        # save the checkpoint info
        with open(eval_dir / "checkpoint_info.json", "w") as f:
            json.dump(checkpoint_info, f)

        return cls(
            model=model,
            datasets=datasets,
            eval_dir=eval_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
        )

    @staticmethod
    def _load_checkpoint(
        path: Path,
        device: torch.device,
        model_config: dict,
        checkpoint_name: str,
    ) -> tuple[torch.nn.Module, dict]:
        """Load a model from a checkpoint.

        Parameters
        ----------
        path : Path
            Path to the checkpoint
        device : torch.device
            Device to load the model to
        model_config : dict
            Model configuration dictionary

        Returns
        -------
        tuple[torch.nn.Module, dict]
            Loaded model and checkpoint information
        """
        subdir_name = "val_"
        checkpoint_path = find_checkpoint(
            path, subdir_name=subdir_name, specific_checkpoint=checkpoint_name
        )
        model = get_model(model_config)
        model.to(device)
        data = load_stored_model(checkpoint_path, device, ddp=False)
        model.load_state_dict(data["model_state_dict"], strict=True)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint_info = {
            "samples_trained": checkpoint["samples_trained"],
            "batches_trained": checkpoint["batches_trained"],
            "cycle_idx": checkpoint["cycle_idx"],
        }

        return model, checkpoint_info

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

    def _high_loss_idx(self, losses: torch.Tensor):
        """Get the indices of the losses that are too high."""
        high_losses = losses > 10
        return high_losses

    @torch.inference_mode()
    def _eval_on_dataset(self, dataset: PhysicsDataset) -> dict[str, torch.Tensor]:
        loader = self._get_dataloader(dataset, is_distributed=self.ddp_enabled)

        all_losses = {name: [] for name in self.criteria.keys()}
        
        for i, (x, target) in enumerate(loader):
            self._log_msg(f"  Batch {i}/{len(loader)}")

            x = x.to(self.device)
            target = target.to(self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                y = self.model(x)
                
            # Compute losses for each criterion
            batch_losses = {}
            for name, criterion in self.criteria.items():
                if name == "MSE":
                    loss = criterion(y, target).squeeze(1)  # remove T dimension
                    loss = torch.mean(loss, dim=(1, 2, 3))  # B
                elif name == "RVMSE":
                    # RVMSE expects (B, T, H, W, C) and returns (B, T, H, W, C) with dims reduced
                    loss = criterion(y, target)  # (B, 1, 1, 1, C) after dimension reduction
                    loss = loss.squeeze()  # Remove singleton dimensions
                    if loss.dim() > 1:
                        loss = torch.mean(loss, dim=-1)  # Average over channels -> (B,)
                else:
                    # For other custom criteria
                    loss = criterion(y, target)
                    if loss.dim() > 1:
                        loss = torch.mean(loss, dim=tuple(range(1, loss.dim())))
                
                batch_losses[name] = loss

            # Gather losses from all GPUs if distributed
            if self.ddp_enabled:
                for name, loss in batch_losses.items():
                    gathered_losses = [
                        torch.zeros_like(loss) for _ in range(self.world_size)
                    ]
                    dist.gather(loss, gathered_losses, dst=0)
                    if self.global_rank == 0:
                        batch_losses[name] = torch.cat(gathered_losses, dim=0)

            if self.global_rank == 0:
                for name, loss in batch_losses.items():
                    all_losses[name].append(loss.cpu())

        # Concatenate all losses
        result = {}
        for name, losses in all_losses.items():
            if losses:  # Check if losses is not empty
                result[name] = torch.cat(losses, dim=0)
            else:
                result[name] = torch.tensor([])
        
        return result

    def eval_all(self, datasets: dict[str, PhysicsDataset]) -> dict[str, pd.DataFrame]:
        all_criterion_losses = {criterion_name: {} for criterion_name in self.criteria.keys()}
        max_timesteps = 0
        
        for name, dataset in datasets.items():
            criterion_names = ", ".join(self.criteria.keys())
            self._log_msg(f"Evaluating on dataset {name} with {criterion_names}")
            losses_dict = self._eval_on_dataset(dataset)
            
            for criterion_name, losses in losses_dict.items():
                max_timesteps = max(max_timesteps, losses.shape[0])
                all_criterion_losses[criterion_name][name] = losses.cpu().numpy()

        # Pad all losses to max timesteps and create DataFrames
        result_dfs = {}
        for criterion_name, dataset_losses in all_criterion_losses.items():
            padded_losses = {}
            for dataset_name, losses in dataset_losses.items():
                padded_losses[dataset_name] = np.pad(
                    losses,
                    (0, max_timesteps - losses.shape[0]),
                    mode="constant",
                    constant_values=np.nan,
                )
            result_dfs[criterion_name] = pd.DataFrame(padded_losses)
        
        return result_dfs

    @torch.inference_mode()
    def _rollout(
        self,
        dataset: PhysicsDataset,
        traj_idx: int = 0,
        num_timesteps: int = -1,
        rollout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout the model on a trajectory.

        Parameters
        ----------
        dataset : PhysicsDataset
            The dataset to evaluate on
        traj_idx : int, optional
            The index of the trajectory to evaluate on, by default 0
        num_timesteps : int, optional
            The number of timesteps to rollout, by default 50
            if -1, rollout until the end of the trajectory
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
        if num_timesteps == -1:
            num_timesteps = T
        else:
            num_timesteps = min(
                num_timesteps, T
            )  # Ensure we don't exceed trajectory length

        outputs = []
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            for i in range(num_timesteps):
                # Predict next timestep
                output = self.model(input)  # (B, 1T, H, W, C)
                # if the output is nan, stop the rollout
                if torch.isnan(output).any() or torch.isinf(output).any():
                    break

                outputs.append(output.clone())
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
        self,
        datasets: dict[str, PhysicsDataset],
        num_samples: int = 10,
        num_timesteps: int = -1,
        rollout: bool = False,
    ) -> pd.DataFrame:
        all_losses = {}
        max_timesteps = 0
        for name, dataset in datasets.items():
            # copy the dataset with max rollout steps and full trajectory mode
            dataset = dataset.copy(
                overwrites={
                    "max_rollout_steps": num_timesteps,
                    "full_trajectory_mode": True,
                }
            )
            self._log_msg(f"Rolling out on dataset {name}. full rollout: {rollout}")

            # random trajectory indices
            indices = np.arange(len(dataset))
            if num_samples > len(indices):
                num_samples = len(indices)
            traj_idxs = np.random.choice(indices, size=num_samples, replace=False)

            traj_losses = []
            for i, traj_idx in enumerate(traj_idxs):
                self._log_msg(f"  Trajectory {i}/{num_samples}")
                _, _, loss = self._rollout(
                    dataset, traj_idx, num_timesteps, rollout
                )  # loss is (T, C)
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

    def visualize_rollout(
        self,
        dataset: PhysicsDataset,
        num_timesteps: int,
        save_path: Path,
        traj_idx: int = 0,
        rollout: bool = False,
    ) -> None:
        """Visualize the model predictions for a trajectory.

        Parameters
        ----------
        dataset : PhysicsDataset
            The dataset to evaluate on
        num_timesteps : int
            The number of timesteps to rollout
        save_path : Path | None, optional
            Path to save the visualizations, by default None
        traj_idx : int, optional
            The index of the trajectory to evaluate on, by default 0
        rollout : bool, optional
            Whether to rollout the full trajectory, by default False
        """
        # copy the dataset with max rollout steps and full trajectory mode
        dataset = dataset.copy(
            overwrites={
                "max_rollout_steps": num_timesteps,
                "full_trajectory_mode": True,
            }
        )

        # Get predictions and ground truth
        predictions, ground_truth, loss = self._rollout(
            dataset, traj_idx, num_timesteps, rollout
        )

        # Convert to numpy and transpose to match visualization format
        predictions = predictions.cpu().numpy()
        ground_truth = ground_truth.cpu().numpy()

        # Transpose to match visualization format (T, H, W, C) -> (T, W, H, C)
        predictions = predictions.transpose(0, 2, 1, 3)
        ground_truth = ground_truth.transpose(0, 2, 1, 3)

        # Calculate velocity magnitude
        vel_mag_pred = np.linalg.norm(predictions[..., -2:], axis=-1)
        vel_mag_gt = np.linalg.norm(ground_truth[..., -2:], axis=-1)

        # Add velocity magnitude as a new channel
        predictions = np.concatenate([predictions, vel_mag_pred[..., None]], axis=-1)
        ground_truth = np.concatenate([ground_truth, vel_mag_gt[..., None]], axis=-1)

        # Field names and colormaps
        field_names = [
            ("pressure", "inferno"),
            ("density", "viridis"),
            ("temperature", "magma"),
            ("velocity_x", "viridis"),
            ("velocity_y", "viridis"),
            ("velocity_mag", "viridis"),
        ]

        # Create save directory if needed
        save_path.mkdir(parents=True, exist_ok=True)

        # Visualize each field
        for i, (field, colormap) in enumerate(field_names):
            # Get min and max values for consistent color scaling
            vmin = min(np.nanmin(predictions[..., i]), np.nanmin(ground_truth[..., i]))
            vmax = max(np.nanmax(predictions[..., i]), np.nanmax(ground_truth[..., i]))

            for t in range(predictions.shape[0]):
                # Normalize the data to 0-1 range for colormap
                pred_norm = (predictions[t, ..., i] - vmin) / (vmax - vmin)
                gt_norm = (ground_truth[t, ..., i] - vmin) / (vmax - vmin)

                # Apply viridis colormap
                colormap = plt.get_cmap(colormap)
                pred_rgb = colormap(pred_norm)[..., :3]  # Get RGB channels
                gt_rgb = colormap(gt_norm)[..., :3]  # Get RGB channels

                # Convert to uint8 for PIL
                pred_rgb = (pred_rgb * 255).astype(np.uint8)
                gt_rgb = (gt_rgb * 255).astype(np.uint8)

                # Create PIL images
                pred_img = Image.fromarray(pred_rgb)
                gt_img = Image.fromarray(gt_rgb)

                # Save prediction
                pred_path = save_path / f"{field}_pred_t{t}.png"
                pred_img.save(pred_path)

                # Save ground truth
                gt_path = save_path / f"{field}_gt_t{t}.png"
                gt_img.save(gt_path)

    def main(self, overwrite: bool = False, subdir_name: str | None = None):
        if subdir_name is not None:
            self.eval_dir = self.eval_dir / subdir_name
            self.eval_dir.mkdir(parents=True, exist_ok=True)

        criterion_files = [f"{name.lower()}_losses.csv" for name in self.criteria.keys()]
        files_exist = all((self.eval_dir / f).exists() for f in criterion_files)
        
        if not overwrite and files_exist:
            criterion_names = ", ".join(self.criteria.keys())
            self.logger.info(f"{criterion_names} losses already evaluated, skipping...")
        else:
            # Evaluate on all datasets with all criteria
            criterion_dfs = self.eval_all(self.datasets)
            for criterion_name, df in criterion_dfs.items():
                filename = f"{criterion_name.lower()}_losses.csv"
                df.to_csv(self.eval_dir / filename, index=False)

        if not overwrite and (self.eval_dir / "single_step_losses.csv").exists():
            self.logger.info("Single step losses already evaluated, skipping...")
        else:
            # Rollout on all datasets
            df = self.rollout_all(
                self.datasets, num_samples=10, num_timesteps=50, rollout=False
            )
            df.to_csv(self.eval_dir / "single_step_losses.csv", index=False)

        if not overwrite and (self.eval_dir / "rollout_losses.csv").exists():
            self.logger.info("Rollout losses already evaluated, skipping...")
        else:
            df = self.rollout_all(
                self.datasets, num_samples=10, num_timesteps=50, rollout=True
            )
            df.to_csv(self.eval_dir / "rollout_losses.csv", index=False)

        try:
            # Visualize rollout on all datasets
            for name, dataset in self.datasets.items():
                self.visualize_rollout(
                    dataset,
                    num_timesteps=50,
                    save_path=self.eval_dir / "images_rollout" / name,
                    rollout=True,
                )
        except Exception as e:
            self.logger.error(f"Error visualizing rollout: {e}")

        try:
            # Visualize rollout on all datasets
            for name, dataset in self.datasets.items():
                self.visualize_rollout(
                    dataset,
                    num_timesteps=50,
                    save_path=self.eval_dir / "images_single_step" / name,
                    rollout=False,
                )
        except Exception as e:
            self.logger.error(f"Error visualizing single step: {e}")


def setup_ddp():
    """Initialize distributed data parallel training."""
    dist.init_process_group()


def main(
    config_path: Path,
    log_dir: Path | None,
    checkpoint_name: str,
    sim_name: str | None,
    data_dir: Path | None,
    subdir_name: str | None,
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
    subdir_name : str | None
        Name of the subdirectory where the evaluation is stored
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

    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]

    data_config["datasets"] = data_config["datasets"]  # + eval_ds

    evaluator = Evaluator.from_checkpoint(
        base_path=log_dir / sim_name,
        data_config=data_config,
        model_config=model_config,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        checkpoint_name=checkpoint_name,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    evaluator.main(subdir_name=subdir_name)


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
    parser.add_argument("--subdir_name", type=str, default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))

    config_path = args.config_file
    log_dir = args.log_dir
    sim_name = args.sim_name
    data_dir = args.data_dir
    checkpoint_name = args.checkpoint_name
    subdir_name = args.subdir_name

    main(
        config_path=config_path,
        log_dir=log_dir,
        sim_name=sim_name,
        data_dir=data_dir,
        checkpoint_name=checkpoint_name,
        subdir_name=subdir_name,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
