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

from gphyt.model.transformer.model import get_model as get_model_gphyt
from gphyt.model.unet import get_model as get_model_unet
from gphyt.model.model_specs import UNet_M
from gphyt.model.transformer.loss_fns import RVMSELoss, RNMSELoss
from gphyt.data.dataset_utils import get_dt_datasets
from gphyt.data.phys_dataset import PhysicsDataset
from gphyt.utils.logger import get_logger
from gphyt.run.run_utils import load_stored_model, find_checkpoint


# Which fields are actually used in which dataset.
# 0=pressure, 1=density, 2=temp, 3=velx, 4=vely
DATASET_FIELDS = {
    "cylinder_sym_flow_water": (0, 3, 4),
    "cylinder_pipe_flow_water": (0, 3, 4),
    "object_periodic_flow_water": (0, 3, 4),
    "object_sym_flow_water": (0, 3, 4),
    "object_sym_flow_air": (0, 3, 4),
    "heated_object_pipe_flow_air": (0, 1, 2, 3, 4),
    "cooled_object_pipe_flow_air": (0, 1, 2, 3, 4),
    "rayleigh_benard_obstacle": (0, 1, 2, 3, 4),
    "twophase_flow": (0, 1, 3, 4),
    "rayleigh_benard": (0, 1, 3, 4),
    "shear_flow": (0, 3, 4),
    "euler_multi_quadrants_periodicbc": (0, 1, 3, 4),
    "acoustic_scattering_inclusions": (0, 3, 4),
    "turbulent_radiative_layer_2d": (0, 1, 3, 4),
    "supersonic_flow": (0, 1, 3, 4),
    "open_obj_water": (0, 3, 4),
    "euler_multi_quadrants_openbc": (0, 1, 3, 4),
}


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
    """

    def __init__(
        self,
        model: torch.nn.Module,
        datasets: dict[str, PhysicsDataset],
        eval_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        log_level: str = "INFO",
    ):
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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

        self.eval_criteria = {
            "RNMSE": RNMSELoss(dims=(1, 2, 3), return_scalar=False),
            "RVMSE": RVMSELoss(dims=(1, 2, 3), return_scalar=False),
        }
        self.rollout_criteria = {
            "RNMSE": RNMSELoss(dims=(2, 3), return_scalar=False),
            "RVMSE": RVMSELoss(dims=(2, 3), return_scalar=False),
        }

    @classmethod
    def from_checkpoint(
        cls,
        base_path: Path,
        data_config: dict,
        model_config: dict,
        batch_size: int = 64,
        num_workers: int = 4,
        checkpoint_name: str = "best_model",
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
            torch.device(f"cuda:{0}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if model_config == "unet-M":
            model = get_model_unet(UNet_M())
        else:
            model = get_model_gphyt(model_config)

        model.to(device)
        torch.set_float32_matmul_precision("high")
        if not platform.system() == "Windows":
            model = torch.compile(model, mode="max-autotune")

        model, checkpoint_info = cls._load_checkpoint(
            base_path, device, model, checkpoint_name
        )
        model.eval()
        datasets = get_dt_datasets(data_config, split="test")

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
        )

    @staticmethod
    def _load_checkpoint(
        path: Path,
        device: torch.device,
        model: torch.nn.Module,
        checkpoint_name: str,
    ) -> tuple[torch.nn.Module, dict]:
        """Load a model from a checkpoint.

        Parameters
        ----------
        path : Path
            Path to the checkpoint
        device : torch.device
            Device to load the model to
        model : torch.nn.Module
            Model to load the checkpoint into

        Returns
        -------
        tuple[torch.nn.Module, dict]
            Loaded model and checkpoint information
        """
        subdir_name = "val_"
        checkpoint_path = find_checkpoint(
            path, subdir_name=subdir_name, specific_checkpoint=checkpoint_name
        )
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
        self.logger.info(msg)

    def _high_loss_idx(self, losses: torch.Tensor):
        """Get the indices of the losses that are too high."""
        high_losses = losses > 10
        return high_losses

    @torch.inference_mode()
    def _eval_on_dataset(self, dataset: PhysicsDataset) -> dict[str, torch.Tensor]:
        loader = self._get_dataloader(dataset, is_distributed=self.ddp_enabled)

        all_losses = {name: [] for name in self.eval_criteria.keys()}

        for i, (xx, target) in enumerate(loader):
            self._log_msg(f"  Batch {i}/{len(loader)}")

            xx = xx.to(self.device)
            target = target.to(self.device)

            # Perform autoregressive prediction
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                ar_steps = target.shape[1]  # num of timesteps
                outputs = []
                output = torch.tensor(0.0, device=self.device)  # Initialize for linter
                for _ar_step in range(ar_steps):
                    if _ar_step == 0:
                        x = xx
                    else:
                        x = torch.cat(
                            (x[:, 1:, ...], output),
                            dim=1,
                        )  # remove first input step, append output step
                    output = self.model(x)
                    outputs.append(output)

                # Use the final step for evaluation
                final_output = outputs[-1]  # (B, 1, H, W, C)
                final_target = target[:, -1:, ...]  # (B, 1, H, W, C)

            # Compute losses for each criterion using final step
            batch_losses = {}
            ds_name = dataset.dataset_name.lower()
            fields = DATASET_FIELDS.get(ds_name)
            if fields is None:
                raise ValueError(
                    f"Dataset '{ds_name}' not found in DATASET_FIELDS mapping"
                )
            y_loss = final_output[..., fields]
            target_loss = final_target[..., fields]
            for name, criterion in self.eval_criteria.items():
                # RVMSE expects (B, T, H, W, C) and returns (B, T, H, W, C) with dims reduced
                loss = criterion(
                    y_loss, target_loss
                )  # (B, 1, 1, 1, C) after dimension reduction
                # Only squeeze the singleton spatial and temporal dimensions, keep batch and channel dims
                loss = loss.squeeze(1).squeeze(1).squeeze(1)  # -> (B, C)
                loss = torch.mean(loss, dim=-1)  # Average over channels -> (B,)
                batch_losses[name] = loss

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
        all_criterion_losses = {
            criterion_name: {} for criterion_name in self.eval_criteria.keys()
        }
        max_timesteps = 0

        for name, dataset in datasets.items():
            criterion_names = ", ".join(self.eval_criteria.keys())
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
            Tuple containing the predicted outputs,
            the ground truth, and the losses dict for each criterion at each timestep
        """
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

        # Compute losses for each criterion similar to _eval_on_dataset
        losses_dict = {}
        ds_name = dataset.dataset_name.lower()
        fields = DATASET_FIELDS.get(ds_name)
        if fields is None:
            raise ValueError(f"Dataset '{ds_name}' not found in DATASET_FIELDS mapping")

        # Select only the relevant fields for loss computation
        outputs_loss = outputs[..., fields]  # (T, H, W, C_subset)
        full_traj_loss = full_traj[..., fields]  # (T, H, W, C_subset)

        for name, criterion in self.rollout_criteria.items():
            if name == "MSE":
                loss = criterion(outputs_loss, full_traj_loss)  # (T, H, W, C_subset)
                loss = torch.mean(loss, dim=(1, 2))  # (T, C_subset)
            elif name == "RVMSE":
                # Add batch dimension for RVMSE calculation
                outputs_batch = outputs_loss.unsqueeze(0)  # (1, T, H, W, C_subset)
                full_traj_batch = full_traj_loss.unsqueeze(0)  # (1, T, H, W, C_subset)

                # RVMSE with dims=(2, 3) reduces over spatial dims only, keeping (B, T, C)
                loss = criterion(
                    outputs_batch, full_traj_batch
                )  # (1, T, 1, 1, C_subset)
                loss = loss.squeeze(0).squeeze(-2).squeeze(-2)  # -> (T, C_subset)
            else:
                # For other custom criteria
                loss = criterion(outputs_loss, full_traj_loss)
                if loss.dim() > 1:
                    loss = torch.mean(
                        loss, dim=tuple(range(1, loss.dim() - 1))
                    )  # Keep (T, C)

            losses_dict[name] = loss

        # Return predictions and ground truth (excluding first timestep)
        return outputs, full_traj, losses_dict

    def rollout_all(
        self,
        datasets: dict[str, PhysicsDataset],
        num_samples: int = 10,
        num_timesteps: int = -1,
        rollout: bool = False,
    ) -> dict[str, pd.DataFrame]:
        all_criterion_losses = {
            criterion_name: {} for criterion_name in self.rollout_criteria.keys()
        }
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

            # Store losses for each criterion separately
            criterion_traj_losses = {name: [] for name in self.rollout_criteria.keys()}

            for i, traj_idx in enumerate(traj_idxs):
                self._log_msg(f"  Trajectory {i}/{num_samples}")
                _, _, losses_dict = self._rollout(
                    dataset, traj_idx, num_timesteps, rollout
                )  # losses_dict contains losses for each criterion

                for criterion_name, loss in losses_dict.items():
                    max_timesteps = max(max_timesteps, loss.shape[0])
                    criterion_traj_losses[criterion_name].append(loss)

            # Process each criterion separately
            for criterion_name, traj_losses in criterion_traj_losses.items():
                # (samples, T, C)
                traj_losses = torch.stack(traj_losses, dim=0)
                traj_losses = traj_losses.cpu().numpy()
                # mean, std and median over the samples (T, C)
                mean_loss = np.nanmean(traj_losses, axis=0)
                std_loss = np.nanstd(traj_losses, axis=0)
                median_loss = np.nanmedian(traj_losses, axis=0)

                all_criterion_losses[criterion_name][name] = {
                    "mean": mean_loss,
                    "std": std_loss,
                    "median": median_loss,
                }

        # pad all losses to max timesteps for each criterion
        for criterion_name, dataset_losses in all_criterion_losses.items():
            for dataset_name, metrics in dataset_losses.items():
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

        # Create DataFrames for each criterion
        result_dfs = {}
        for criterion_name, dataset_losses in all_criterion_losses.items():
            # Create multi-level index DataFrame for this criterion
            index_tuples = []
            data = []

            for dataset_name, metrics in dataset_losses.items():
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
            result_dfs[criterion_name] = df

        return result_dfs

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
        predictions, ground_truth, losses_dict = self._rollout(
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

    def main(
        self,
        overwrite: bool = False,
        subdir_name: str | None = None,
        forecast_horizons: list[int] | None = None,
    ):
        if subdir_name is not None:
            self.eval_dir = self.eval_dir / subdir_name
            self.eval_dir.mkdir(parents=True, exist_ok=True)

        # If no forecast horizons specified, use default behavior
        if forecast_horizons is None:
            criterion_files = [
                f"{name.lower()}_losses.csv" for name in self.eval_criteria.keys()
            ]
            files_exist = all((self.eval_dir / f).exists() for f in criterion_files)

            if not overwrite and files_exist:
                criterion_names = ", ".join(self.eval_criteria.keys())
                self.logger.info(
                    f"{criterion_names} losses already evaluated, skipping..."
                )
            else:
                # Evaluate on all datasets with all criteria
                criterion_dfs = self.eval_all(self.datasets)
                for criterion_name, df in criterion_dfs.items():
                    filename = f"{criterion_name.lower()}_losses.csv"
                    df.to_csv(self.eval_dir / filename, index=False)
        else:
            # Evaluate for each forecast horizon
            for horizon in forecast_horizons:
                self._log_msg(f"Evaluating for forecast horizon: {horizon} steps")

                # Create datasets with the specified n_steps_output
                horizon_datasets = {}
                for name, dataset in self.datasets.items():
                    horizon_datasets[name] = dataset.copy(
                        overwrites={"n_steps_output": horizon}
                    )

                # Check if files exist for this horizon
                criterion_files = [
                    f"{name.lower()}_losses_h{horizon}.csv"
                    for name in self.eval_criteria.keys()
                ]
                files_exist = all((self.eval_dir / f).exists() for f in criterion_files)

                if not overwrite and files_exist:
                    criterion_names = ", ".join(self.eval_criteria.keys())
                    self.logger.info(
                        f"Horizon {horizon}: {criterion_names} losses already evaluated, skipping..."
                    )
                else:
                    # Evaluate on datasets with this horizon
                    criterion_dfs = self.eval_all(horizon_datasets)
                    for criterion_name, df in criterion_dfs.items():
                        filename = f"{criterion_name.lower()}_losses_h{horizon}.csv"
                        df.to_csv(self.eval_dir / filename, index=False)
                        self._log_msg(f"Saved {filename}")

        # # Check if single step rollout files exist for all criteria
        # single_step_files = [
        #     f"single_step_{name.lower()}_losses.csv"
        #     for name in self.eval_criteria.keys()
        # ]
        # single_step_files_exist = all(
        #     (self.eval_dir / f).exists() for f in single_step_files
        # )

        # if not overwrite and single_step_files_exist:
        #     criterion_names = ", ".join(self.eval_criteria.keys())
        #     self.logger.info(
        #         f"Single step {criterion_names} losses already evaluated, skipping..."
        #     )
        # else:
        #     # Rollout on all datasets
        #     criterion_dfs = self.rollout_all(
        #         self.datasets, num_samples=10, num_timesteps=50, rollout=False
        #     )
        #     for criterion_name, df in criterion_dfs.items():
        #         filename = f"single_step_{criterion_name.lower()}_losses.csv"
        #         df.to_csv(self.eval_dir / filename, index=False)

        # # Check if rollout files exist for all criteria
        # rollout_files = [
        #     f"rollout_{name.lower()}_losses.csv" for name in self.eval_criteria.keys()
        # ]
        # rollout_files_exist = all((self.eval_dir / f).exists() for f in rollout_files)

        # if not overwrite and rollout_files_exist:
        #     criterion_names = ", ".join(self.eval_criteria.keys())
        #     self.logger.info(
        #         f"Rollout {criterion_names} losses already evaluated, skipping..."
        #     )
        # else:
        #     criterion_dfs = self.rollout_all(
        #         self.datasets, num_samples=10, num_timesteps=50, rollout=True
        #     )
        #     for criterion_name, df in criterion_dfs.items():
        #         filename = f"rollout_{criterion_name.lower()}_losses.csv"
        #         df.to_csv(self.eval_dir / filename, index=False)

        # try:
        #     # Visualize rollout on all datasets
        #     for name, dataset in self.datasets.items():
        #         print(f"Visualizing rollout for dataset {name}")
        #         self.visualize_rollout(
        #             dataset,
        #             num_timesteps=50,
        #             save_path=self.eval_dir / "images_rollout" / name,
        #             rollout=True,
        #         )
        # except Exception as e:
        #     self.logger.error(f"Error visualizing rollout: {e}")

        # try:
        #     # Visualize rollout on all datasets
        #     for name, dataset in self.datasets.items():
        #         print(f"Visualizing single step for dataset {name}")
        #         self.visualize_rollout(
        #             dataset,
        #             num_timesteps=50,
        #             save_path=self.eval_dir / "images_single_step" / name,
        #             rollout=False,
        #         )
        # except Exception as e:
        #     self.logger.error(f"Error visualizing single step: {e}")


def main(
    config_path: Path,
    log_dir: Path | None,
    checkpoint_name: str,
    sim_name: str | None,
    data_dir: Path | None,
    subdir_name: str | None,
    forecast_horizons: list[int] | None = None,
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
    forecast_horizons : list[int] | None
        List of forecast horizons (n_steps_output) to evaluate.
        E.g., [4, 8, 12] will evaluate with 4, 8, and 12 output steps.
        If None, uses default behavior with config's n_steps_output.
    """
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
    )
    evaluator.main(subdir_name=subdir_name, forecast_horizons=forecast_horizons)


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
    parser.add_argument(
        "--forecast_horizons",
        type=int,
        nargs="+",
        default=None,
        help="List of forecast horizons to evaluate (e.g., 4 8 12 16)",
    )
    args = parser.parse_args()

    config_path = args.config_file
    log_dir = args.log_dir
    sim_name = args.sim_name
    data_dir = args.data_dir
    checkpoint_name = args.checkpoint_name
    subdir_name = args.subdir_name
    forecast_horizons = args.forecast_horizons

    main(
        config_path=config_path,
        log_dir=log_dir,
        sim_name=sim_name,
        data_dir=data_dir,
        checkpoint_name=checkpoint_name,
        subdir_name=subdir_name,
        forecast_horizons=forecast_horizons,
    )
