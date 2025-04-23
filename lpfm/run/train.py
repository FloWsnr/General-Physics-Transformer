"""
Training script for transformer models.

Author: Florian Wiesner
Date: 2025-04-07
"""

import os
from pathlib import Path
import time
import argparse
import platform

import wandb
import wandb.wandb_run
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from torch.amp.grad_scaler import GradScaler

import dadaptation
import prodigyopt

from lpfm.data.dataset_utils import get_dataloader
from lpfm.model.transformer.model import get_model
from lpfm.utils.train_vis import log_predictions_wandb, visualize_predictions
from lpfm.utils.logger import get_logger
from lpfm.model.transformer.loss_fns import NMSELoss, RNMSELoss, VMSELoss, RVMSELoss
from lpfm.run.run_utils import find_last_checkpoint


class Trainer:
    def __init__(
        self,
        config: Path | dict,
        global_rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
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

        if isinstance(config, dict):
            self.config = config
        else:
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)

        ################################################################
        ############ Initialize time limit #############################
        ################################################################
        self.avg_sec_per_epoch = 0
        self.start_time = time.time()
        if "time_limit" in self.config["training"]:
            self.time_limit = self.config["training"]["time_limit"]
        else:
            self.time_limit = None

        ################################################################
        ########### Initialize config #################################
        ################################################################
        img_size = self.config["data"]["out_shape"]
        time_steps = self.config["data"]["n_steps_input"]
        self.config["model"]["img_size"] = (time_steps, img_size[0], img_size[1])

        self.logger = get_logger(
            "Trainer",
            log_file=self.config["logging"]["log_file"],
            log_level=self.config["logging"]["log_level"],
        )
        ################################################################
        ########### Initialize logging #################################
        ################################################################
        self.log_dir = (
            Path(self.config["logging"]["log_dir"]) / self.config["wandb"]["id"]
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.global_rank == 0:
            self.wandb_run = login_wandb(self.config)
        else:
            self.wandb_run = None

        ################################################################
        ############# Log ddp info ####################################
        ################################################################
        self.log_msg(f"DDP enabled: {self.ddp_enabled}")
        self.log_msg(f"World size: {self.world_size}")
        self.log_msg(f"Local rank: {self.local_rank}")
        self.log_msg(f"Global rank: {self.global_rank}")

        ################################################################
        ########### Set random seeds ##################################
        ################################################################
        torch.manual_seed(self.config["training"]["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config["training"]["seed"])

        ################################################################
        ########### Initialize model ##################################
        ################################################################
        self.log_msg(f"Using device: {self.device}")

        self.model = get_model(model_config=self.config["model"])

        total_params = sum(p.numel() for p in self.model.parameters())
        self.log_msg(f"Model size: {total_params / 1e6:.2f}M parameters")
        # self.log_msg(f"Model architecture: {self.model}")

        if self.global_rank == 0:
            self.wandb_run.config.update(
                {"model/model_size [M]": total_params / 1e6}, allow_val_change=True
            )

        # print the model architecture
        self.model.to(self.device)
        torch.set_float32_matmul_precision("high")
        if self.config["training"]["compile"] and not platform.system() == "Windows":
            self.log_msg("Compiling model")
            self.model = torch.compile(self.model)
        if self.config["training"]["amp"] and torch.cuda.is_available():
            self.log_msg("Using AMP")
            self.use_amp = True
            self.grad_scaler = GradScaler()
        else:
            self.use_amp = False
            self.grad_scaler = None

        if self.ddp_enabled:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.device,
            )

        ################################################################
        ########### Initialize data loaders ##########################
        ################################################################
        self.train_loader = get_dataloader(
            self.config["data"],
            self.config["training"],
            split="train",
            is_distributed=self.ddp_enabled,
        )
        self.val_loader = get_dataloader(
            self.config["data"],
            self.config["training"],
            split="val",
            is_distributed=self.ddp_enabled,
        )
        ################################################################
        ########### Initialize training parameters ##################
        ################################################################
        self.total_epochs = self.config["training"]["epochs"]
        self.epoch = 1
        self.world_total_batches_trained = 0
        self.world_total_samples_trained = 0

        # NOTE: These are per GPU worker
        self.batch_size = self.config["training"]["batch_size"]
        self.train_batches_per_epoch = len(self.train_loader)
        self.val_batches_per_epoch = len(self.val_loader)
        self.train_samples_per_epoch = self.train_batches_per_epoch * self.batch_size

        # NOTE: This is across all GPU workers
        self.world_total_samples = (
            self.train_samples_per_epoch * self.total_epochs * self.world_size
        )
        self.world_total_batches = (
            self.train_batches_per_epoch * self.total_epochs * self.world_size
        )
        self.world_train_batches_per_epoch = (
            self.train_batches_per_epoch * self.world_size
        )
        self.world_val_batches_per_epoch = self.val_batches_per_epoch * self.world_size
        self.world_train_samples_per_epoch = (
            self.train_samples_per_epoch * self.world_size
        )

        self.log_msg(f"Training for {self.total_epochs} epochs")
        self.log_msg(
            f"Training on {self.world_train_batches_per_epoch} batches per epoch, {self.world_total_batches} batches in total"
        )
        self.log_msg(
            f"Training for {self.world_train_samples_per_epoch} samples per epoch, {self.world_total_samples} samples in total"
        )
        self.log_msg(
            f"Validating on {self.world_val_batches_per_epoch} batches per epoch"
        )

        if self.global_rank == 0:
            self.wandb_run.config.update(
                {
                    "training/total_samples": self.world_total_samples,
                    "training/samples_per_epoch": self.world_train_samples_per_epoch,
                    "training/batches_per_epoch": self.world_train_batches_per_epoch,
                },
                allow_val_change=True,
            )

        ################################################################
        ########### Initialize loss function and optimizer ###########
        ################################################################
        # all losses which should be computed and logged
        self.loss_fns = {
            "MAE": nn.L1Loss(),
            "MSE": nn.MSELoss(),
            "RMSE": nn.MSELoss(),
            "NMSE": NMSELoss(),
            "RNMSE": RNMSELoss(),
            "VMSE": VMSELoss(),
            "RVMSE": RVMSELoss(),
        }

        opt_config = self.config["training"]["optimizer"]
        if self.config["training"]["criterion"] == "MSE":
            self.criterion = self.loss_fns.pop("MSE")
        elif self.config["training"]["criterion"] == "RMSE":
            self.criterion = self.loss_fns.pop("RMSE")
        elif self.config["training"]["criterion"] == "NMSE":
            self.criterion = self.loss_fns.pop("NMSE")
        elif self.config["training"]["criterion"] == "RNMSE":
            self.criterion = self.loss_fns.pop("RNMSE")
        elif self.config["training"]["criterion"] == "MAE":
            self.criterion = self.loss_fns.pop("MAE")
        elif self.config["training"]["criterion"] == "VMSE":
            self.criterion = self.loss_fns.pop("VMSE")
        elif self.config["training"]["criterion"] == "RVMSE":
            self.criterion = self.loss_fns.pop("RVMSE")
        else:
            raise ValueError(
                f"Criterion {self.config['training']['criterion']} not supported"
            )
        self.optimizer = get_optimizer(self.model, opt_config)
        ################################################################
        ########### Initialize learning rate scheduler ################
        ################################################################
        if "lr_scheduler" in self.config["training"]:
            lrs_config = self.config["training"]["lr_scheduler"]
            self.scheduler = get_lr_scheduler(
                optimizer=self.optimizer,
                lrs_config=lrs_config,
                learning_rate=float(opt_config["learning_rate"]),
                total_epochs=self.total_epochs,
                train_batches_per_epoch=self.train_batches_per_epoch,
            )
        else:
            self.scheduler = None

        ################################################################
        ########### Watch model ########################################
        ################################################################
        if self.global_rank == 0:
            self.wandb_run.watch(
                self.model,
                criterion=self.criterion,
                log=self.config["wandb"]["log_model"],
                log_freq=100,
            )

    def log_msg(self, msg: str):
        """Log a message."""
        if self.global_rank == 0:
            self.logger.info(msg)

    def load_checkpoint(self, checkpoint_path: Path):
        """Restart training from a checkpoint."""
        self.log_msg(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"] + 1
        self.world_total_samples_trained = checkpoint["samples_trained"]
        self.world_total_batches_trained = checkpoint["batches_trained"]
        self.log_msg(
            f"Restarting training from epoch {self.epoch} with {self.world_total_samples_trained} samples trained"
        )

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "samples_trained": self.world_total_samples_trained,
            "batches_trained": self.world_total_batches_trained,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.grad_scaler is not None:
            checkpoint["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.epoch_dir / f"{name}.pth")
        self.log_msg(f"Summary: Checkpoint saved to {self.epoch_dir / f'{name}.pth'}")

    def save_config(self):
        """Save the config to the log directory."""
        if self.global_rank == 0:
            with open(self.log_dir / "config.yaml", "w") as f:
                yaml.dump(self.config, f)

    def _compute_log_metrics(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute the log metrics."""
        with torch.inference_mode():
            losses = {}
            for loss_name, loss_fn in self.loss_fns.items():
                loss = loss_fn(x, target)
                losses[loss_name] = loss.detach()
        return losses

    def _reduce_all_losses(
        self, losses: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Reduce the losses across all GPUs."""
        for loss_name, loss in losses.items():
            losses[loss_name] = self._reduce_loss(loss)
        return losses

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce the loss across all GPUs."""
        loss_tensor = loss.clone().detach()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor

    def _get_lr(self) -> float:
        """Get the learning rate."""
        match self.optimizer:
            case torch.optim.AdamW if self.scheduler is not None:
                return self.scheduler.get_last_lr()[0]
            case dadaptation.DAdaptAdam | prodigyopt.Prodigy if (
                self.scheduler is not None
            ):
                return (
                    self.optimizer.param_groups[0]["lr"]
                    * self.optimizer.param_groups[0]["d"]
                )
            case _:
                return self.optimizer.param_groups[0]["lr"]

    def train_epoch(self) -> float:
        """Train the model for one epoch.

        Parameters
        ----------
        Returns
        -------
        float
            Average training loss for the epoch
        """
        self.model.train()

        if self.ddp_enabled:
            self.train_loader.sampler.set_epoch(self.epoch)
        train_losses_per_epoch = {
            "total-MAE": torch.tensor(0.0, device=self.device),
            "total-MSE": torch.tensor(0.0, device=self.device),
            "total-RMSE": torch.tensor(0.0, device=self.device),
            "total-NMSE": torch.tensor(0.0, device=self.device),
            "total-RNMSE": torch.tensor(0.0, device=self.device),
            "total-VMSE": torch.tensor(0.0, device=self.device),
            "total-RVMSE": torch.tensor(0.0, device=self.device),
        }

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            x, target = batch
            x = x.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
            ):
                output = self.model(x)
                raw_loss = self.criterion(output, target)

            # Log training loss
            log_losses = self._compute_log_metrics(output.detach(), target.detach())
            log_losses[self.config["training"]["criterion"]] = raw_loss.detach()

            if self.use_amp:
                # Scale loss, backpropagate, unscale, clip, step, update
                self.grad_scaler.scale(raw_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                # Clip gradients to norm 1
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config["training"]["grad_clip"],
                )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                raw_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config["training"]["grad_clip"],
                )
                self.optimizer.step()

            ############################################################
            # Step learning rate scheduler #############################
            ############################################################
            lr = self._get_lr()  # get learning rate for logging
            if self.scheduler is not None:
                self.scheduler.step()

            ###############################################################
            # Accumulate losses ###########################################
            ###############################################################
            if self.ddp_enabled:
                # average the losses across all GPUs
                log_losses = self._reduce_all_losses(log_losses)
            for loss_name, log_loss in log_losses.items():
                train_losses_per_epoch[f"total-{loss_name}"] += log_loss

            ############################################################
            # Log training progress ####################################
            ############################################################
            self.world_total_samples_trained += self.batch_size * self.world_size
            self.world_total_batches_trained += self.world_size
            self.log_msg(
                "Training - Total: "
                f"Epoch {self.epoch}/{self.total_epochs}, "
                f"total batches {self.world_total_batches_trained}/{self.world_total_batches}, "
                f"total samples {self.world_total_samples_trained}/{self.world_total_samples}"
            )
            world_batch_idx = batch_idx * self.world_size
            world_samples_per_batch = self.batch_size * world_batch_idx
            self.log_msg(
                f"Training - Epoch: Batch (all GPUs): {world_batch_idx}/{self.world_train_batches_per_epoch}, "
                f"Samples (all GPUs): {world_samples_per_batch}/{self.world_train_samples_per_epoch}, "
                f"LR: {lr:.8f}"
            )
            ############################################################
            # Log losses ###############################################
            ############################################################
            wandb_log_losses = {}
            log_string = "Training - Epoch: "
            for loss_name, loss in log_losses.items():
                loss = loss.item()
                log_string += f"{loss_name}: {loss:.8f}, "
                wandb_log_losses[f"training-losses/{loss_name}"] = loss
            self.log_msg(log_string)
            self.log_msg("")

            ############################################################
            # Log to wandb #############################################
            ############################################################
            if self.global_rank == 0:
                self.wandb_run.log(
                    {
                        "training/total_batches_trained": self.world_total_batches_trained,
                        "training/total_samples_trained": self.world_total_samples_trained,
                        "training/total_samples_remaining": self.world_total_samples
                        - self.world_total_samples_trained,
                        "training/total_batches_remaining": self.world_total_batches
                        - self.world_total_batches_trained,
                        "training/epoch_batch_remaining": self.world_train_batches_per_epoch
                        - world_batch_idx,
                        "training/epoch_samples_remaining": self.world_train_samples_per_epoch
                        - world_samples_per_batch,
                        "training/learning_rate": lr,
                        "training/batch_idx": world_batch_idx,
                    },
                    commit=False,
                )
                self.wandb_run.log(wandb_log_losses, commit=True)

        ############################################################
        # Visualize predictions ####################################
        ############################################################
        if self.global_rank == 0:
            vis_path = self.epoch_dir / "train.png"
            try:
                visualize_predictions(
                    vis_path, x, output, target, num_samples=4, svg=True
                )
                log_predictions_wandb(
                    run=self.wandb_run,
                    image_path=vis_path.parent,
                    name_prefix=f"epoch_{self.epoch}",
                )
            except Exception as e:
                self.log_msg(f"Error visualizing predictions: {e}")
                self.log_msg(f"Error type: {type(e)}")
                self.log_msg(f"Error args: {e.args}")

        ############################################################
        # Calculate average loss per batch ########################
        ############################################################
        for loss_name, loss in train_losses_per_epoch.items():
            train_losses_per_epoch[loss_name] /= self.train_batches_per_epoch

        return train_losses_per_epoch

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        val_losses_per_epoch = {
            "total-MAE": torch.tensor(0.0, device=self.device),
            "total-MSE": torch.tensor(0.0, device=self.device),
            "total-RMSE": torch.tensor(0.0, device=self.device),
            "total-NMSE": torch.tensor(0.0, device=self.device),
            "total-RNMSE": torch.tensor(0.0, device=self.device),
            "total-VMSE": torch.tensor(0.0, device=self.device),
            "total-RVMSE": torch.tensor(0.0, device=self.device),
        }

        if self.ddp_enabled:
            self.val_loader.sampler.set_epoch(self.epoch)

        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.val_loader, start=1):
                x, target = batch
                x = x.to(self.device)
                target = target.to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_amp,
                ):
                    output = self.model(x)
                    raw_loss = self.criterion(output, target)

                # Log validation loss
                log_losses = self._compute_log_metrics(output.detach(), target.detach())
                log_losses[self.config["training"]["criterion"]] = raw_loss.detach()
                # Accumulate losses
                for loss_name, loss in log_losses.items():
                    val_losses_per_epoch[f"total-{loss_name}"] += loss

                self.log_msg(
                    "Validation - Total: "
                    f"Epoch {self.epoch}/{self.total_epochs}, "
                    f"Batch (all GPUs): {batch_idx * self.world_size}/{self.val_batches_per_epoch * self.world_size}, "
                )
                ############################################################
                # Log losses ###############################################
                ############################################################
                wandb_log_losses = {}
                log_string = "Validation - Epoch: "
                for loss_name, loss in log_losses.items():
                    loss = loss.item()
                    log_string += f"{loss_name}: {loss:.8f}, "
                    wandb_log_losses[f"validation-losses/{loss_name}"] = loss
                self.log_msg(log_string)
                self.log_msg("")

        ############################################################
        # Visualize predictions ####################################
        ############################################################
        if self.global_rank == 0:
            # Visualize predictions
            vis_path = self.epoch_dir / "val.png"
            try:
                visualize_predictions(
                    vis_path, x, output, target, num_samples=4, svg=True
                )
                log_predictions_wandb(
                    run=self.wandb_run,
                    image_path=vis_path.parent,
                    name_prefix=f"epoch_{self.epoch}",
                )
            except Exception as e:
                self.log_msg(f"Error visualizing predictions: {e}")
                self.log_msg(f"Error type: {type(e)}")
                self.log_msg(f"Error args: {e.args}")

        # Calculate average loss per batch
        for loss_name, loss in val_losses_per_epoch.items():
            val_losses_per_epoch[loss_name] /= self.val_batches_per_epoch

        return val_losses_per_epoch

    def train(self):
        """Train the model."""
        best_loss = float("inf")
        epochs = range(self.epoch, self.total_epochs + 1)
        for epoch in epochs:
            start_epoch_time = time.time()
            self.epoch = epoch
            self.epoch_dir = self.log_dir / f"epoch_{epoch:04d}"
            self.epoch_dir.mkdir(parents=True, exist_ok=True)
            ######################################################################
            ########### Training ###############################################
            ######################################################################
            self.log_msg("=" * 100)
            self.log_msg(f"Training: Training epoch {epoch}")
            train_losses = self.train_epoch()
            self.log_msg("=" * 100)
            log_string = "Training - Losses: "
            for loss_name, loss in train_losses.items():
                log_string += f"{loss_name}: {loss.item():.8f}, "
            self.log_msg(log_string)
            self.log_msg("=" * 100)
            if self.global_rank == 0:
                train_losses_wandb = {
                    f"training-summary/{k}": v for k, v in train_losses.items()
                }
                self.wandb_run.log(train_losses_wandb, commit=True)
            ######################################################################
            ########### Save checkpoint ########################################
            ######################################################################
            if self.global_rank == 0:
                self.save_checkpoint(name="checkpoint")
            if self.ddp_enabled:
                dist.barrier()

            ######################################################################
            ########### Validation ###############################################
            ######################################################################
            self.log_msg(f"Validation: Validating epoch {epoch}")
            val_losses = self.validate()
            self.log_msg("=" * 100)
            # Log all other losses for both training and validation
            log_string = "Validation - Losses: "
            for loss_name, loss in val_losses.items():
                log_string += f"{loss_name}: {loss.item():.8f}, "
            self.log_msg(log_string)
            self.log_msg("=" * 100)
            ############################################################
            # Calculate average time per epoch #########################
            ############################################################
            duration = time.time() - start_epoch_time
            self.log_msg(f"Summary: Epoch {epoch} took {duration / 60:.2f} minutes")
            self.avg_sec_per_epoch = (
                self.avg_sec_per_epoch * (self.epoch - 1) + duration
            ) / self.epoch
            self.log_msg(
                f"Summary: Average time per epoch: {self.avg_sec_per_epoch / 60:.2f} minutes"
            )
            if self.global_rank == 0:
                val_losses_wandb = {
                    f"validation-summary/{k}": v for k, v in val_losses.items()
                }
                self.wandb_run.log(val_losses_wandb, commit=True)
            ############################################################
            # Calculate time remaining #################################
            ############################################################
            proj_time_remaining = self.avg_sec_per_epoch * (
                self.total_epochs - self.epoch
            )
            self.log_msg(
                f"Summary: Projected time remaining: {proj_time_remaining / 60:.2f} minutes"
            )

            ######################################################################
            ########### Wandb logging ###########################################
            ######################################################################
            if self.global_rank == 0:
                self.wandb_run.log(
                    {
                        "summary/epoch": epoch,
                        "summary/minutes_per_epoch": duration / 60,
                        "summary/avg_minutes_per_epoch": self.avg_sec_per_epoch / 60,
                        "summary/projected_minutes_remaining": proj_time_remaining / 60,
                    },
                    commit=True,
                )

            ######################################################################
            ########### Save best model #########################################
            ######################################################################
            if self.global_rank == 0:
                criterion = self.config["training"]["criterion"]
                if val_losses[f"total-{criterion}"] < best_loss:
                    best_loss = val_losses[f"total-{criterion}"]
                    self.save_checkpoint(name="best_model")
                    self.log_msg(f"Model saved with loss: {best_loss:.8f}")

            ############################################################
            # Shut down if time limit is set and next epoch would exceed it
            ############################################################
            if self.time_limit is not None:
                time_passed = time.time() - self.start_time
                time_remaining = self.time_limit - time_passed
                if time_remaining < self.avg_sec_per_epoch:
                    self.log_msg(
                        "Summary: Next epoch would exceed time limit, shutting down"
                    )
                    break

        self.cleanup()

    def cleanup(self):
        if self.global_rank == 0:
            self.wandb_run.finish()
        if self.ddp_enabled:
            dist.destroy_process_group()


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lrs_config: dict,
    learning_rate: float,
    total_epochs: int,
    train_batches_per_epoch: int,
) -> optim.lr_scheduler.SequentialLR:
    """Create a learning rate scheduler.
    Options are only linear warmup or linear warmup followed by cosine annealing.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for training
    lrs_config : dict
        Configuration dictionary for the learning rate scheduler
    learning_rate : float
        Learning rate for training
    total_epochs : int
        Total number of training epochs
    train_batches_per_epoch : int
        Number of training batches per epoch

    Returns
    -------
    optim.lr_scheduler.SequentialLR
        Learning rate scheduler
    """
    # If total_epochs is 1, don't use a scheduler (debugging)
    if total_epochs == 1:
        return None

    first_stage = lrs_config["first_stage"]
    second_stage = lrs_config["second_stage"]
    third_stage = lrs_config["third_stage"] if "third_stage" in lrs_config else None

    ############################################################
    ###### Get batches for each stage #########################
    ############################################################
    milestones = []
    total_batches = total_epochs * train_batches_per_epoch
    first_stage_batches = first_stage["num_batches"]
    second_stage_batches = second_stage["num_batches"]
    if third_stage is not None:
        third_stage_batches = third_stage["num_batches"]
    else:
        third_stage_batches = 0

    if second_stage_batches == -1:
        second_stage_batches = total_batches - first_stage_batches - third_stage_batches

    first_milestone = first_stage_batches
    milestones.append(first_milestone)
    if third_stage is not None:
        # only add the second stage milestone if there is a third stage
        second_milestone = first_milestone + second_stage_batches
        milestones.append(second_milestone)

    ############################################################
    schedulers = []
    ############################################################
    ###### First stage #########################################
    ############################################################
    first_stage_name = first_stage["name"]
    if first_stage_name == "LinearLR":
        first_stage_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=first_stage["start_factor"],
            end_factor=first_stage["end_factor"],
            total_iters=first_stage_batches,
        )
        schedulers.append(first_stage_scheduler)
    else:
        raise ValueError(f"Scheduler {first_stage_name} not supported for first stage")

    ############################################################
    ###### Second stage ########################################
    ############################################################
    second_stage_name = second_stage["name"]
    if second_stage_name == "CosineAnnealingLR":
        second_stage_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=second_stage_batches,
            eta_min=float(second_stage["end_factor"]) * learning_rate,
        )
        schedulers.append(second_stage_scheduler)
    elif second_stage_name == "LinearLR":
        second_stage_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=second_stage["end_factor"],
            total_iters=second_stage_batches,
        )
        schedulers.append(second_stage_scheduler)

    else:
        raise ValueError(
            f"Scheduler {second_stage_name} not supported for second stage"
        )

    ############################################################
    ###### Third stage #########################################
    ############################################################
    if third_stage is not None:
        third_stage_name = third_stage["name"]
        if third_stage_name == "LinearLR":
            third_stage_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=second_stage["end_factor"],
                end_factor=third_stage["end_factor"],
                total_iters=third_stage_batches,
            )
            schedulers.append(third_stage_scheduler)
        else:
            raise ValueError(
                f"Scheduler {third_stage_name} not supported for third stage"
            )

    # agument the last scheduler with the remaining epochs
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )

    return scheduler


def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create an optimizer.

    Parameters
    ----------
    model : nn.Module
        The model to optimize
    config : dict
        Configuration dictionary for the optimizer

    Returns
    -------
    torch.optim.Optimizer
        Optimizer
    """
    if config["name"] == "Adam":
        betas = config["betas"]
        weight_decay = config["weight_decay"]
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            betas=betas,
            weight_decay=weight_decay,
        )
    elif config["name"] == "AdamW":
        weight_decay = config["weight_decay"]
        betas = config["betas"]
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=weight_decay,
            betas=betas,
        )
    elif config["name"] == "AdaptAdamW":
        weight_decay = config["weight_decay"]
        betas = config["betas"]
        optimizer = dadaptation.DAdaptAdam(
            model.parameters(),
            lr=config["learning_rate"],
            betas=betas,
            weight_decay=weight_decay,
            decouple=True,
        )
    elif config["name"] == "Prodigy":
        optimizer = prodigyopt.Prodigy(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=config["betas"],
            decouple=True,
            slice_p=config["slice_p"],
        )
    else:
        raise ValueError(f"Optimizer {config['name']} not supported")

    return optimizer


def login_wandb(config: dict) -> wandb.wandb_run.Run:
    """Log into wandb."""
    wandb_id = config["wandb"]["id"]
    wandb.login()
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config,
        id=wandb_id,
        tags=config["wandb"]["tags"],
        notes=config["wandb"]["notes"],
        resume="allow",
    )
    return run


def setup_ddp():
    dist.init_process_group()


@record
def main(
    config_path: Path,
    log_dir: Path | None,
    restart: bool,
    sim_name: str | None,
    data_dir: Path | None,
    time_limit: str | None,
    global_rank: int,
    local_rank: int,
    world_size: int,
):
    """Main training function."""
    load_dotenv()
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
        config["logging"]["log_dir"] = log_dir  # the actual dir is set in the trainer

    if data_dir is not None:
        data_dir = Path(data_dir)
        config["data"]["data_dir"] = data_dir

    if time_limit is not None:
        time_limit_seconds = sum(
            x * int(t) for x, t in zip([3600, 60, 1], time_limit.split(":"))
        )
        config["training"]["time_limit"] = time_limit_seconds

    if sim_name is not None:
        config["wandb"]["id"] = sim_name
        # config["logging"]["log_file"] = log_dir / sim_name / f"{sim_name}.log"

    if restart:
        checkpoint_dir = config["logging"]["log_dir"] / config["wandb"]["id"]
        checkpoint_path = find_last_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            print("No checkpoint found, starting from scratch")
        else:
            print(f"Restarting from checkpoint {checkpoint_path}")
    else:
        checkpoint_path = None

    ####################################################################
    ########### Initialize trainer #####################################
    ####################################################################
    if world_size > 1:
        setup_ddp()

    trainer = Trainer(config, global_rank, local_rank, world_size)
    if restart and checkpoint_path is not None:
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)
    trainer.save_config()
    trainer.train()


if __name__ == "__main__":
    ############################################################
    ########### Default arguments ##############################
    ############################################################

    default_config_path = Path(
        r"C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\logs\ti-main-run-single-0001\config.yaml"
    )
    default_log_dir = Path("logs")
    default_sim_name = None
    default_data_dir = Path("data/datasets")
    default_time_limit = None
    default_restart = True

    ############################################################
    ########### Parse arguments ################################
    ############################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=default_config_path)
    parser.add_argument("--log_dir", type=str, default=default_log_dir)
    parser.add_argument(
        "--restart", action=argparse.BooleanOptionalAction, default=default_restart
    )
    parser.add_argument("--sim_name", type=str, default=default_sim_name)
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--time_limit", type=str, default=default_time_limit)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))

    config_path = args.config_file
    log_dir = args.log_dir
    sim_name = args.sim_name
    data_dir = args.data_dir
    time_limit = args.time_limit
    restart = args.restart

    main(
        config_path=config_path,
        log_dir=log_dir,
        sim_name=sim_name,
        data_dir=data_dir,
        restart=restart,
        time_limit=time_limit,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
