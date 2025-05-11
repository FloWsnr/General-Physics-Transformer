"""
Training script for VQ-VAE models.

Author: Florian Wiesner
Date: 2025-04-07
"""

import os
from pathlib import Path
import time
import argparse
import platform
import math
from dataclasses import dataclass

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
from lpfm.model.tokenizer.vqvae import VQVAE
from lpfm.utils.train_vis import log_predictions_wandb, visualize_predictions
from lpfm.utils.logger import get_logger
from lpfm.model.transformer.loss_fns import RMSE
from lpfm.run.run_utils import (
    find_last_checkpoint,
    human_format,
    load_stored_model,
    path_to_string,
)
from lpfm.run.lr_scheduler import get_lr_scheduler


@dataclass
class LogState:
    total_samples: str
    total_batches: str
    val_every_x_samples: str
    val_every_x_batches: str
    val_samples: str
    val_batches: str


class VQVAETrainer:
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
        self.avg_sec_per_cycle = 0
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
            "VQVAETrainer",
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
        if "subdir_name" in self.config["logging"]:
            self.cycle_name = self.config["logging"]["subdir_name"]
        else:
            self.cycle_name = "val_"

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

        # Initialize VQ-VAE model
        vqvae_config = self.config["model"]["vqvae"]
        self.model = VQVAE(
            in_channels=vqvae_config["in_channels"],
            hidden_dim=vqvae_config["hidden_dim"],
            codebook_size=vqvae_config["num_embeddings"],
            codebook_dim=vqvae_config["embedding_dim"],
            commitment_cost=vqvae_config["commitment_cost"],
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        self.log_msg(f"Model size: {total_params / 1e6:.2f}M parameters")

        if self.global_rank == 0:
            self.wandb_run.config.update(
                {"model/model_size [M]": total_params / 1e6}, allow_val_change=True
            )

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
        batch_size = self.config["training"]["batch_size"]
        self.batch_size = batch_size * self.world_size

        self.total_batches_trained = 0
        self.total_samples_trained = 0
        self.cycle_idx = 0

        self.total_samples = int(float(self.config["training"]["samples"]))
        self.total_batches = self.total_samples // self.batch_size

        #################################################################
        ########### Initialize validation parameters ##################
        #################################################################
        frac_val_samples = float(self.config["training"]["val_frac_samples"])
        total_val_samples = len(self.val_loader) * self.batch_size
        self.val_samples = int(frac_val_samples * total_val_samples)
        self.val_every_x_samples = int(
            float(self.config["training"]["val_every_samples"])
        )
        self.val_every_x_batches = self.val_every_x_samples // self.batch_size
        self.val_batches = self.val_samples // self.batch_size

        self.h_log_state = LogState(
            total_samples=human_format(self.total_samples),
            total_batches=human_format(self.total_batches),
            val_every_x_samples=human_format(self.val_every_x_samples),
            val_every_x_batches=human_format(self.val_every_x_batches),
            val_samples=human_format(self.val_samples),
            val_batches=human_format(self.val_batches),
        )

        ################################################################
        ########### Log parameters #####################################
        ################################################################
        self.log_msg(f"Training for {self.h_log_state.total_samples} samples")
        self.log_msg(f"Training for {self.h_log_state.total_batches} batches")
        self.log_msg(f"Training on {self.batch_size} samples per batch")
        self.log_msg(
            f"Training with {self.config['training']['num_workers']} workers per GPU"
        )
        self.log_msg(f"Validating every {self.h_log_state.val_every_x_samples} samples")
        self.log_msg(f"Validating on {self.h_log_state.val_samples} samples")

        if self.global_rank == 0:
            self.wandb_run.config.update(
                {
                    "training/total_samples": self.total_samples,
                    "training/samples_per_batch": self.batch_size,
                    "training/val_every_samples": self.val_every_x_samples,
                    "training/num_val_samples": self.val_samples,
                    "training/slurm_id": os.environ.get("SLURM_JOB_ID", ""),
                },
                allow_val_change=True,
            )

        ################################################################
        ########### Initialize loss function and optimizer ###########
        ################################################################
        self.loss_fns = {
            "MAE": nn.L1Loss(),
            "MSE": nn.MSELoss(),
            "RMSE": RMSE(),
        }

        if self.config["training"]["criterion"] == "MSE":
            self.criterion = self.loss_fns.pop("MSE")
        elif self.config["training"]["criterion"] == "RMSE":
            self.criterion = self.loss_fns.pop("RMSE")
        elif self.config["training"]["criterion"] == "MAE":
            self.criterion = self.loss_fns.pop("MAE")
        else:
            raise ValueError(
                f"Criterion {self.config['training']['criterion']} not supported"
            )

        opt_config = self.config["training"]["optimizer"]
        self.optimizer = get_optimizer(self.model, opt_config)
        self.max_grad_norm = self.config["training"]["grad_clip"]

        ################################################################
        ########### Initialize learning rate scheduler ################
        ################################################################
        if "lr_scheduler" in self.config["training"]:
            lrs_config = self.config["training"]["lr_scheduler"]
            self.scheduler = get_lr_scheduler(
                optimizer=self.optimizer,
                lrs_config=lrs_config,
                total_batches=self.total_batches,
                total_batches_trained=self.total_batches_trained,
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
                log_freq=self.val_every_x_batches,
            )

    def log_msg(self, msg: str):
        """Log a message."""
        if self.global_rank == 0:
            self.logger.info(msg)

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        restart: bool = False,
        new_training: bool = False,
    ):
        """Restart training from a checkpoint."""
        if (restart and new_training) or (not restart and not new_training):
            raise ValueError("Invalid combination of restart and new_training")

        self.log_msg(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_stored_model(checkpoint_path, self.device, remove_ddp=False)
        if not new_training:
            self.total_samples_trained = checkpoint["samples_trained"]
            self.total_batches_trained = checkpoint["batches_trained"]
            self.cycle_idx = checkpoint["cycle_idx"]

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        if self.scheduler is not None and restart:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif self.scheduler is not None and new_training:
            self.scheduler = get_lr_scheduler(
                optimizer=self.optimizer,
                lrs_config=self.config["training"]["lr_scheduler"],
                total_batches=self.total_batches,
                total_batches_trained=self.total_batches_trained,
            )
        self.log_msg(
            f"Restarting training from {self.total_samples_trained} samples trained"
        )

    def save_checkpoint(self, path: Path):
        """Save a checkpoint."""
        checkpoint = {
            "samples_trained": self.total_samples_trained,
            "batches_trained": self.total_batches_trained,
            "cycle_idx": self.cycle_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": path_to_string(self.config),
        }
        if self.grad_scaler is not None:
            checkpoint["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        self.log_msg(f"Summary: Checkpoint saved to {path}")

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
        if isinstance(self.optimizer, torch.optim.AdamW):
            return self.scheduler.get_last_lr()[0]
        elif isinstance(self.optimizer, dadaptation.DAdaptAdam):
            return (
                self.optimizer.param_groups[0]["lr"]
                * self.optimizer.param_groups[0]["d"]
            )
        elif isinstance(self.optimizer, prodigyopt.Prodigy):
            return (
                self.optimizer.param_groups[0]["lr"]
                * self.optimizer.param_groups[0]["d"]
            )
        else:
            return self.optimizer.param_groups[0]["lr"]

    def train_for_x_samples(self, num_samples: int) -> float:
        """Train the model for a given number of samples."""
        self.model.train()

        if self.ddp_enabled:
            self.train_loader.sampler.set_epoch(self.cycle_idx)

        loss_per_cycle = {
            "total-MAE": torch.tensor(0.0, device=self.device),
            "total-MSE": torch.tensor(0.0, device=self.device),
            "total-RMSE": torch.tensor(0.0, device=self.device),
            "total-codebook": torch.tensor(0.0, device=self.device),
        }

        samples_trained = 0
        batches_trained = 0
        train_iter = iter(self.train_loader)
        while samples_trained < num_samples:
            try:
                x, target = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, target = next(train_iter)

            x = x.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
            ):
                x_recon, codebook_loss, _ = self.model(x)
                recon_loss = self.criterion(x_recon, target)
                loss = recon_loss + codebook_loss

            # Log training loss
            log_losses = self._compute_log_metrics(x_recon.detach(), target.detach())
            log_losses[self.config["training"]["criterion"]] = recon_loss.detach()
            log_losses["codebook"] = codebook_loss.detach()

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.optimizer.step()

            lr = self._get_lr()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.ddp_enabled:
                log_losses = self._reduce_all_losses(log_losses)
            for loss_name, log_loss in log_losses.items():
                loss_per_cycle[f"total-{loss_name}"] += log_loss

            samples_trained += self.batch_size
            batches_trained += 1
            self.total_samples_trained += self.batch_size
            self.total_batches_trained += 1

            s_cycle_trained_human = human_format(samples_trained)
            b_cycle_trained_human = human_format(batches_trained)
            s_cycle_human = self.h_log_state.val_every_x_samples
            b_cycle_human = self.h_log_state.val_every_x_batches

            s_total_trained_human = human_format(self.total_samples_trained)
            b_total_trained_human = human_format(self.total_batches_trained)
            s_total_human = self.h_log_state.total_samples
            b_total_human = self.h_log_state.total_batches

            self.log_msg(
                "Training - Total: "
                f"Samples: {s_total_trained_human}/{s_total_human}, "
                f"Batches: {b_total_trained_human}/{b_total_human}, "
                f"LR: {lr:.2e}"
            )
            self.log_msg(
                f"Training - Cycle: {self.cycle_idx} "
                f"Samples: {s_cycle_trained_human}/{s_cycle_human}, "
                f"Batches: {b_cycle_trained_human}/{b_cycle_human}, "
                f"LR: {lr:.2e}"
            )

            wandb_log_losses = {}
            log_string = "Training - Batch: "
            for loss_name, loss in log_losses.items():
                loss = loss.item()
                log_string += f"{loss_name}: {loss:.8f}, "
                wandb_log_losses[f"training-losses/{loss_name}"] = loss
            self.log_msg(log_string)
            self.log_msg("")

            if self.global_rank == 0:
                self.wandb_run.log(
                    {
                        "training/total_batches_trained": self.total_batches_trained,
                        "training/total_samples_trained": self.total_samples_trained,
                        "training/total_samples_remaining": self.total_samples
                        - self.total_samples_trained,
                        "training/total_batches_remaining": self.total_batches
                        - self.total_batches_trained,
                        "training/learning_rate": lr,
                    },
                    commit=False,
                )
                self.wandb_run.log(wandb_log_losses, commit=True)

        if self.global_rank == 0:
            vis_path = self.val_dir / "train.png"
            try:
                visualize_predictions(
                    vis_path,
                    x.float(),
                    x_recon.float(),
                    target.float(),
                    num_samples=4,
                    svg=True,
                )
                log_predictions_wandb(
                    run=self.wandb_run,
                    image_path=vis_path.parent,
                    name_prefix=f"cycle_{self.cycle_idx}",
                )
            except Exception as e:
                self.log_msg(f"Error visualizing predictions: {e}")
                self.log_msg(f"Error type: {type(e)}")
                self.log_msg(f"Error args: {e.args}")

        for loss_name, loss in loss_per_cycle.items():
            loss_per_cycle[loss_name] /= batches_trained

        return loss_per_cycle

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        loss_per_cycle = {
            "total-MAE": torch.tensor(0.0, device=self.device),
            "total-MSE": torch.tensor(0.0, device=self.device),
            "total-RMSE": torch.tensor(0.0, device=self.device),
            "total-codebook": torch.tensor(0.0, device=self.device),
        }

        if self.ddp_enabled:
            self.val_loader.sampler.set_epoch(self.cycle_idx)

        val_iter = iter(self.val_loader)
        samples_validated = 0
        batches_validated = 0
        with torch.inference_mode():
            while samples_validated < self.val_samples:
                try:
                    x, target = next(val_iter)
                except StopIteration:
                    val_iter = iter(self.val_loader)
                    x, target = next(val_iter)

                x = x.to(self.device)
                target = target.to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_amp,
                ):
                    x_recon, codebook_loss, _ = self.model(x)
                    recon_loss = self.criterion(x_recon, target)

                log_losses = self._compute_log_metrics(
                    x_recon.detach(), target.detach()
                )
                log_losses[self.config["training"]["criterion"]] = recon_loss.detach()
                log_losses["codebook"] = codebook_loss.detach()

                if self.ddp_enabled:
                    log_losses = self._reduce_all_losses(log_losses)

                for loss_name, log_loss in log_losses.items():
                    loss_per_cycle[f"total-{loss_name}"] += log_loss

                samples_validated += self.batch_size
                batches_validated += 1

                s_cycle_validated_human = human_format(samples_validated)
                b_cycle_validated_human = human_format(batches_validated)
                s_cycle_human = self.h_log_state.val_samples
                b_cycle_human = self.h_log_state.val_batches
                self.log_msg(
                    f"Validation - Cycle: {self.cycle_idx} "
                    f"Samples: {s_cycle_validated_human}/{s_cycle_human}, "
                    f"Batches: {b_cycle_validated_human}/{b_cycle_human}, "
                )
                self.log_msg("")

        if self.global_rank == 0:
            vis_path = self.val_dir / "val.png"
            try:
                visualize_predictions(
                    vis_path,
                    x.float(),
                    x_recon.float(),
                    target.float(),
                    num_samples=4,
                    svg=True,
                )
                log_predictions_wandb(
                    run=self.wandb_run,
                    image_path=vis_path.parent,
                    name_prefix=f"cycle_{self.cycle_idx}",
                )
            except Exception as e:
                self.log_msg(f"Error visualizing predictions: {e}")
                self.log_msg(f"Error type: {type(e)}")
                self.log_msg(f"Error args: {e.args}")

        for loss_name, loss in loss_per_cycle.items():
            loss_per_cycle[loss_name] /= batches_validated

        return loss_per_cycle

    def train(self):
        """Train the model."""
        best_loss = float("inf")
        while self.total_samples_trained < self.total_samples:
            self.cycle_idx += 1
            start_epoch_time = time.time()
            self.val_dir = self.log_dir / f"{self.cycle_name}{self.cycle_idx:04d}"
            self.val_dir.mkdir(parents=True, exist_ok=True)

            self.log_msg("=" * 100)
            self.log_msg(f"Training - Cycle {self.cycle_idx}")
            self.log_msg(f"Training - train on next {self.val_every_x_samples} samples")
            train_losses = self.train_for_x_samples(self.val_every_x_samples)
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
                train_losses_wandb["training-summary/samples_trained"] = (
                    self.total_samples_trained
                )
                train_losses_wandb["training-summary/samples_remaining"] = (
                    self.total_samples - self.total_samples_trained
                )
                train_losses_wandb["training-summary/cycle_idx"] = self.cycle_idx
                self.wandb_run.log(train_losses_wandb, commit=True)

            if self.global_rank == 0:
                self.save_checkpoint(path=self.val_dir / "checkpoint.pth")
            if self.ddp_enabled:
                dist.barrier()

            self.log_msg("Validation")
            val_losses = self.validate()
            self.log_msg("=" * 100)
            log_string = "Validation - Losses: "
            for loss_name, loss in val_losses.items():
                log_string += f"{loss_name}: {loss.item():.8f}, "
            self.log_msg(log_string)
            self.log_msg("=" * 100)

            if self.global_rank == 0:
                val_losses_wandb = {
                    f"validation-summary/{k}": v for k, v in val_losses.items()
                }
                val_losses_wandb["validation-summary/samples_trained"] = (
                    self.total_samples_trained
                )
                val_losses_wandb["validation-summary/batches_trained"] = (
                    self.total_batches_trained
                )
                val_losses_wandb["validation-summary/cycle_idx"] = self.cycle_idx
                self.wandb_run.log(val_losses_wandb, commit=True)

            duration = time.time() - start_epoch_time
            self.log_msg(f"Summary: Training cycle took {duration / 60:.2f} minutes")
            self.avg_sec_per_cycle = (
                self.avg_sec_per_cycle * (self.cycle_idx - 1) + duration
            ) / self.cycle_idx

            self.log_msg(
                f"Summary: Average time per cycle: {self.avg_sec_per_cycle / 60:.2f} minutes"
            )

            rem_cycles = (
                self.total_samples - self.total_samples_trained
            ) / self.val_every_x_samples
            rem_cycles = math.ceil(rem_cycles)

            proj_time_remaining = self.avg_sec_per_cycle * rem_cycles
            self.log_msg(
                f"Summary: Projected time remaining: {proj_time_remaining / 60:.2f} minutes"
            )

            if self.global_rank == 0:
                self.wandb_run.log(
                    {
                        "summary/cycle_idx": self.cycle_idx,
                        "summary/samples_trained": self.total_samples_trained,
                        "summary/batches_trained": self.total_batches_trained,
                        "summary/minutes_per_cycle": duration / 60,
                        "summary/avg_minutes_per_cycle": self.avg_sec_per_cycle / 60,
                        "summary/projected_minutes_remaining": proj_time_remaining / 60,
                    },
                    commit=True,
                )

            if self.global_rank == 0:
                criterion = self.config["training"]["criterion"]
                if val_losses[f"total-{criterion}"] < best_loss:
                    best_loss = val_losses[f"total-{criterion}"]
                    self.save_checkpoint(path=self.log_dir / "best_model.pth")
                    self.log_msg(f"Model saved with loss: {best_loss:.8f}")

            if self.time_limit is not None:
                time_passed = time.time() - self.start_time
                time_remaining = self.time_limit - time_passed
                if time_remaining < self.avg_sec_per_cycle:
                    self.log_msg(
                        "Summary: Next cycle would exceed time limit, shutting down"
                    )
                    break

        self.cleanup()

    def cleanup(self):
        if self.global_rank == 0:
            self.wandb_run.finish()
        if self.ddp_enabled:
            dist.destroy_process_group()


def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create an optimizer."""
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
    new_training: bool,
    best_model: bool,
    sim_name: str | None,
    data_dir: Path | None,
    time_limit: str | None,
    global_rank: int,
    local_rank: int,
    world_size: int,
):
    """Main training function."""
    load_dotenv()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)

    if log_dir is not None:
        log_dir = Path(log_dir)
        config["logging"]["log_dir"] = log_dir

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

    if restart or new_training:
        checkpoint_dir = config["logging"]["log_dir"] / config["wandb"]["id"]

        if "subdir_name" in config["logging"]:
            subdir_name = config["logging"]["subdir_name"]
        else:
            subdir_name = "val_"

        checkpoint_path = find_last_checkpoint(checkpoint_dir, subdir_name, best_model)
        if checkpoint_path is None:
            if global_rank == 0:
                print("No checkpoint found, starting from scratch")
        else:
            if global_rank == 0:
                print(f"Restarting from checkpoint {checkpoint_path}")
    else:
        checkpoint_path = None

    if world_size > 1:
        setup_ddp()

    trainer = VQVAETrainer(config, global_rank, local_rank, world_size)
    if checkpoint_path is not None:
        trainer.load_checkpoint(
            checkpoint_path=checkpoint_path,
            restart=restart,
            new_training=new_training,
        )
    trainer.save_config()
    trainer.train()


if __name__ == "__main__":
    default_config_path = Path(
        r"C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\lpfm\run\config.yaml"
    )
    default_log_dir = Path("logs")
    default_sim_name = None
    default_data_dir = Path("data/datasets")
    default_time_limit = None
    default_restart = False
    default_new_training = False
    default_best_model = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=default_config_path)
    parser.add_argument("--log_dir", type=str, default=default_log_dir)
    parser.add_argument(
        "--restart", action=argparse.BooleanOptionalAction, default=default_restart
    )
    parser.add_argument(
        "--new_training",
        action=argparse.BooleanOptionalAction,
        default=default_new_training,
    )
    parser.add_argument(
        "--best_model",
        action=argparse.BooleanOptionalAction,
        default=default_best_model,
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
    new_training = args.new_training
    best_model = args.best_model

    main(
        config_path=config_path,
        log_dir=log_dir,
        sim_name=sim_name,
        data_dir=data_dir,
        restart=restart,
        new_training=new_training,
        best_model=best_model,
        time_limit=time_limit,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
