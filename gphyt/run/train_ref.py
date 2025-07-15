"""
Training script for the reference models.

Author: Florian Wiesner
Date: 2025-07-11
"""

import os
from pathlib import Path
import time
import argparse
import math
from dataclasses import dataclass

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

from gphyt.data.dataset_utils import get_dataloader
from gphyt.utils.train_vis import visualize_predictions
from gphyt.utils.logger import get_logger
from gphyt.utils.wandb_logger import WandbLogger
from gphyt.model.transformer.loss_fns import RMSE
from gphyt.run.run_utils import (
    find_checkpoint,
    human_format,
    load_stored_model,
    path_to_string,
)
from gphyt.run.lr_scheduler import get_lr_scheduler
from gphyt.model.model_specs import FNO_M, FNO_S, ResNet_M, ResNet_S
import gphyt.model.resnet as resnet
import gphyt.model.fno as fno


@dataclass
class LogState:
    total_samples: str
    total_batches: str
    val_every_x_samples: str
    val_every_x_batches: str
    val_samples: str
    val_batches: str


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
        self.avg_sec_per_cycle = 0
        self.avg_sec_per_checkpoint = 0
        self.avg_sec_per_1k_samples = 0
        self.avg_sec_per_val_cycle = 0
        self.shutdown_flag = torch.tensor(0, device=self.device)

        if "time_limit" in self.config["training"]:
            self.time_limit = self.config["training"]["time_limit"]
        else:
            self.time_limit = None

        ################################################################
        ########### Initialize config #################################
        ################################################################
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
        if "subdir_name" in self.config["logging"]:
            self.cycle_name = self.config["logging"]["subdir_name"]
        else:
            self.cycle_name = "val_"

        if self.global_rank == 0:
            self.wandb_logger = WandbLogger(
                config=self.config,
                log_file=self.config["logging"]["log_file"],
                log_level=self.config["logging"]["log_level"],
            )
        else:
            self.wandb_logger = None

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

        # check if model is FNO_M or FNO_S
        if self.config["model"] == "fno-M":
            self.model = fno.get_model(FNO_M())
            # fno models are not supported with AMP
            self.use_amp = False
            self.grad_scaler = None
        elif self.config["model"] == "fno-S":
            self.model = fno.get_model(FNO_S())
            # fno models are not supported with AMP
            self.use_amp = False
            self.grad_scaler = None
        elif self.config["model"] == "resnet-M":
            n_time_steps = self.config["data"]["n_steps_input"]
            model = resnet.get_model(ResNet_M(), n_time_steps=n_time_steps)
            self.model = torch.compile(model, mode="max-autotune")
            self.use_amp = True
            self.grad_scaler = GradScaler()
        elif self.config["model"] == "resnet-S":
            n_time_steps = self.config["data"]["n_steps_input"]
            model = resnet.get_model(ResNet_S(), n_time_steps=n_time_steps)
            self.model = torch.compile(model, mode="max-autotune")
            self.use_amp = True
            self.grad_scaler = GradScaler()
        else:
            raise ValueError(f"Model {self.config['model']} not supported")

        total_params = sum(p.numel() for p in self.model.parameters())
        self.log_msg(f"Model size: {total_params / 1e6:.2f}M parameters")

        if self.global_rank == 0:
            self.wandb_logger.update_config(
                {"model/model_size [M]": total_params / 1e6}
            )

        # print the model architecture
        self.model.to(self.device)
        torch.set_float32_matmul_precision("high")

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
            seed=self.config["training"]["seed"],
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"]["num_workers"],
            prefetch_factor=self.config["training"]["prefetch_factor"],
            split="train",
            is_distributed=self.ddp_enabled,
            shuffle=True,
        )
        self.val_loader = get_dataloader(
            self.config["data"],
            seed=self.config["training"]["seed"],
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"]["num_workers"],
            prefetch_factor=self.config["training"]["prefetch_factor"],
            split="val",
            is_distributed=self.ddp_enabled,
            shuffle=False,
            data_fraction=self.config["training"]["val_frac_samples"],
        )
        ################################################################
        ########### Initialize training parameters ##################
        ################################################################
        # NOTE: These are per GPU worker
        batch_size = self.config["training"]["batch_size"]
        # NOTE: This is across all GPU workers
        self.batch_size = batch_size * self.world_size

        self.total_batches_trained = 0
        self.total_samples_trained = 0
        self.cycle_idx = 0

        # Change from samples to batches
        self.total_batches = int(float(self.config["training"]["batches"]))
        self.total_samples = self.total_batches * self.batch_size

        #################################################################
        ########### Initialize validation parameters ##################
        #################################################################
        val_batches = len(self.val_loader)
        total_val_samples = val_batches * self.batch_size
        # num training batches per validation loop
        self.val_every_x_batches = int(
            float(self.config["training"]["val_every_batches"])
        )
        self.val_every_x_samples = self.val_every_x_batches * self.batch_size

        self.checkpoint_every_x_batches = int(
            float(self.config["training"]["checkpoint_every_batches"])
        )

        ###################################################################
        self.h_log_state = LogState(
            total_samples=human_format(self.total_samples),
            total_batches=human_format(self.total_batches),
            val_every_x_samples=human_format(self.val_every_x_samples),
            val_every_x_batches=human_format(self.val_every_x_batches),
            val_samples=human_format(total_val_samples),
            val_batches=human_format(val_batches),
        )
        ################################################################
        ########### Log parameters #####################################
        ################################################################
        self.log_msg(f"Number of unique training batches: {len(self.train_loader)}")
        self.log_msg(
            f"Number of unique training samples: {len(self.train_loader) * self.batch_size}"
        )

        self.log_msg(f"Training for {self.h_log_state.total_samples} samples")
        self.log_msg(f"Training for {self.h_log_state.total_batches} batches")
        self.log_msg(f"Training on {self.batch_size} samples per batch")

        self.log_msg(f"Validating every {self.h_log_state.val_every_x_samples} samples")
        self.log_msg(f"Validating on {self.h_log_state.val_samples} samples")
        self.log_msg(f"Validating on {self.h_log_state.val_batches} batches")
        self.log_msg(
            f"Checkpoint every {self.config['training']['checkpoint_every_batches']} batches"
        )
        self.log_msg(
            f"Running {self.config['training']['num_workers']} dataloader workers per GPU"
        )

        if self.global_rank == 0:
            self.wandb_logger.update_config(
                {
                    "training/total_samples": self.total_samples,
                    "training/samples_per_batch": self.batch_size,
                    "training/val_every_batches": self.val_every_x_batches,
                    "training/checkpoint_every_batches": self.checkpoint_every_x_batches,
                    "training/num_val_samples": total_val_samples,
                    "training/slurm_id": os.environ.get("SLURM_JOB_ID", ""),
                },
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
            self.criterion = self.loss_fns["MSE"]
        elif self.config["training"]["criterion"] == "RMSE":
            self.criterion = self.loss_fns["RMSE"]
        elif self.config["training"]["criterion"] == "MAE":
            self.criterion = self.loss_fns["MAE"]
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
            self.wandb_logger.watch(
                model=self.model,
                criterion=self.criterion,
                log=self.config["wandb"]["log_model"],
                log_freq=self.checkpoint_every_x_batches,
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

        # can be either a restart or a new training from checkpoint
        # check xor for both false or both true
        if (restart and new_training) or (not restart and not new_training):
            raise ValueError("Invalid combination of restart and new_training")

        self.log_msg(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_stored_model(checkpoint_path, self.device, remove_ddp=False)
        if not new_training:
            self.total_samples_trained = checkpoint["samples_trained"]
            self.total_batches_trained = checkpoint["batches_trained"]
            self.cycle_idx = checkpoint["cycle_idx"]
        ##################################################################
        ########## Load model, optimizer, and scheduler ##################
        ##################################################################
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
            if self.scheduler is not None:
                return self.scheduler.get_last_lr()[0]
            else:
                return self.optimizer.param_groups[0]["lr"]

        else:
            return self.optimizer.param_groups[0]["lr"]

    def train_for_x_batches(self, num_batches: int) -> float:
        """Train the model for a given number of samples.

        Parameters
        ----------
        num_batches : int
            Number of batches to train for

        Returns
        -------
        float
            Average training loss for the samples
        """
        self.model.train()

        if self.ddp_enabled:
            self.train_loader.sampler.set_epoch(self.cycle_idx)

        loss_per_cycle = {
            "total-MAE": torch.tensor(0.0, device=self.device),
            "total-MSE": torch.tensor(0.0, device=self.device),
            "total-RMSE": torch.tensor(0.0, device=self.device),
        }

        samples_trained = 0
        train_time = time.time()
        # check that num_batches does not exceed remaining batches
        num_batches = min(num_batches, self.total_batches - self.total_batches_trained)

        batches_trained = 0
        train_iter = iter(self.train_loader)
        while (batches_trained < num_batches) and self.shutdown_flag.item() == 0:
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
                output = self.model(x)
                raw_loss = self.criterion(output, target)

            # Log training loss
            log_losses = self._compute_log_metrics(output.detach(), target.detach())

            if self.use_amp:
                self.grad_scaler.scale(raw_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                raw_loss.backward()
            if self.max_grad_norm is not None:
                # Clip gradients to norm 1
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm,
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
                loss_per_cycle[f"total-{loss_name}"] += log_loss

            ############################################################
            # Log training progress ####################################
            ############################################################
            samples_trained += self.batch_size
            batches_trained += 1
            self.total_samples_trained += self.batch_size
            self.total_batches_trained += 1
            # convert to human readable
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
            ############################################################
            # Log losses ###############################################
            ############################################################
            wandb_log_losses = {}
            log_string = "Training - Batch: "
            for loss_name, loss in log_losses.items():
                loss = loss.item()
                log_string += f"{loss_name}: {loss:.8f}, "
                wandb_log_losses[f"training-losses/{loss_name}"] = loss
            self.log_msg(log_string)
            self.log_msg("")

            ############################################################
            ############ Update time estimates ########################
            ############################################################
            total_train_duration = time.time() - train_time
            seconds_per_batch = total_train_duration / batches_trained
            self.avg_sec_per_1k_batches = seconds_per_batch * 1000

            self.avg_sec_per_checkpoint = (
                seconds_per_batch * self.checkpoint_every_x_batches
            )

            ############################################################
            # Log to wandb #############################################
            ############################################################
            if self.global_rank == 0:
                self.wandb_logger.log(
                    {
                        "training/total_batches_trained": self.total_batches_trained,
                        "training/total_samples_trained": self.total_samples_trained,
                        "training/total_samples_remaining": self.total_samples
                        - self.total_samples_trained,
                        "training/total_batches_remaining": self.total_batches
                        - self.total_batches_trained,
                        "training/learning_rate": lr,
                        "training/avg_sec_per_1k_batches": self.avg_sec_per_1k_batches,
                        "training/avg_sec_per_checkpoint": self.avg_sec_per_checkpoint,
                    },
                    commit=False,
                )
                self.wandb_logger.log(wandb_log_losses, commit=True)
            ############################################################
            # Save checkpoint ##########################################
            ############################################################
            next_checkpoint = (
                self.total_batches_trained // self.checkpoint_every_x_batches + 1
            ) * self.checkpoint_every_x_batches
            if self.total_batches_trained >= next_checkpoint - 1:
                if self.global_rank == 0:
                    self.save_checkpoint(path=self.log_dir / "last_checkpoint.pth")
                if self.ddp_enabled:
                    dist.barrier()

            ############################################################
            # Shut down if next checkpoint would exceed time limit
            ############################################################
            time_remaining = self.time_limit - (time.time() - self.start_time)
            time_needed = (
                self.avg_sec_per_checkpoint + self.avg_sec_per_val_cycle
            ) * 1.2

            # only check if we have already trained for at least one cycle to make sure
            # the estimate is not too far off
            if (
                time_remaining < time_needed
                and batches_trained > self.checkpoint_every_x_batches / 2
            ):
                self.shutdown_flag = torch.tensor(
                    1, device=self.device
                )  # set flag to tell outer loop to shut down
                if self.ddp_enabled:
                    dist.all_reduce(self.shutdown_flag, op=dist.ReduceOp.SUM)
                self.log_msg(
                    "Summary: Next checkpoint would exceed time limit, shutting down"
                )
            if self.ddp_enabled:
                dist.barrier()
        ############################################################
        # Visualize predictions ####################################
        ############################################################
        if self.global_rank == 0:
            vis_path = self.val_dir / "train.png"
            try:
                visualize_predictions(
                    vis_path,
                    x.float(),
                    output.float(),
                    target.float(),
                    num_samples=4,
                    svg=True,
                )
                self.wandb_logger.log_predictions(
                    image_path=vis_path.parent,
                    name_prefix=f"cycle_{self.cycle_idx}",
                )
                self.log_msg("Predictions visualized and logged to wandb")
            except Exception as e:
                self.log_msg(f"Error visualizing predictions: {e}")
                self.log_msg(f"Error type: {type(e)}")
                self.log_msg(f"Error args: {e.args}")

        ############################################################
        # Calculate average loss per batch ########################
        ############################################################
        for loss_name, loss in loss_per_cycle.items():
            loss_per_cycle[loss_name] /= batches_trained

        return loss_per_cycle

    @torch.inference_mode()
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        loss_per_cycle = {
            "total-MAE": torch.tensor(0.0, device=self.device),
            "total-MSE": torch.tensor(0.0, device=self.device),
            "total-RMSE": torch.tensor(0.0, device=self.device),
        }

        if self.ddp_enabled:
            self.val_loader.sampler.set_epoch(self.cycle_idx)

        samples_validated = 0
        batches_validated = 0
        start_val_time = time.time()
        for x, target in self.val_loader:
            x = x.to(self.device)
            target = target.to(self.device)

            output = self.model(x)
            # Log validation loss
            log_losses = self._compute_log_metrics(output.detach(), target.detach())
            ###############################################################
            # Accumulate losses ###########################################
            ###############################################################
            if self.ddp_enabled:
                # average the losses across all GPUs
                log_losses = self._reduce_all_losses(log_losses)

            for loss_name, log_loss in log_losses.items():
                loss_per_cycle[f"total-{loss_name}"] += log_loss

            ####################################################
            # Update cycle index #################################
            ####################################################
            samples_validated += self.batch_size
            batches_validated += 1
            ####################################################
            # Log validation progress ##########################
            ####################################################
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

        ############################################################
        # Visualize predictions ####################################
        ############################################################
        if self.global_rank == 0:
            # Visualize predictions
            vis_path = self.val_dir / "val.png"
            try:
                visualize_predictions(
                    vis_path,
                    x.float(),
                    output.float(),
                    target.float(),
                    num_samples=4,
                    svg=True,
                )
                self.wandb_logger.log_predictions(
                    image_path=vis_path.parent,
                    name_prefix=f"cycle_{self.cycle_idx}",
                )
            except Exception as e:
                self.log_msg(f"Error visualizing predictions: {e}")
                self.log_msg(f"Error type: {type(e)}")
                self.log_msg(f"Error args: {e.args}")

        ############################################################
        # Calculate average time per cycle #########################
        ############################################################
        duration = time.time() - start_val_time
        self.avg_sec_per_val_cycle = (
            self.avg_sec_per_val_cycle * (self.num_cycles - 1) + duration
        ) / self.num_cycles

        # Calculate average loss per batch
        for loss_name, loss in loss_per_cycle.items():
            loss_per_cycle[loss_name] /= batches_validated

        return loss_per_cycle

    def train(self):
        """Train the model."""
        best_loss = float("inf")
        # we use this to count the number of cycles of this training run
        # this is used to calculate the projected time remaining. If we use cycle idx and do a
        # restart, the projected time remaining will be wrong.
        self.num_cycles = 0
        self.start_time = time.time()
        while (
            self.total_batches_trained < self.total_batches
        ) and self.shutdown_flag.item() == 0:
            self.num_cycles += 1
            self.cycle_idx += 1
            start_cycle_time = time.time()
            self.val_dir = self.log_dir / f"{self.cycle_name}{self.cycle_idx:04d}"
            self.val_dir.mkdir(parents=True, exist_ok=True)
            ######################################################################
            ########### Training ###############################################
            ######################################################################
            self.log_msg("=" * 100)
            self.log_msg(f"Training - Cycle {self.cycle_idx}")
            self.log_msg(f"Training - train on next {self.val_every_x_batches} batches")
            train_losses = self.train_for_x_batches(self.val_every_x_batches)
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
                train_losses_wandb["training-summary/batches_trained"] = (
                    self.total_batches_trained
                )
                train_losses_wandb["training-summary/batches_remaining"] = (
                    self.total_batches - self.total_batches_trained
                )
                train_losses_wandb["training-summary/samples_trained"] = (
                    self.total_samples_trained
                )
                train_losses_wandb["training-summary/cycle_idx"] = self.cycle_idx
                self.wandb_logger.log(train_losses_wandb, commit=True)
            ######################################################################
            ########### Save checkpoint ########################################
            ######################################################################
            if self.global_rank == 0:
                self.save_checkpoint(path=self.val_dir / "checkpoint.pth")
            if self.ddp_enabled:
                dist.barrier()
            ######################################################################
            ########### Validation ###############################################
            ######################################################################
            self.log_msg("Validation")
            val_losses = self.validate()
            self.log_msg("=" * 100)
            # Log all other losses for both training and validation
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
                val_losses_wandb["validation-summary/avg_sec_per_val_cycle"] = (
                    self.avg_sec_per_val_cycle
                )
                self.wandb_logger.log(val_losses_wandb, commit=True)

            ############################################################
            # Calculate average time per cycle #########################
            ############################################################
            duration = time.time() - start_cycle_time
            self.log_msg(f"Summary: Training cycle took {duration / 60:.2f} minutes")
            self.avg_sec_per_cycle = (
                self.avg_sec_per_cycle * (self.num_cycles - 1) + duration
            ) / self.num_cycles

            self.log_msg(
                f"Summary: Average time per cycle: {self.avg_sec_per_cycle / 60:.2f} minutes"
            )
            ############################################################
            # Calculate time remaining #################################
            ############################################################
            rem_cycles = (
                self.total_batches - self.total_batches_trained
            ) / self.val_every_x_batches
            # round up to nearest integer
            rem_cycles = math.ceil(rem_cycles)

            proj_time_remaining = self.avg_sec_per_cycle * rem_cycles
            self.log_msg(
                f"Summary: Projected time remaining: {proj_time_remaining / 60:.2f} minutes"
            )
            ######################################################################
            ########### Wandb logging ###########################################
            ######################################################################
            if self.global_rank == 0:
                self.wandb_logger.log(
                    {
                        "summary/cycle_idx": self.cycle_idx,
                        "summary/samples_trained": self.total_samples_trained,
                        "summary/batches_trained": self.total_batches_trained,
                        "summary/minutes_per_cycle": duration / 60,
                        "summary/avg_minutes_per_cycle": self.avg_sec_per_cycle / 60,
                        "summary/projected_minutes_remaining": proj_time_remaining / 60,
                        "summary/seconds_per_1k_batches": self.avg_sec_per_1k_batches,
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
                    self.save_checkpoint(path=self.log_dir / "best_model.pth")
                    self.log_msg(f"Model saved with loss: {best_loss:.8f}")

            ############################################################
            # Shut down if next checkpoint would exceed time limit
            ############################################################
            time_remaining = self.time_limit - (time.time() - self.start_time)
            time_needed = (
                self.avg_sec_per_checkpoint * 1.2
            )  # time needed for next checkpoint
            if time_remaining < time_needed:
                self.shutdown_flag = torch.tensor(
                    1, device=self.device
                )  # set flag to tell outer loop to shut down
                if self.ddp_enabled:
                    dist.all_reduce(self.shutdown_flag, op=dist.ReduceOp.SUM)
                self.log_msg(
                    "Summary: Next checkpoint would exceed time limit, shutting down"
                )
            if self.ddp_enabled:
                dist.barrier()

        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.global_rank == 0:
            self.wandb_logger.finish()
        if self.ddp_enabled:
            dist.destroy_process_group()


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
    else:
        raise ValueError(f"Optimizer {config['name']} not supported")

    return optimizer


def setup_ddp():
    dist.init_process_group()


@record
def main(
    config_path: Path,
    log_dir: Path | None,
    restart: bool,
    new_training: bool,
    checkpoint_name: str,
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

    if restart or new_training:
        checkpoint_dir = config["logging"]["log_dir"] / config["wandb"]["id"]

        # get subdir name to look for in the checkpoint dir
        if "subdir_name" in config["logging"]:
            subdir_name = config["logging"]["subdir_name"]
        else:
            subdir_name = "val_"

        checkpoint_path = find_checkpoint(
            checkpoint_dir, subdir_name, specific_checkpoint=checkpoint_name
        )
        if checkpoint_path is None:
            if global_rank == 0:
                print("No checkpoint found, starting from scratch")
        else:
            if global_rank == 0:
                print(f"Restarting from checkpoint {checkpoint_path}")
    else:
        checkpoint_path = None

    ####################################################################
    ########### Initialize trainer #####################################
    ####################################################################
    if world_size > 1:
        setup_ddp()

    trainer = Trainer(config, global_rank, local_rank, world_size)
    if checkpoint_path is not None:
        trainer.load_checkpoint(
            checkpoint_path=checkpoint_path,
            restart=restart,
            new_training=new_training,
        )
    trainer.save_config()
    trainer.train()


if __name__ == "__main__":
    ############################################################
    ########### Default arguments ##############################
    ############################################################

    default_config_path = Path(r"config.yaml")
    default_log_dir = Path("logs")
    default_sim_name = None
    default_data_dir = Path("data/datasets")
    default_time_limit = None
    default_restart = False
    default_new_training = False
    default_checkpoint_name = "last_checkpoint"
    default_best_model = False
    ############################################################
    ########### Parse arguments ################################
    ############################################################

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
        "--checkpoint_name",
        type=str,
        default=default_checkpoint_name,
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
    checkpoint_name = args.checkpoint_name
    best_model = args.best_model

    # For backward compatibility, if best_model is True, use it as checkpoint_name
    if best_model:
        checkpoint_name = "best_model"

    main(
        config_path=config_path,
        log_dir=log_dir,
        sim_name=sim_name,
        data_dir=data_dir,
        restart=restart,
        new_training=new_training,
        checkpoint_name=checkpoint_name,
        time_limit=time_limit,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
