"""
Training script for transformer models.

Author: Florian Wiesner
Date: 2025-04-07
"""

import os
from pathlib import Path
import time
import argparse

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
from dadaptation import DAdaptAdam

from lpfm.data.dataset_utils import get_dataloader
from lpfm.model.transformer.model import get_model
from lpfm.utils.train_vis import log_predictions_wandb, visualize_predictions
from lpfm.utils.logger import get_logger
from lpfm.model.transformer.nmse_loss import NMSELoss
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
        self.start_epoch_time = 0
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
        if self.config["training"]["compile"]:
            self.log_msg("Compiling model")
            self.model = torch.compile(self.model)
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
        self.samples_trained = 0
        self.epoch = 1

        self.batch_size = self.config["training"]["batch_size"]
        self.total_epochs = self.config["training"]["epochs"]

        # NOTE: These are per GPU worker
        self.train_batches_per_epoch = len(self.train_loader)
        self.val_batches_per_epoch = len(self.val_loader)
        self.train_samples_per_epoch = self.train_batches_per_epoch * self.batch_size

        # NOTE: This is per GPU worker
        self.total_samples = self.train_samples_per_epoch * self.total_epochs

        # NOTE: This is across all GPU workers
        self.sum_total_samples = self.total_samples * self.world_size
        self.sum_train_batches = self.train_batches_per_epoch * self.world_size
        self.sum_val_batches = self.val_batches_per_epoch * self.world_size
        self.sum_train_samples = self.train_samples_per_epoch * self.world_size

        self.log_msg(f"Training for {self.total_epochs} epochs")
        self.log_msg(f"Training on {self.sum_train_batches} batches per epoch")
        self.log_msg(f"Training for {self.sum_train_samples} samples per epoch")
        self.log_msg(f"Training for {self.sum_total_samples} total samples")
        self.log_msg(f"Validating on {self.sum_val_batches} batches per epoch")

        if self.global_rank == 0:
            self.wandb_run.config.update(
                {
                    "training/total_samples": self.sum_total_samples,
                    "training/train_samples_per_epoch": self.sum_train_samples,
                    "training/train_batches_per_epoch": self.sum_train_batches,
                },
                allow_val_change=True,
            )

        ################################################################
        ########### Initialize loss function and optimizer ###########
        ################################################################
        opt_config = self.config["training"]["optimizer"]
        if self.config["training"]["criterion"] == "MSE":
            self.criterion = nn.MSELoss()
        elif self.config["training"]["criterion"] == "NMSE":
            self.criterion = NMSELoss()
        elif self.config["training"]["criterion"] == "MAE":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(
                f"Criterion {self.config['training']['criterion']} not supported"
            )
        self.optimizer = get_optimizer(self.model, opt_config)
        self.grad_scaler = GradScaler()
        ################################################################
        ########### Initialize learning rate scheduler ################
        ################################################################
        if "lr_scheduler" in self.config["training"]:
            lrs_config = self.config["training"]["lr_scheduler"]
            self.scheduler = get_lr_scheduler(
                optimizer=self.optimizer,
                lrs_config=lrs_config,
                total_epochs=self.total_epochs,
                train_batches_per_epoch=self.train_batches_per_epoch,
            )
        else:
            self.scheduler = None

        ################################################################
        ########### Watch model ########################################
        ################################################################
        if self.global_rank == 0:
            log_interval = self.config["wandb"]["log_interval"]
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
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"] + 1
        self.samples_trained = checkpoint["samples_trained"]
        self.log_msg(
            f"Restarting training from epoch {self.epoch} with {self.samples_trained} samples trained"
        )

    def save_config(self):
        """Save the config to the log directory."""
        if self.global_rank == 0:
            with open(self.log_dir / "config.yaml", "w") as f:
                yaml.dump(self.config, f)

    def train_epoch(self) -> float:
        """Train the model for one epoch.

        Parameters
        ----------
        Returns
        -------
        float
            Average training loss for the epoch
        """
        self.start_epoch_time = time.time()
        self.model.train()

        if self.ddp_enabled:
            self.train_loader.sampler.set_epoch(self.epoch)

        acc_train_loss = 0

        total_batches = self.train_batches_per_epoch
        for batch_idx, batch in enumerate(self.train_loader, start=1):
            x, target = batch
            x = x.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                output = self.model(x)
                loss = self.criterion(output, target)
            acc_train_loss += loss.item()

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # Clip gradients to norm 1
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config["training"]["grad_clip"]
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            ############################################################
            # Step learning rate scheduler #############################
            ############################################################
            if self.scheduler is not None:
                self.scheduler.step()

            ############################################################
            # Log training progress ####################################
            ############################################################

            self.samples_trained += self.batch_size

            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()[0]
            elif isinstance(self.optimizer, DAdaptAdam):
                lr = (
                    self.optimizer.param_groups[0]["lr"]
                    * self.optimizer.param_groups[0]["d"]
                )
            else:
                lr = self.optimizer.param_groups[0]["lr"]

            world_batch_idx = batch_idx * self.world_size
            world_samples_trained = self.samples_trained * self.world_size
            self.log_msg(
                "Training: "
                f"Epoch {self.epoch}/{self.total_epochs}, "
                f"Batch (all GPUs): {world_batch_idx}/{self.sum_train_batches}, "
                f"Samples (all GPUs): {world_samples_trained}/{self.sum_total_samples}, "
                f"LR: {lr:.8f}"
            )

            ############################################################
            # Log to wandb #############################################
            ############################################################
            log_interval = self.config["wandb"]["log_interval"]
            if batch_idx % log_interval == 0:
                self.log_msg("Training: Collecting loss across all GPUs")

                if self.ddp_enabled:
                    # accumulate loss for all gpu workers
                    acc_train_loss_tensor = torch.tensor(
                        acc_train_loss, device=self.device
                    )
                    dist.all_reduce(acc_train_loss_tensor, op=dist.ReduceOp.AVG)
                    sum_acc_train_loss = acc_train_loss_tensor.item() / world_batch_idx
                else:
                    sum_acc_train_loss = acc_train_loss / world_batch_idx

                total_b_idx = (
                    world_batch_idx + (self.epoch - 1) * self.sum_train_batches
                )
                self.log_msg(f"Training: Accumulated loss: {sum_acc_train_loss:.8f}")
                if self.global_rank == 0:
                    self.wandb_run.log(
                        {
                            "training/num_batches": total_b_idx,
                            "training/num_samples": world_samples_trained,
                            "training/acc_batch_loss": sum_acc_train_loss,
                            "training/learning_rate": lr,
                        }
                    )

            ############################################################
            # Check time limit #########################################
            ############################################################
            if self.time_limit is not None:
                if time.time() - self.start_time > self.time_limit:
                    self.log_msg("Time limit reached, stopping training")
                    return acc_train_loss / total_batches

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

        return acc_train_loss / total_batches

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0

        if self.ddp_enabled:
            self.val_loader.sampler.set_epoch(self.epoch)

        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.val_loader, start=1):
                x, target = batch
                x = x.to(self.device)
                target = target.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    output = self.model(x)
                    loss = self.criterion(output, target)

                val_loss += loss.item()

                world_batch_idx = batch_idx * self.world_size
                self.log_msg(
                    f"Validation: Epoch {self.epoch}/{self.total_epochs}, Batch {world_batch_idx}/{self.sum_val_batches}, "
                )

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

        return val_loss / self.val_batches_per_epoch

    def train(self):
        """Train the model."""
        best_loss = float("inf")
        epochs = range(self.epoch, self.total_epochs + 1)
        for epoch in epochs:
            self.epoch = epoch
            self.epoch_dir = self.log_dir / f"epoch_{epoch:04d}"
            self.epoch_dir.mkdir(parents=True, exist_ok=True)
            ######################################################################
            ########### Training ###############################################
            ######################################################################
            self.log_msg(f"Training: Training epoch {epoch}")
            train_loss = self.train_epoch()

            # Average training loss across all GPUs if DDP is enabled
            if self.ddp_enabled:
                self.log_msg("Training: Reducing training loss across all GPUs")
                train_loss_tensor = torch.tensor(train_loss, device=self.device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = train_loss_tensor.item()

            self.log_msg(f"Training: Epoch {epoch}, Training loss: {train_loss:.8f}")

            ######################################################################
            ########### Validation ###############################################
            ######################################################################
            self.log_msg(f"Validation: Validating epoch {epoch}")
            val_loss = self.validate()

            # Average validation loss across all GPUs if DDP is enabled
            if self.ddp_enabled:
                self.log_msg("Validation: Reducing validation loss across all GPUs")
                val_loss_tensor = torch.tensor(val_loss, device=self.device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss = val_loss_tensor.item()

            self.log_msg(f"Validation: Epoch {epoch}, Validation loss: {val_loss:.8f}")

            ############################################################
            # Calculate average time per epoch #########################
            ############################################################
            duration = time.time() - self.start_epoch_time
            self.log_msg(f"Summary: Epoch {epoch} took {duration / 60:.2f} minutes")
            self.avg_sec_per_epoch = (
                self.avg_sec_per_epoch * (self.epoch - 1) + duration
            ) / self.epoch
            self.log_msg(
                f"Summary: Average time per epoch: {self.avg_sec_per_epoch / 60:.2f} minutes"
            )
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
                        "epoch": epoch,
                        "training/epoch_loss": train_loss,
                        "validation/epoch_loss": val_loss,
                        "training/minutes_per_epoch": duration / 60,
                        "training/avg_minutes_per_epoch": self.avg_sec_per_epoch / 60,
                        "training/projected_minutes_remaining": proj_time_remaining
                        / 60,
                    }
                )

            ######################################################################
            ########### Save best model #########################################
            ######################################################################
            if self.global_rank == 0:
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.model.state_dict(), self.log_dir / "best_model.pth")
                    self.log_msg(f"Model saved with loss: {best_loss:.8f}")

            ######################################################################
            ########### Save checkpoint ########################################
            ######################################################################
            if self.global_rank == 0:
                checkpoint = {
                    "epoch": epoch,
                    "samples_trained": self.samples_trained,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": self.config,
                }
                if self.scheduler is not None:
                    checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
                else:
                    checkpoint["scheduler_state_dict"] = None
                torch.save(
                    checkpoint,
                    self.epoch_dir / "checkpoint.pth",
                )
                self.log_msg(
                    f"Summary: Checkpoint saved to {self.epoch_dir / 'checkpoint.pth'}"
                )
            if self.ddp_enabled:
                dist.barrier()

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

    scheduler_names = list(lrs_config["schedulers"].keys())
    if len(scheduler_names) == 1:
        lrs_lin = lrs_config["schedulers"]["LinearLR"]
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=lrs_lin["start_factor"],
            end_factor=lrs_lin["end_factor"],
            total_iters=lrs_lin["samples"],
        )

    elif len(scheduler_names) == 2:
        # remove LinearLR from scheduler_names
        scheduler_names.remove("LinearLR")
        lrs_lin = lrs_config["schedulers"]["LinearLR"]

        lrs1_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=lrs_lin["start_factor"],
            end_factor=lrs_lin["end_factor"],
            total_iters=lrs_lin["samples"],
        )

        lrs2_name = scheduler_names[0]
        lrs2 = lrs_config["schedulers"][lrs2_name]

        if lrs2_name == "CosineAnnealingWarmRestarts":
            T_0 = train_batches_per_epoch * lrs2["T_0"]
            # if T_max is -1, use all remaining epochs
            if lrs2["T_max"] == -1:
                T_max = total_epochs * train_batches_per_epoch - lrs_lin["samples"]
            else:
                T_max = train_batches_per_epoch * lrs2["T_max"]

            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=lrs2["T_mult"], eta_min=float(lrs2["min_lr"])
            )
        elif lrs2_name == "CosineAnnealingLR":
            # if T_max is -1, use all remaining epochs
            if lrs2["T_max"] == -1:
                T_max = total_epochs * train_batches_per_epoch - lrs_lin["samples"]
            else:
                T_max = train_batches_per_epoch * lrs2["T_max"]

            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=float(lrs2["min_lr"])
            )
        else:
            raise ValueError(f"Scheduler {lrs2_name} not supported")

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[lrs1_scheduler, cosine_scheduler],
            milestones=[lrs_lin["samples"]],
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
    elif config["name"] == "DAdaptAdam":
        optimizer = DAdaptAdam(
            model.parameters(),
            lr=1,
            betas=config["betas"],
            weight_decay=config["weight_decay"],
        )
    elif config["name"] == "DAdaptAdamW":
        optimizer = DAdaptAdam(
            model.parameters(),
            lr=1,
            betas=config["betas"],
            weight_decay=config["weight_decay"],
            decouple=True,
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
    dist.init_process_group(backend="nccl")


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

    default_config_path = Path("lpfm/run/config.yaml")
    default_log_dir = Path("logs")
    default_sim_name = None
    default_data_dir = Path("data/datasets")
    default_time_limit = None
    default_restart = False

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
