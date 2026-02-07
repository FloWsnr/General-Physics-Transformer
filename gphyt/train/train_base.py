from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path
import time
import math

import torch
import torch.distributed as dist
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from gphyt.train.utils.logger import setup_logger
from gphyt.train.utils.wandb_logger import WandbLogger
from gphyt.train.utils.run_utils import (
    compute_metrics,
    reduce_all_losses,
)
from gphyt.train.utils.time_keeper import TimeKeeper
from gphyt.train.utils.checkpoint_utils import save_checkpoint
from gphyt.train.eval import Evaluator


@dataclass
class TrainingState:
    """State of the training process, changes every batch"""

    epoch: int = 0
    samples_trained: int = 0
    batches_trained: int = 0
    current_lr: float = 0.0
    batch_size: int = 0
    shutdown: torch.Tensor = torch.tensor(False)


class Trainer:
    """
    A comprehensive training framework for PyTorch models with distributed training support.

    This trainer handles the complete training lifecycle including model training, validation,
    checkpointing, logging, and time management. It supports distributed data parallel (DDP)
    training, automatic mixed precision (AMP), and integrates with Weights & Biases for
    experiment tracking.

    Parameters
    ----------
    model : torch.nn.Module or DDP or Any
        The PyTorch model to train. Can be a regular nn.Module or DDP-wrapped model.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer for parameter updates.
    criterion : torch.nn.Module
        Loss function used for training.
    lr_scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
        Learning rate scheduler for adaptive learning rates.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_dataloader : torch.utils.data.DataLoader
        DataLoader for validation data.
    scaler : GradScaler
        Gradient scaler for automatic mixed precision training.
    total_updates : int
        Total number of training updates/batches to perform.
    updates_per_epoch : int
        Number of updates to perform per epoch (before eval is done again).
    checkpoint_every_updates : int
        Frequency of checkpoint saves (in updates).
    eval_fraction : float
        Fraction of validation data to use during evaluation.
    epoch : int
        Current epoch (for resuming from checkpoints).
    batches_trained : int
        Batches already trained in previous runs (for resuming).
    samples_trained : int
        Samples already trained in previous runs (for resuming).
    output_dir : Path
        Directory path where checkpoints and logs will be saved.
    loss_fns : dict
        Dictionary of additional loss functions for metric computation.
        The criterion should be also included in this dictionary.
    max_grad_norm : float, optional
        Maximum gradient norm for gradient clipping. If None, no clipping is applied.
    amp : bool, optional
        Whether to use automatic mixed precision (AMP) for training, by default True.
    amp_precision : torch.dtype, optional
        Precision to use for AMP (e.g., torch.float16, torch.bfloat16), by default torch.bfloat16.
    time_limit : float, optional
        Time limit (in seconds) for training. If specified, training will stop
        gracefully before this limit is reached.
    wandb_logger : WandbLogger, optional
        Weights & Biases logger for experiment tracking.
    global_rank : int, default 0
        Global rank of current process in distributed training.
    local_rank : int, default 0
        Local rank of current process within a node.
    world_size : int, default 1
        Total number of processes in distributed training.

    Examples
    --------
    >>> trainer = Trainer(
    ...     model=my_model,
    ...     optimizer=optimizer,
    ...     criterion=nn.MSELoss(),
    ...     lr_scheduler=scheduler,
    ...     train_dataloader=train_loader,
    ...     val_dataloader=val_loader,
    ...     total_updates=10000,
    ...     updates_per_epoch=1000,
    ...     checkpoint_every_updates=500,
    ...     output_dir=Path("./outputs"),
    ...     loss_fns={"rmse": RMSE()}
    ... )
    >>> trainer.run()
    """

    def __init__(
        self,
        model: torch.nn.Module | DDP | Any,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        scaler: GradScaler,
        total_updates: int,
        updates_per_epoch: int,
        checkpoint_every_updates: int,
        eval_fraction: float,
        epoch: int,
        batches_trained: int,
        samples_trained: int,
        output_dir: Path,
        loss_fns: dict,
        amp: bool = True,
        amp_precision: torch.dtype = torch.bfloat16,
        max_grad_norm: Optional[float] = None,
        n_ar_steps: int = 1,
        time_limit: Optional[float] = None,
        wandb_logger: Optional[WandbLogger] = None,
        global_rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = setup_logger("gphyt", rank=global_rank)
        self.wandb_logger = wandb_logger
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.time_limit = time_limit
        self.total_updates = total_updates
        self.updates_per_epoch = updates_per_epoch
        self.checkpoint_every_updates = checkpoint_every_updates
        self.eval_fraction = eval_fraction

        self.time_keeper = TimeKeeper(time_limit=time_limit, global_rank=global_rank)

        self.loss_fns = loss_fns
        self.max_grad_norm = max_grad_norm
        self.n_ar_steps = n_ar_steps
        self.use_amp = amp
        self.amp_precision = amp_precision

        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.ddp_enabled = dist.is_initialized()
        self.scaler = scaler

        if train_dataloader.batch_size is not None:
            batch_size = train_dataloader.batch_size * world_size
        else:
            batch_size = 1 * world_size

        self.state = TrainingState(
            epoch=epoch,
            samples_trained=samples_trained,
            batches_trained=batches_trained,
            current_lr=self.optimizer.param_groups[0]["lr"],
            batch_size=batch_size,
            shutdown=torch.tensor(False, device=self.device),
        )

    def run(self) -> None:
        """Run training with error handling and cleanup."""
        try:
            self._run()
        except Exception as e:
            self.log_msg(f"Error during training: {e}")
            if self.wandb_logger is not None:
                self.wandb_logger.finish()
            if self.ddp_enabled:
                dist.destroy_process_group()
            raise e

    def _run(self):
        self.log_msg("Starting training")
        while self.state.batches_trained < self.total_updates:
            epoch_dir = self.output_dir / f"epoch_{self.state.epoch:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            ###################################################################################
            ######################## Update state #############################################
            ###################################################################################
            # Check if we should stop the update loop due to time constraints
            if self.time_keeper.should_stop_update_loop():
                self.log_msg("Stopping update loop due to time constraints")
                self.state.shutdown = torch.tensor(True, device=self.device)
                if self.ddp_enabled:
                    dist.all_reduce(self.state.shutdown, op=dist.ReduceOp.SUM)

            if self.state.shutdown:
                break
            self.log_msg(f"Starting epoch {self.state.epoch}")
            self.train_updates(n_updates=self.updates_per_epoch, epoch=self.state.epoch)

            # Save epoch checkpoint
            save_checkpoint(
                global_rank=self.global_rank,
                checkpoint_path=epoch_dir / "checkpoint.pt",
                model=self.model,
                optimizer=self.optimizer,
                samples_trained=self.state.samples_trained,
                batches_trained=self.state.batches_trained,
                epoch=self.state.epoch,
                grad_scaler=self.scaler,
                scheduler=self.lr_scheduler,
            )

            ###################################################################################
            ######################## Validate #################################################
            ###################################################################################
            # Check if we should skip validation due to time constraints
            if self.time_keeper.should_stop_validation():
                self.log_msg("Skipping validation due to time constraints")
                self.state.shutdown = torch.tensor(True, device=self.device)
                if self.ddp_enabled:
                    dist.all_reduce(self.state.shutdown, op=dist.ReduceOp.SUM)

            if self.state.shutdown:
                break
            self.validate(epoch=self.state.epoch)

            self.time_keeper.update_estimate(
                time.time() - self.time_keeper.time_start,
                "avg_sec_per_epoch",
                self.state.epoch,
            )
            self.state.epoch += 1

        save_checkpoint(
            global_rank=self.global_rank,
            checkpoint_path=self.output_dir / "latest.pt",
            model=self.model,
            optimizer=self.optimizer,
            samples_trained=self.state.samples_trained,
            batches_trained=self.state.batches_trained,
            epoch=self.state.epoch,
            grad_scaler=self.scaler,
            scheduler=self.lr_scheduler,
        )
        if self.wandb_logger is not None:
            self.wandb_logger.finish()
        if self.ddp_enabled:
            dist.destroy_process_group()
        self.log_msg("Training complete")

    def validate(self, epoch: int) -> None:
        # Clear CUDA cache before validation to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        epoch_dir = self.output_dir / f"epoch_{epoch:04d}"
        t_eval_start = time.time()
        evaluator = Evaluator(
            model=self.model,
            dataloader=self.val_dataloader,
            metrics=self.loss_fns,
            eval_dir=epoch_dir,
            eval_fraction=self.eval_fraction,
            n_ar_steps=self.n_ar_steps,
            amp=self.use_amp,
            amp_precision=self.amp_precision,
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            logger=self.logger,
        )
        eval_metrics = evaluator.eval()
        # images = evaluator.vis_predictions()

        ###################################################################################
        ######################## Logging ##################################################
        ###################################################################################
        # Update validation loop time
        val_duration = time.time() - t_eval_start
        self.time_keeper.update_estimate(
            val_duration, "avg_sec_per_val_loop", self.state.epoch
        )
        if self.wandb_logger is not None:
            log_state = {
                "samples_trained": self.state.samples_trained,
                "batches_trained": self.state.batches_trained,
                "epoch": epoch,
                "avg_sec_per_val_loop": self.time_keeper.estimates.avg_sec_per_val_loop,
            }
            self.wandb_logger.log(log_state, folder="eval", commit=False)
            self.wandb_logger.log(eval_metrics, folder="eval", commit=True)
            # self.wandb_logger.log(images, folder="eval", commit=True)

    def train_updates(self, n_updates: int, epoch: int) -> None:
        t_update_start = time.time()
        n_updates = min(
            n_updates,
            len(self.train_dataloader),
            self.total_updates - self.state.batches_trained,
        )
        self.model.train()
        if self.ddp_enabled:
            self.train_dataloader.sampler.set_epoch(epoch)

        log_interval = max(1, 10 ** math.floor(math.log10(max(1, n_updates // 100))))

        for i, data in enumerate(self.train_dataloader):
            x = data[0]
            target = data[1]
            x = x.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            if self.n_ar_steps > 1:
                # Autoregressive training loop
                ar_steps = target.shape[1]  # number of output timesteps from data
                for _ar_step in range(ar_steps):
                    if _ar_step == 0:
                        x_ar = x
                    else:
                        x_ar = torch.cat((x_ar[:, 1:, ...], output.detach()), dim=1)

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_precision,
                        enabled=self.use_amp,
                    ):
                        output = self.model(x_ar)
                        step_target = target[:, _ar_step, ...].unsqueeze(1)
                        step_loss = self.criterion(output, step_target) / ar_steps

                    self.scaler.scale(step_loss).backward()

                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                current_metrics = compute_metrics(output, step_target, self.loss_fns)
            else:
                # Standard single-step forward pass
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_precision,
                    enabled=self.use_amp,
                ):
                    output = self.model(x)
                    raw_loss = self.criterion(output, target)

                self.scaler.scale(raw_loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                current_metrics = compute_metrics(output, target, self.loss_fns)
            current_metrics = reduce_all_losses(current_metrics)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            ###################################################################################
            ######################## Update state #############################################
            ###################################################################################
            self.state.samples_trained += self.state.batch_size
            self.state.batches_trained += 1
            self.state.current_lr = self.optimizer.param_groups[0]["lr"]

            # log to wandb
            if self.wandb_logger is not None:
                log_state = {
                    "samples_trained": self.state.samples_trained,
                    "batches_trained": self.state.batches_trained,
                    "epoch": self.state.epoch,
                    "learning_rate": self.state.current_lr,
                }
                self.wandb_logger.log(log_state, folder="train", commit=False)
                self.wandb_logger.log(current_metrics, folder="train", commit=True)

            if (i + 1) % log_interval == 0 or i == n_updates - 1:
                self.log_msg(
                    f"Epoch {self.state.epoch:03d} | Update {i + 1}/{n_updates} | "
                    + " | ".join([f"{k}: {v:.4f}" for k, v in current_metrics.items()])
                )

            ###################################################################################
            ######################## Checkpointing ############################################
            ###################################################################################
            next_checkpoint = (
                self.state.batches_trained // self.checkpoint_every_updates + 1
            ) * self.checkpoint_every_updates

            if self.state.batches_trained >= next_checkpoint - 1:
                save_checkpoint(
                    global_rank=self.global_rank,
                    checkpoint_path=self.output_dir / "latest.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    samples_trained=self.state.samples_trained,
                    batches_trained=self.state.batches_trained,
                    epoch=self.state.epoch,
                    grad_scaler=self.scaler,
                    scheduler=self.lr_scheduler,
                )
                self.log_msg("Saved latest checkpoint")

            if i >= n_updates - 1:
                break

        # Update training loop time
        update_duration = time.time() - t_update_start
        self.time_keeper.update_estimate(
            update_duration, "avg_sec_per_update_loop", self.state.epoch
        )

    def log_msg(self, msg: str):
        """Log a message."""
        self.logger.info(f"{msg}")
