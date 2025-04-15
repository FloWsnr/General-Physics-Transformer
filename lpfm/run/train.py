"""
Training script for transformer models.

Author: Florian Wiesner
Date: 2025-04-07
"""

from pathlib import Path
from datetime import datetime

import wandb
import wandb.wandb_run
import yaml
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from dadaptation import DAdaptAdam

from lpfm.data.dataset_utils import get_dataloader
from lpfm.model.transformer.model import get_model
from lpfm.utils.train_vis import log_predictions_wandb, visualize_predictions
from lpfm.utils.logger import get_logger


class NMSELoss(nn.Module):
    """Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)
        which is time, height, width

    Attributes
    ----------
    dims : tuple
        Dimensions to reduce over
    """

    def __init__(self, dims=(1, 2, 3)):
        """Initialize NMSE loss.

        Parameters
        ----------
        dims : tuple, optional
            Dimensions to reduce over, by default (1, 2, 3)
        """
        super().__init__()
        self.dims = dims

    def forward(self, pred, target):
        """Calculate the normalized mean square error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Normalized MSE loss
        """
        # Calculate residuals
        residuals = pred - target
        # Normalize by mean squared target values (with small epsilon)
        target_norm = target.pow(2).mean(self.dims, keepdim=True) + 1e-8
        # Calculate normalized MSE
        nmse = residuals.pow(2).mean(self.dims, keepdim=True) / target_norm
        # Return mean over batch dimensions
        return nmse.mean()


class Trainer:
    def __init__(self, config: Path | dict):
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)

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
        # Add date and time to wandb id
        self.config["wandb"]["id"] = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{self.config['wandb']['id']}"
        )

        self.log_dir = (
            Path(self.config["logging"]["log_dir"]) / self.config["wandb"]["id"]
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_run = login_wandb(self.config)

        ################################################################
        ########### Set random seeds ##################################
        ################################################################
        torch.manual_seed(self.config["training"]["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config["training"]["seed"])

        ################################################################
        ########### Initialize model ##################################
        ################################################################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.model = get_model(model_config=self.config["model"])

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model size: {total_params / 1e6:.2f}M parameters")
        self.logger.info(f"Model architecture: {self.model}")

        self.wandb_run.config.update({"model/model_size [M]": total_params / 1e6})

        # print the model architecture
        self.model.to(self.device)

        ################################################################
        ########### Initialize data loaders ##########################
        ################################################################
        self.train_loader = get_dataloader(
            self.config["data"], self.config["training"], split="train"
        )
        self.val_loader = get_dataloader(
            self.config["data"], self.config["training"], split="val"
        )
        ################################################################
        ########### Initialize training parameters ##################
        ################################################################
        self.samples_trained = 0
        self.epoch = 1

        self.batch_size = self.config["training"]["batch_size"]
        self.total_epochs = self.config["training"]["epochs"]

        self.train_batches_per_epoch = len(self.train_loader)
        self.val_batches_per_epoch = len(self.val_loader)
        self.train_samples_per_epoch = self.train_batches_per_epoch * self.batch_size

        self.total_samples = self.train_samples_per_epoch * self.total_epochs

        self.logger.info(f"Training for {self.total_epochs} epochs")
        self.logger.info(
            f"Training on {self.train_batches_per_epoch} batches per epoch"
        )
        self.logger.info(
            f"Validating on {self.val_batches_per_epoch} batches per epoch"
        )
        self.logger.info(
            f"Training for {self.train_samples_per_epoch} samples per epoch"
        )
        self.logger.info(f"Training for {self.total_samples} total samples")

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

        ################################################################
        ########### Initialize learning rate scheduler ################
        ################################################################
        if "lr_scheduler" in self.config["training"]:
            lrs_config = self.config["training"]["lr_scheduler"]
            self.scheduler = get_lr_scheduler(
                self.optimizer, lrs_config, self.train_batches_per_epoch
            )
        else:
            self.scheduler = None

        ################################################################
        ########### Watch model ########################################
        ################################################################
        log_interval = self.config["wandb"]["log_interval"]
        self.wandb_run.watch(
            self.model,
            criterion=self.criterion,
            log=self.config["wandb"]["log_model"],
            log_freq=log_interval,
        )

    def restart_training(self, checkpoint_path: Path):
        """Restart training from a checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.samples_trained = checkpoint["samples_trained"]
        self.logger.info(
            f"Restarting training from epoch {self.epoch} with {self.samples_trained} samples trained"
        )

    def save_config(self):
        """Save the config to the log directory."""
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
        self.model.train()
        acc_train_loss = 0

        total_batches = self.train_batches_per_epoch
        for batch_idx, batch in enumerate(self.train_loader):
            x, target = batch
            x = x.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, target)
            loss.backward()

            # Clip gradients to norm 1
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config["training"]["grad_clip"]
            )
            self.optimizer.step()

            acc_train_loss += loss.item()

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

            self.logger.info(
                f"Epoch {self.epoch}/{self.total_epochs}, Batch {batch_idx}/{total_batches}, "
                f"Loss: {loss.item():.8f}, Acc. Loss: {acc_train_loss / (batch_idx + 1):.8f}, LR: {lr:.8f}, "
                f"Samples: {self.samples_trained}/{self.total_samples}"
            )

            ############################################################
            # Log to wandb #############################################
            ############################################################
            if batch_idx % self.config["wandb"]["log_interval"] == 0:
                total_b_idx = (
                    batch_idx + (self.epoch - 1) * self.train_batches_per_epoch
                )

                self.wandb_run.log(
                    {
                        "training/num_batches": total_b_idx,
                        "training/num_samples": self.samples_trained,
                        "training/acc_batch_loss": acc_train_loss / (batch_idx + 1),
                        "training/batch_loss": loss.item(),
                        "training/learning_rate": lr,
                    }
                )

        ############################################################
        # Visualize predictions ####################################
        ############################################################
        vis_path = self.epoch_dir / "train.png"
        visualize_predictions(vis_path, x, output, target, svg=True)
        log_predictions_wandb(
            run=self.wandb_run,
            image_path=vis_path.parent,
            name_prefix=f"epoch_{self.epoch}",
        )

        return acc_train_loss / total_batches

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0

        total_batches = self.val_batches_per_epoch
        num_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                x, target = batch
                x = x.to(self.device)
                target = target.to(self.device)

                output = self.model(x)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                num_batches += 1

                self.logger.info(
                    f"Epoch {self.epoch}/{self.total_epochs}, Batch {batch_idx}/{total_batches}, "
                    f"Loss: {loss.item():.8f}, Acc. Loss: {val_loss / num_batches:.8f}"
                )

        # Visualize predictions
        vis_path = self.epoch_dir / "val.png"
        visualize_predictions(vis_path, x, output, target, svg=True)
        log_predictions_wandb(
            run=self.wandb_run,
            image_path=vis_path.parent,
            name_prefix=f"epoch_{self.epoch}",
        )

        return val_loss / total_batches

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
            self.logger.info(f"Training epoch {epoch}")
            train_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch}, Training loss: {train_loss:.8f}")

            ######################################################################
            ########### Validation ###############################################
            ######################################################################
            self.logger.info(f"Validating epoch {epoch}")
            val_loss = self.validate()
            self.logger.info(f"Epoch {epoch}, Validation loss: {val_loss:.8f}")

            ######################################################################
            ########### Wandb logging ###########################################
            ######################################################################
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "training/epoch_loss": train_loss,
                    "validation/epoch_loss": val_loss,
                }
            )

            ######################################################################
            ########### Save best model #########################################
            ######################################################################
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.log_dir / "best_model.pth")
                self.logger.info(f"Model saved with loss: {best_loss:.8f}")

            ######################################################################
            ########### Save checkpoint ########################################
            ######################################################################
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
            torch.save(
                checkpoint,
                self.epoch_dir / "checkpoint.pth",
            )

        self.wandb_run.finish()


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: dict, train_batches_per_epoch: int
) -> optim.lr_scheduler.SequentialLR:
    """Create a learning rate scheduler.
    Options are only linear warmup or linear warmup followed by cosine annealing.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for training
    config : dict
        Configuration dictionary for the learning rate scheduler
    train_batches_per_epoch : int
        Number of training batches per epoch

    Returns
    -------
    optim.lr_scheduler.SequentialLR
        Learning rate scheduler
    """

    num_schedulers = len(config["schedulers"])
    if num_schedulers == 1:
        lrs_lin = config["schedulers"]["LinearLR"]
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=lrs_lin["start_factor"],
            end_factor=lrs_lin["end_factor"],
            total_iters=train_batches_per_epoch * lrs_lin["epochs"],
        )

    elif num_schedulers == 2:
        lrs_lin = config["schedulers"]["LinearLR"]
        t_steps = train_batches_per_epoch * lrs_lin["epochs"]

        lrs1_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=lrs_lin["start_factor"],
            end_factor=lrs_lin["end_factor"],
            total_iters=t_steps,
        )

        lrs2_name = list(config["schedulers"].keys())[1]
        lrs2 = config["schedulers"][lrs2_name]

        if lrs2_name == "CosineAnnealingWarmRestarts":
            T_0 = train_batches_per_epoch * lrs2["T_0"]
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=lrs2["T_mult"], eta_min=float(lrs2["min_lr"])
            )
        elif lrs2_name == "CosineAnnealingLR":
            T_max = train_batches_per_epoch * lrs2["T_max"]
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=float(lrs2["min_lr"])
            )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[lrs1_scheduler, cosine_scheduler],
            milestones=[t_steps],
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
    )
    return run


def main(config_path: Path):
    """Main training function."""
    load_dotenv()
    trainer = Trainer(config_path)
    trainer.save_config()
    trainer.train()


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train a transformer model for physics simulations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/zsa8rk/Coding/Large-Physics-Foundation-Model/lpfm/run/config.yaml",
        help="Path to the configuration file",
    )

    # Parse arguments
    args = parser.parse_args()
    config_path = Path(args.config)
    main(config_path)
