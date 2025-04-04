"""
Training script for convolutional neural network models.

Author: Florian Wiesner
Date: 2025-03-10
"""

from pathlib import Path

import wandb.wandb_run
import yaml
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from dadaptation import DAdaptAdam

from metaparc.data.dataset_utils import get_dataloader
from metaparc.model.transformer.model import get_model
from metaparc.utils.train_vis import visualize_predictions
from metaparc.utils.logger import get_logger


class Trainer:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

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

        self.wandb_run = login_wandb(self.config)

        ################################################################
        ########### Initialize model ##################################
        ################################################################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.model = get_model(model_config=self.config["model"])

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model size: {total_params / 1e6:.2f}M parameters")
        self.logger.info(f"Model architecture: {self.model}")
        self.model.to(self.device)

        ################################################################
        ########### Initialize data loaders ##########################
        ################################################################
        self.train_loader = get_dataloader(
            self.config["data"], self.config["training"], split="train"
        )
        self.epoch_size = len(self.train_loader)
        self.val_loader = get_dataloader(
            self.config["data"], self.config["training"], split="val"
        )
        self.logger.info(f"Training for {self.config['training']['epochs']} epochs")
        self.logger.info(f"Training on {len(self.train_loader)} samples per epoch")
        self.logger.info(f"Validating on {len(self.val_loader)} samples per epoch")

        ################################################################
        ########### Initialize loss function and optimizer ###########
        ################################################################
        opt_config = self.config["training"]["optimizer"]
        self.criterion = nn.MSELoss()
        self.optimizer = get_optimizer(self.model, opt_config)

        ################################################################
        ########### Initialize learning rate scheduler ################
        ################################################################
        if "lr_scheduler" in self.config["training"]:
            lrs_config = self.config["training"]["lr_scheduler"]
            self.scheduler = get_lr_scheduler(
                self.optimizer, lrs_config, self.epoch_size
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
        train_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            target = torch.cat((x[:, 1:, :, :, :], y), dim=1)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, target)
            loss.backward()

            # Clip gradients to norm 1
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config["training"]["grad_clip"]
            )
            self.optimizer.step()

            train_loss += loss.item()

            # Step learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"\t Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.6f}"
            )

            # Log to wandb
            if batch_idx % self.config["wandb"]["log_interval"] == 0:
                self.wandb_run.log(
                    {
                        "training/batch_idx": batch_idx,
                        "training/batch_loss": loss,
                        "training/learning_rate": lr,
                    }
                )

        return train_loss / len(self.train_loader)

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                target = torch.cat((x[:, 1:, :, :, :], y), dim=1)

                output = self.model(x)
                loss = self.criterion(output, target)

                val_loss += loss.item()

                self.logger.info(f"\t Batch {batch_idx}, Loss: {loss.item():.4f}")

                # Log to wandb
                if batch_idx % self.config["wandb"]["log_interval"] == 0:
                    self.wandb_run.log(
                        {
                            "validation/batch_idx": batch_idx,
                            "validation/batch_loss": loss,
                        }
                    )

        # Visualize predictions
        visualize_predictions(
            save_path=self.log_dir / f"epoch_{self.epoch}_vis.png",
            predictions=output,
            targets=target,
        )

        return val_loss / len(self.val_loader)

    def train(self):
        """Train the model."""
        best_loss = float("inf")
        for epoch in range(1, self.config["training"]["epochs"] + 1):
            self.epoch = epoch
            ######################################################################
            ########### Training ###############################################
            ######################################################################
            self.logger.info(f"Training epoch {epoch}")
            train_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch}, Training loss: {train_loss:.4f}")
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "training/epoch_loss": train_loss,
                }
            )
            ######################################################################
            ########### Validation ###############################################
            ######################################################################
            self.logger.info(f"Validating epoch {epoch}")
            val_loss = self.validate()
            self.logger.info(f"Epoch {epoch}, Validation loss: {val_loss:.4f}")
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "validation/epoch_loss": val_loss,
                }
            )
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.log_dir / "best_model.pth")
                self.logger.info(f"Model saved with loss: {best_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            torch.save(checkpoint, self.log_dir / f"checkpoint_epoch_{epoch}.pth")

        self.wandb_run.finish()


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: dict, epoch_size: int
) -> optim.lr_scheduler.SequentialLR:
    """Create a learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for training
    config : dict
        Configuration dictionary for the learning rate scheduler
    epoch_size : int
        Number of training steps per epoch

    Returns
    -------
    optim.lr_scheduler.SequentialLR
        Learning rate scheduler
    """

    lrs1 = config["schedulers"][0]
    lrs2 = config["schedulers"][1]

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=lrs1["start_factor"],
        end_factor=lrs1["end_factor"],
        total_iters=lrs1["total_iters"],
    )

    if lrs2["name"] == "CosineAnnealingWarmRestarts":
        t_steps = epoch_size * lrs2["T_0"]
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_steps, eta_min=lrs2["min_lr"]
        )
    elif lrs2["name"] == "CosineAnnealingLR":
        t_steps = epoch_size * lrs2["T_max"]
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_steps, eta_min=lrs2["min_lr"]
        )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[lrs1["total_iters"]],
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
        optimizer = optim.Adam(
            model.parameters(), lr=config["learning_rate"], betas=betas
        )
    elif config["name"] == "AdamW":
        weight_decay = config["weight_decay"]
        betas = config["betas"]
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
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
    wandb.login(key=config["wandb"]["api_key"])
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config,
        id=config["wandb"]["id"],
        tags=config["wandb"]["tags"],
    )
    return run


def main():
    """Main training function."""
    config_path = Path("/Users/zsa8rk/Coding/MetaPARC/metaparc/run/config.yaml")
    trainer = Trainer(config_path)
    trainer.train()


if __name__ == "__main__":
    main()
