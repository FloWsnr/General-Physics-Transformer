"""
Training script for transformer models.

Author: Florian Wiesner
Date: 2025-04-07
"""

from pathlib import Path

import wandb
import wandb.wandb_run
import yaml
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from dadaptation import DAdaptAdam

from metaparc.data.dataset_utils import get_dataloader

from metaparc.utils.train_vis import log_predictions_wandb, visualize_predictions
from metaparc.utils.logger import get_logger

from metaparc.model.ax_vit.avit import build_avit, AViTParams

params = AViTParams(
    patch_size=(16, 16),
    embed_dim=384,
    processor_blocks=4,
    n_states=5,
    block_type="axial",
    time_type="attention",
    num_heads=4,
    bias_type="rel",
    gradient_checkpointing=False,
)


class Trainer:
    def __init__(self, config: Path | dict):
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)

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

        self.wandb_run = login_wandb(self.config)

        ################################################################
        ########### Initialize model ##################################
        ################################################################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.model = build_avit(model_config=self.config["model"])

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
        self.criterion = nn.MSELoss()
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
        train_loss = 0

        total_batches = self.train_batches_per_epoch
        for batch_idx, batch in enumerate(self.train_loader):
            x, y = batch
            b, t, h, w, c = x.shape
            x = x.to(self.device)
            y = y.to(self.device)

            # rearrange the input and target to the correct shape for avit
            x = rearrange(x, "b t h w c -> t b c h w")
            y = rearrange(y, "b t h w c -> t b c h w")

            target = y.squeeze(0)
            labels = [[0, 1, 2, 3, 4] for _ in range(b)]
            labels = torch.tensor(labels).to(self.device)
            bcs = torch.tensor([[0, 0]]).to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x, labels, bcs)
            loss = self.criterion(output, target)
            loss.backward()

            # Clip gradients to norm 1
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config["training"]["grad_clip"]
            )
            self.optimizer.step()

            train_loss += loss.item()

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
                f"Loss: {loss.item():.6f}, Acc. Loss: {train_loss / (batch_idx + 1):.6f}, LR: {lr:.6f}, "
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
                        "training/samples_trained": self.samples_trained,
                        "training/batch_idx": total_b_idx,
                        "training/acc_batch_loss": train_loss / (batch_idx + 1),
                        "training/learning_rate": lr,
                    }
                )

        ############################################################
        # Visualize predictions ####################################
        ############################################################
        vis_path = self.log_dir / f"epoch_{self.epoch}" / "train"

        # rearrange the input and target to the correct shape for visualization
        x = rearrange(x, "t b c h w -> b t h w c")
        y = rearrange(y, "t b c h w -> b t h w c")

        visualize_predictions(vis_path, x, output, target)
        log_predictions_wandb(
            run=self.wandb_run,
            image_path=vis_path,
            name_prefix=f"epoch_{self.epoch}_train",
        )

        return train_loss / total_batches

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0

        total_batches = self.val_batches_per_epoch
        num_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                target = x
                # target = torch.cat((x[:, 1:, :, :, :], y), dim=1)

                output = self.model(x)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                num_batches += 1

                self.logger.info(
                    f"Epoch {self.epoch}/{self.total_epochs}, Batch {batch_idx}/{total_batches}, "
                    f"Loss: {loss.item():.6f}, Acc. Loss: {val_loss / num_batches:.6f}"
                )

        # Visualize predictions
        vis_path = self.log_dir / f"epoch_{self.epoch}" / "val"
        visualize_predictions(vis_path, x, output, target)
        log_predictions_wandb(
            run=self.wandb_run,
            image_path=vis_path,
            name_prefix=f"epoch_{self.epoch}_val",
        )

        return val_loss / total_batches

    def train(self):
        """Train the model."""
        best_loss = float("inf")
        self.samples_trained = 0
        for epoch in range(1, self.total_epochs + 1):
            self.epoch = epoch
            ######################################################################
            ########### Training ###############################################
            ######################################################################
            self.logger.info(f"Training epoch {epoch}")
            train_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch}, Training loss: {train_loss:.4f}")

            ######################################################################
            ########### Validation ###############################################
            ######################################################################
            self.logger.info(f"Validating epoch {epoch}")
            val_loss = self.validate()
            self.logger.info(f"Epoch {epoch}, Validation loss: {val_loss:.4f}")

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
                self.logger.info(f"Model saved with loss: {best_loss:.4f}")

            ######################################################################
            ########### Save checkpoint ########################################
            ######################################################################
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
    optimizer: torch.optim.Optimizer, config: dict, train_batches_per_epoch: int
) -> optim.lr_scheduler.SequentialLR:
    """Create a learning rate scheduler.

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

    lrs1 = config["schedulers"][0]
    lrs2 = config["schedulers"][1]

    total_iters = train_batches_per_epoch * lrs1["epochs"]
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=lrs1["start_factor"],
        end_factor=lrs1["end_factor"],
        total_iters=total_iters,
    )

    if lrs2["name"] == "CosineAnnealingWarmRestarts":
        t_steps = train_batches_per_epoch * lrs2["T_0"]
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_steps, eta_min=lrs2["min_lr"]
        )
    elif lrs2["name"] == "CosineAnnealingLR":
        t_steps = train_batches_per_epoch * lrs2["T_max"]
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_steps, eta_min=lrs2["min_lr"]
        )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[total_iters],
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
    wandb.login()
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config,
        id=config["wandb"]["id"],
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
    config_path = Path("/Users/zsa8rk/Coding/MetaPARC/metaparc/run/config.yaml")
    main(config_path)
