"""
Training script for convolutional neural network models.

Author: Florian Wiesner
Date: 2025-03-10
"""

from pathlib import Path

import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from metaparc.data.dataset_utils import get_dataloader
from metaparc.model.transformer.model import get_model
from metaparc.utils.train_vis import visualize_predictions


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    save_dir: Path,
    scheduler: optim.lr_scheduler.SequentialLR,
) -> float:
    """Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train
    device : torch.device
        Device to use for training
    train_loader : DataLoader
        DataLoader for training data
    optimizer : torch.optim.Optimizer
        Optimizer for training
    criterion : nn.Module
        Loss function
    scheduler : optim.lr_scheduler.SequentialLR
        Learning rate scheduler
    save_dir : Path
        The directory to save the predictions

    Returns
    -------
    float
        Average training loss for the epoch
    """
    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc="Training one epoch")
    for batch_idx, batch in enumerate(pbar):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        target = torch.cat((x[:, 1:, :, :, :], y), dim=1)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()

        # Clip gradients to norm 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            # Visualize predictions
            visualize_predictions(
                save_path=save_dir / f"pred_target_{batch_idx:04d}.png",
                predictions=output,
                targets=target,
            )
        # Step learning rate scheduler
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        pbar.set_postfix({"loss": train_loss / (batch_idx + 1), "lr": lr})

    return train_loss / len(train_loader)


def validate(
    model: nn.Module, device: torch.device, val_loader: DataLoader, criterion: nn.Module
) -> float:
    """Validate the model.

    Parameters
    ----------
    model : nn.Module
        The model to validate
    device : torch.device
        Device to use for validation
    val_loader : DataLoader
        DataLoader for validation data
    criterion : nn.Module
        Loss function

    Returns
    -------
    float
        Average validation loss for the epoch
    """
    model.eval()
    val_loss = 0

    pbar = tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            target = torch.cat((x[:, 1:, :, :, :], y), dim=1)

            output = model(x)
            loss = criterion(output, target)

            val_loss += loss.item()

            pbar.set_postfix({"loss": val_loss / (batch_idx + 1)})

    return val_loss / len(val_loader)


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
    else:
        raise ValueError(f"Optimizer {config['name']} not supported")

    return optimizer


def main():
    """Main training function."""
    config_path = Path("/Users/zsa8rk/Coding/MetaPARC/metaparc/run/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader = get_dataloader(config["data"], config["training"], split="train")
    epoch_size = len(train_loader)
    val_loader = get_dataloader(config["data"], config["training"], split="val")

    # Create model
    model = get_model(model_config=config["model"])
    model.to(device)
    # model = torch.compile(model)

    # Define loss function and optimizer
    lrs_config = config["training"]["lr_scheduler"]
    opt_config = config["training"]["optimizer"]
    criterion = nn.MSELoss()
    optimizer = get_optimizer(model, opt_config)

    # Create learning rate schedulers
    scheduler = get_lr_scheduler(optimizer, lrs_config, epoch_size)

    # Create save directory if it doesn't exist
    save_dir = Path(config["data"]["model_checkpoint_dir"]) / "transformer"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    train_losses = []
    val_losses = []

    best_loss = float("inf")

    epochs = config["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss = train_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            save_dir=save_dir / f"epoch_{epoch:03d}",
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, device, val_loader, criterion)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"Model saved with loss: {best_loss:.4f}")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "scheduler_state_dict": scheduler.state_dict(),
            },
            save_dir / f"checkpoint_epoch_{epoch}.pth",
        )


if __name__ == "__main__":
    main()
