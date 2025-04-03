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
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            # Visualize predictions
            visualize_predictions(
                save_path=save_dir / f"pred_target_{batch_idx:04d}.png",
                predictions=output,
                targets=target,
            )
        pbar.set_postfix({"loss": train_loss / (batch_idx + 1)})

    return train_loss / len(train_loader)


def validate(model, device, val_loader, criterion):
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

            target = torch.cat((x[:, 1:, :, :, :], y), dim=1, device=device)

            output = model(x)
            loss = criterion(output, target)

            val_loss += loss.item()

            pbar.set_postfix({"loss": val_loss / (batch_idx + 1)})

    return val_loss / len(val_loader)


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
    val_loader = get_dataloader(config["data"], config["training"], split="val")

    # Create model
    model = get_model(model_config=config["model"])
    model.to(device)

    # Define loss function and optimizer
    lr = config["training"]["learning_rate"]
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create save directory if it doesn't exist
    save_dir = config["model_checkpoint_dir"] / "transformer"
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
            },
            save_dir / f"checkpoint_epoch_{epoch}.pth",
        )


if __name__ == "__main__":
    main()
