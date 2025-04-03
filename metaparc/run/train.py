"""
Training script for convolutional neural network models.

Author: Florian Wiesner
Date: 2025-03-10
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from metaparc.model.transformer.model import PhysicsTransformer
from metaparc.data.datasets import PhysicsDataset, get_dataloader
from metaparc.utils.train_vis import visualize_predictions


def get_model(
    input_channels,
    hidden_channels,
    num_heads,
    dropout,
    patch_size,
    num_layers,
):
    """Get the model."""
    return PhysicsTransformer(
        input_channels=input_channels,
        hidden_dim=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
        patch_size=patch_size,
        num_layers=num_layers,
    )


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
    data_dir = Path("/Users/zsa8rk/Coding/MetaPARC/data/datasets/turbulent_radiative_layer_2D/data")
    model_checkpoint_dir = Path("/Users/zsa8rk/Coding/MetaPARC/model_checkpoints")
    # Training parameters
    batch_size = 2
    input_channels = 4
    hidden_channels = 96 * 4
    num_heads = 16
    dropout = 0.1
    patch_size = (4, 16, 16)
    num_layers = 8
    lr = 0.001
    epochs = 10

    n_steps_input = 4
    n_steps_output = 1
    num_workers = 4

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader = get_dataloader(
        PhysicsDataset(
            data_dir / "train",
            split="train",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=1,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = get_dataloader(
        PhysicsDataset(
            data_dir / "valid",
            split="val",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=1,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Create model
    model = get_model(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
        patch_size=patch_size,
        num_layers=num_layers,
    )
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create save directory if it doesn't exist
    save_dir = model_checkpoint_dir / "transformer"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    train_losses = []
    val_losses = []

    best_loss = float("inf")

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
