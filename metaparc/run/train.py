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
from torchvision import transforms

import einops

import matplotlib.pyplot as plt
from tqdm import tqdm

from metaparc.data.datasets import PhysicsDataset
from metaparc.model.base_models.conv_model import ConvModel


def get_data_loaders(
    data_dir: Path,
    batch_size: int,
    n_steps_input: int,
    n_steps_output: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation.

    Parameters
    ----------
    data_dir : Path
        Directory containing the dataset
    batch_size : int
        Batch size for training
    n_steps_input : int
        Number of consecutive time steps in the input
    n_steps_output : int
        Number of consecutive time steps in the output

    Returns
    -------
    tuple
        (train_loader, val_loader)
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader = DataLoader(
        train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_loader, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def get_model(input_channels, output_dim, hidden_channels, dropout_rate):
    """Get the model.

    Parameters
    ----------
    input_channels : int
    """
    return ConvModel(
        input_channels=input_channels,
        output_channels=output_dim,
        hidden_channels=hidden_channels,
    )



def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
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

    Returns
    -------
    float
        Average training loss for the epoch
    """
    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc="Training one epoch")
    for batch_idx, batch in enumerate(pbar):
        x = batch["input_fields"]
        x = collate_fn(x)
        x = x.to(device)

        y = batch["output_fields"]
        y = collate_fn(y)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pbar.set_postfix({"loss": train_loss / (batch_idx + 1)})
        break

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
            x = batch["input_fields"]
            x = collate_fn(x)
            x = x.to(device)

            y = batch["output_fields"]
            y = collate_fn(y)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            val_loss += loss.item()

            pbar.set_postfix({"loss": val_loss / (batch_idx + 1)})

    return val_loss / len(val_loader)


def main():
    """Main training function."""
    data_dir = Path("C:/Users/zsa8rk/Coding/MetaPARC/data/datasets/shear_flow/data")
    batch_size = 4
    input_channels = 4
    output_dim = 128 * 256
    hidden_channels = 32
    dropout_rate = 0.1
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
    train_loader, val_loader = get_data_loaders(
        data_dir, batch_size, n_steps_input, n_steps_output, num_workers
    )

    # Create model
    model = get_model(
        input_channels=input_channels * n_steps_input,
        output_dim=output_dim * n_steps_output * input_channels,
        hidden_channels=hidden_channels,
        dropout_rate=dropout_rate,
    )
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create save directory if it doesn't exist
    save_dir = Path("models")
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

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
