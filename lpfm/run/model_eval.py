"""
Detailed evaluation of the model, its predictions, and the losses.
By: Florian Wiesner
Date: 2025-05-01
"""

from pathlib import Path

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import torch
import matplotlib.pyplot as plt
from lpfm.model.transformer.model import get_model
from lpfm.data.dataset_utils import get_datasets
from lpfm.data.phys_dataset import PhysicsDataset
from lpfm.utils.logger import get_logger
from lpfm.model.transformer.loss_fns import NMSELoss

logger = get_logger(__name__, log_level="INFO")


def load_config(model_path: Path) -> dict:
    config_path = model_path / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def load_model(
    model_path: Path, device: torch.device, model_config: dict
) -> torch.nn.Module:
    """Load a model from a checkpoint.

    Parameters
    ----------
    model_path : Path
        Path to the model checkpoint
    device : torch.device
        Device to load the model to
    model_config : dict
        Model configuration dictionary

    Returns
    -------
    torch.nn.Module
        Loaded model
    """
    data = torch.load(model_path, map_location=device, weights_only=False)
    model = get_model(model_config)
    model.load_state_dict(data["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def eval_on_dataset(
    model: torch.nn.Module,
    dataset: PhysicsDataset,
    device: torch.device,
    num_samples: int = 100,
) -> dict:
    print(f"   Evaluating on {num_samples} samples")
    criterion = NMSELoss(return_scalar=False)

    losses = {
        "pressure": [],
        "density": [],
        "temperature": [],
        "vel_x": [],
        "vel_y": [],
    }

    samples = torch.randint(0, len(dataset), (num_samples,))
    for i in samples:
        x, target = dataset[i]
        x = x.to(device)
        target = target.to(device)
        # add batch dimension
        x = x.unsqueeze(0)
        target = target.unsqueeze(0)
        y = model(x)

        loss = criterion(y, target)
        loss_pressure = loss[..., 0].mean().item()
        loss_density = loss[..., 1].mean().item()
        loss_temperature = loss[..., 2].mean().item()
        loss_vel_x = loss[..., 3].mean().item()
        loss_vel_y = loss[..., 4].mean().item()

        losses["pressure"].append(loss_pressure)
        losses["density"].append(loss_density)
        losses["temperature"].append(loss_temperature)
        losses["vel_x"].append(loss_vel_x)
        losses["vel_y"].append(loss_vel_y)

    return losses


def main():
    num_samples = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = Path("/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs")
    model_name = "ti-main-run-all-0005"
    model_path = base_path / model_name

    config = load_config(model_path)
    data_config = config["data"]
    datasets = get_datasets(data_config, split="train")

    model_config = config["model"]
    model = load_model(model_path / "best_model.pth", device, model_config)

    # create plot with all losses
    # each dataset is a row, the columns are the losses
    cols = ["pressure", "density", "temperature", "vel_x", "vel_y"]
    num_cols = len(cols)
    num_rows = len(datasets)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    for i, (name, dataset) in enumerate(datasets.items()):
        print(f"Evaluating on {name} dataset")
        losses = eval_on_dataset(model, dataset, device, num_samples=num_samples)
        # print(losses)
        for j, col in enumerate(cols):
            axs[i, j].plot(losses[col])
            axs[i, j].set_title(f"{name} {col}")
    plt.savefig(f"{model_path}/losses.png")
    plt.close()


if __name__ == "__main__":
    main()
