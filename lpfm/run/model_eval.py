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
from torch.utils.data import DataLoader
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


class LossEvaluator:
    def __init__(
        self,
        base_path: Path = None,
        num_samples: int = 10,
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        print("Setting up evaluator")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = load_config(base_path)
        self.model = load_model(
            base_path / "best_model.pth", self.device, self.config["model"]
        )
        self.datasets = get_datasets(self.config["data"], split="train")
        self.base_path = base_path
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.no_grad()
    def eval_on_dataset(self, dataset: PhysicsDataset):
        print(f"   Evaluating on {self.num_samples} samples")
        criterion = NMSELoss(return_scalar=False)

        losses = {
            "pressure": [],
            "density": [],
            "temperature": [],
            "vel_x": [],
            "vel_y": [],
        }

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        for i, (x, target) in enumerate(loader):
            dataset_name = dataset.metadata.dataset_name
            print(f"   Evaluating on {i}th batch")
            x = x.to(self.device)
            target = target.to(self.device)
            y = self.model(x)

            loss = criterion(y, target)
            loss_pressure = torch.mean(
                loss[..., 0], dim=(1, 2, 3)
            )  # get the loss of each sample, dont average accross batches
            loss_density = torch.mean(loss[..., 1], dim=(1, 2, 3))
            loss_temperature = torch.mean(loss[..., 2], dim=(1, 2, 3))
            loss_vel_x = torch.mean(loss[..., 3], dim=(1, 2, 3))
            loss_vel_y = torch.mean(loss[..., 4], dim=(1, 2, 3))

            # convert batches to list
            loss_pressure = loss_pressure.tolist()
            loss_density = loss_density.tolist()
            loss_temperature = loss_temperature.tolist()
            loss_vel_x = loss_vel_x.tolist()
            loss_vel_y = loss_vel_y.tolist()
            self.show_large_losses(
                loss_pressure,
                target[..., 0],
                y[..., 0],
                name=f"{dataset_name}_pressure",
            )
            self.show_large_losses(
                loss_density,
                target[..., 1],
                y[..., 1],
                name=f"{dataset_name}_density_sample_{i}",
            )
            self.show_large_losses(
                loss_temperature,
                target[..., 2],
                y[..., 2],
                name=f"{dataset_name}_temperature_sample_{i}",
            )
            self.show_large_losses(
                loss_vel_x,
                target[..., 3],
                y[..., 3],
                name=f"{dataset_name}_vel_x_sample_{i}",
            )
            self.show_large_losses(
                loss_vel_y,
                target[..., 4],
                y[..., 4],
                name=f"{dataset_name}_vel_y_sample_{i}",
            )

            losses["pressure"].extend(loss_pressure)
            losses["density"].extend(loss_density)
            losses["temperature"].extend(loss_temperature)
            losses["vel_x"].extend(loss_vel_x)
            losses["vel_y"].extend(loss_vel_y)
            if i > self.num_samples:
                break

        return losses

    def show_large_losses(
        self,
        loss_list: list,
        target: torch.Tensor,
        y: torch.Tensor,
        name: str,
    ):
        """Export the largest losses to a file."""
        # target and y are of shape (batch_size, 1, H, W)

        # find losses above 10
        indices = [i for i, loss in enumerate(loss_list) if loss > 10]
        if len(indices) == 0:
            print(f"   No large losses for {name}")
            return
        for i in indices:
            target_large = target[i, 0, ...].squeeze()
            target_norm = target_large.pow(2).mean() + 1e-6
            y_large = y[i, 0, ...].squeeze()
            # convert to numpy
            target_large = target_large.cpu().numpy()
            y_large = y_large.cpu().numpy()

            # transpose target and y
            target_large = target_large.transpose(1, 0)
            y_large = y_large.transpose(1, 0)

            # add a colorbar
            # make a figure with 2 cols, target and y
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # normalize target and y
            vmin = min(target_large.min(), y_large.min())
            vmax = max(target_large.max(), y_large.max())
            im0 = axs[0].imshow(target_large, vmin=vmin, vmax=vmax, cmap="viridis")
            im1 = axs[1].imshow(y_large, vmin=vmin, vmax=vmax, cmap="viridis")
            # add colorbars
            plt.colorbar(im0, ax=axs[0])
            plt.colorbar(im1, ax=axs[1])
            # add a title to the figure
            fig.suptitle(f"Loss: {loss_list[i]}, target_norm: {target_norm}")
            fig.savefig(self.base_path / f"large_losses_{name}_{i}.png")
            plt.close()

            print(f"   Saved large losses to {name}_{i}.png")

    def main(self):
        # create plot with all losses

        # create plot with all losses
        # each dataset is a row, the columns are the losses
        cols = ["pressure", "density", "temperature", "vel_x", "vel_y"]
        num_cols = len(cols)
        num_rows = len(self.datasets)
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows)
        )
        for i, (dataset_name, dataset) in enumerate(self.datasets.items()):
            print(f"Evaluating on {dataset_name} dataset")
            losses = self.eval_on_dataset(dataset)
            for j, col in enumerate(cols):
                axs[i, j].plot(losses[col])
                axs[i, j].set_title(f"{dataset_name} {col}")
        plt.savefig(f"{self.base_path}/losses.png")
        plt.close()


if __name__ == "__main__":
    import os

    # set cuda visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    base_path = Path("/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs")
    model_name = "ti-main-run-all-0005"
    loss_evaluator = LossEvaluator(
        base_path=base_path / model_name, batch_size=128, num_workers=8, num_samples=20
    )
    loss_evaluator.main()
