"""
Use a trained model to predict the next timesteps of a given input.
Evaluate the model on each dataset.
"""

from pathlib import Path
import torch
import yaml
import numpy as np

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import matplotlib.pyplot as plt

from lpfm.model.transformer.model import get_model
from lpfm.data.dataset_utils import get_datasets
from lpfm.data.phys_dataset import PhysicsDataset
from lpfm.utils.logger import get_logger
from lpfm.utils.rollout_video import create_field_video
from lpfm.model.transformer.loss_fns import NMSELoss
from lpfm.utils.plotting.plot_lossVsTime import LossVsTimePlotter
from lpfm.run.run_utils import load_stored_model

logger = get_logger(__name__, log_level="INFO")


class PhysicsPredictor:
    """A class for making physics predictions using a trained model.

    Parameters
    ----------
    model_path : Path
        Path to the model checkpoint
    device : torch.device
        Device to run the model on
    model_config : dict
        Model configuration dictionary
    data_config : dict
        Data configuration dictionary
    """

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        model_config: dict,
        data_config: dict,
        results_dir: Path,
    ):
        self.device = device
        self.model = self._load_model(model_path, device, model_config)
        self.data_config = data_config
        self.criterion = NMSELoss(dims=(2, 3), return_scalar=False)
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self.datasets = get_datasets(data_config, split="test")

    def _load_model(
        self, model_path: Path, device: torch.device, model_config: dict
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
        data = load_stored_model(model_path, device, remove_ddp=True)
        model = get_model(model_config)
        model.load_state_dict(data["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        return model

    @torch.inference_mode()
    def predict(
        self,
        dataset: PhysicsDataset,
        traj_idx: int = 0,
        rollout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make predictions for a given dataset.
        Can be used for next step prediction or rollout prediction.

        Parameters
        ----------
        dataset : PhysicsDataset
            Dataset containing the trajectories
        traj_idx : int, optional
            Index of the trajectory to predict, by default 0
        rollout : bool, optional
            Whether to rollout the full trajectory prediction, by default False


        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing the predicted outputs,
            the ground truth, and the loss at each timestep
        """
        # get first trajectory
        traj_idx = min(traj_idx, len(dataset) - 1)
        input, full_traj = dataset[traj_idx]

        input = input.to(self.device)
        full_traj = full_traj.to(self.device)

        # add batch dimension
        input = input.unsqueeze(0)
        full_traj = full_traj.unsqueeze(0)

        B, T, H, W, C = full_traj.shape

        outputs = []
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            for i in range(T):  # T-1 because we predict the next step
                # Predict next timestep
                output = self.model(input)
                # if the output is nan, stop the rollout
                if torch.isnan(output).any() or torch.isinf(output).any():
                    break

                outputs.append(output)
                # Update input
                if rollout:
                    input = torch.cat([input[:, 1:, ...], output], dim=1)
                else:
                    input = torch.cat(
                        [input[:, 1:, ...], full_traj[:, i, ...].unsqueeze(1)], dim=1
                    )

        outputs = torch.cat(outputs, dim=1)

        # loss is still a tensor of shape (B, T, H, W, C), averaged only over H and W
        loss = self.criterion(outputs, full_traj)
        # reduce over H and W
        loss = torch.mean(loss, dim=(2, 3))  # (B, T, C)
        # remove batch dimension
        loss = loss.squeeze(0)  # (T, C)

        # remove batch dimension
        outputs = outputs.squeeze(0)
        full_traj = full_traj.squeeze(0)

        # Return predictions and ground truth (excluding first timestep)
        return outputs, full_traj, loss

    def average_predictions(
        self,
        dataset: PhysicsDataset,
        num_samples: int = 10,
        rollout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multiple predictions and average the losses.

        Parameters
        ----------
        dataset : PhysicsDataset
            Dataset containing the trajectories
        num_samples : int, optional
            Number of samples to average over, by default 10
        rollout : bool, optional
            Whether to rollout the full trajectory prediction, by default False


        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The averaged loss and standard deviation for each channel and timestep
        """
        losses = []
        # random trajectory indices
        indices = np.arange(len(dataset))
        traj_idxs = np.random.choice(indices, size=num_samples, replace=False)
        for traj_idx in traj_idxs:
            logger.info(f"\tComputing loss for trajectory {traj_idx}")
            _, _, loss = self.predict(dataset, traj_idx, rollout=rollout)
            losses.append(loss)
        losses = torch.stack(losses, dim=0)
        # compute mean and std over the trajectories
        return torch.mean(losses, dim=0), torch.std(losses, dim=0)

    def _create_visualization(
        self,
        dataset: PhysicsDataset,
        traj_idx: int,
        output_dir: Path,
        title: str,
        rollout: bool = False,
        fps: int = 2,
    ) -> None:
        """Create visualization of predictions.

        Parameters
        ----------
        dataset : PhysicsDataset
            Dataset containing the trajectories
        traj_idx : int
            Index of the trajectory to visualize
        output_dir : Path
            Directory to save the visualization
        title : str
            Title for the visualization
        rollout : bool, optional
            Whether to rollout the full trajectory prediction, by default False
        fps : int, optional
            Frames per second for the video, by default 2
        """
        next_step_predictions, full_traj, loss = self.predict(
            dataset, traj_idx, rollout=rollout
        )
        # rotate x and y axis
        next_step_predictions = next_step_predictions.permute(0, 2, 1, 3)
        full_traj = full_traj.permute(0, 2, 1, 3)

        next_step_predictions = next_step_predictions.cpu().numpy()
        full_traj = full_traj.cpu().numpy()
        loss = loss.cpu().numpy()

        # Create videos for both actual and predicted trajectories
        output_dir.mkdir(exist_ok=True)

        # Create video of the ground truth and the predicted trajectory
        logger.info("   Creating video of next step prediction")
        create_field_video(
            full_traj,
            next_step_predictions,
            loss,
            output_dir,
            title,
            fps=fps,
        )

    def predict_all(
        self,
        num_samples: int = 10,
        fps: int = 2,
        rollout: bool = False,
    ):
        """Predict all datasets."""
        dt = self.data_config["dt_stride"]

        if rollout:
            save_dir = self.results_dir / "rollout"
        else:
            save_dir = self.results_dir / "nextstep"

        save_dir.mkdir(exist_ok=True, parents=True)

        for dataset_name, dataset in self.datasets.items():
            logger.info(f"  Computing average loss for {dataset_name}")
            mean_loss, std_loss = self.average_predictions(
                dataset, num_samples=num_samples, rollout=rollout
            )
            logger.info(f"  Finished computing average loss for {dataset_name}")
            plotter = plot_loss(mean_loss, std_loss)

            save_path = save_dir / f"{dataset_name}_loss_dt{dt}.png"
            plotter.save_figure(save_path)

            # Create video of the next step prediction
            output_dir = save_dir / "videos"
            self._create_visualization(
                dataset,
                traj_idx=100,
                output_dir=output_dir,
                title=f"{dataset_name}_next_step_dt{dt}",
                fps=fps,
                rollout=rollout,
            )


def load_config(model_path: Path) -> dict:
    """Load configuration from a yaml file.

    Parameters
    ----------
    model_path : Path
        Path to the model directory containing config.yaml

    Returns
    -------
    dict
        Configuration dictionary
    """
    config_path = model_path / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def plot_loss(mean_loss: torch.Tensor, std_loss: torch.Tensor) -> LossVsTimePlotter:
    """Plot the loss for each channel.
    x=timestep, y=loss

    Parameters
    ----------
    mean_loss : torch.Tensor
        The mean loss for each channel and timestep
    std_loss : torch.Tensor
        The standard deviation of the loss for each channel and timestep
    """
    min_loss = torch.min(mean_loss).item()
    max_loss = torch.max(mean_loss).item()
    time_steps = mean_loss.shape[0]
    x_ticks = [0, time_steps // 2, time_steps]
    y_ticks = [0, max_loss * 0.5, max_loss]
    plotter = LossVsTimePlotter(x_ticks=x_ticks, y_ticks=y_ticks, color="white")
    plotter.plot(mean_loss, std_loss)
    plotter.legend(title="Fields", loc="upper right")
    return plotter


def main():
    model_list = ["ti-cyl-sym-flow-0001c"]

    base_path = Path("C:/Users/zsa8rk/Coding/Large-Physics-Foundation-Model/logs")
    dt = 1
    num_samples = 10
    fps = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in model_list:
        results_dir = base_path / model_name

        config = load_config(results_dir)
        model_config = config["model"]
        model_file = results_dir / "best_model.pth"

        data_config = config["data"]
        data_config["full_trajectory_mode"] = True
        data_config["max_rollout_steps"] = 50
        data_config["dt_stride"] = dt

        # data_config["datasets"] = ["cylinder_flow"]

        predictor = PhysicsPredictor(
            model_file,
            device,
            model_config,
            data_config,
            results_dir,
        )

        logger.info(f"Predicting {model_name} with rollout")
        predictor.predict_all(num_samples=num_samples, fps=fps, rollout=True)
        logger.info(f"Predicting {model_name} without rollout")
        predictor.predict_all(num_samples=num_samples, fps=fps, rollout=False)


if __name__ == "__main__":
    main()
