"""
Use a trained model to predict the next timesteps of a given input.
"""

from pathlib import Path
import torch
import yaml
import numpy as np

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from lpfm.model.transformer.model import get_model
from lpfm.data.dataset_utils import get_datasets
from lpfm.data.phys_dataset import SuperDataset
from lpfm.utils.logger import get_logger
from lpfm.utils.rollout_video import create_field_video
from lpfm.model.transformer.loss_fns import NMSELoss
from lpfm.utils.plotting.plot_lossVsTime import LossVsTimePlotter

logger = get_logger(__name__, log_level="INFO")


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


def rollout_prediction(
    model: torch.nn.Module,
    dataset: SuperDataset,
    device: torch.device,
    traj_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rollout prediction for a given dataset.

    The model gets only the first timestep of the trajectory and then needs to predict all
    subsequent timesteps.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction
    dataset : SuperDataset
        Dataset containing the trajectories
    device : torch.device
        Device to run the model on
    traj_idx : int
        Index of the trajectory to predict

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing the predicted outputs, the ground truth, and the loss
    """
    criterion = NMSELoss(dims=(2, 3), return_scalar=False)

    traj_idx = min(traj_idx, len(dataset) - 1)
    input, full_traj = dataset[traj_idx]

    input = input.to(device)
    full_traj = full_traj.to(device)

    # add batch dimension
    input = input.unsqueeze(0)
    full_traj = full_traj.unsqueeze(0)

    B, T, H, W, C = full_traj.shape

    outputs = []
    with torch.no_grad():
        for i in range(T):
            output = model(input)
            # if the output is nan, stop the rollout
            if torch.isnan(output).any() or torch.isinf(output).any():
                break

            outputs.append(output)

            # Update input
            input = torch.cat([input[:, 1:, ...], output], dim=1)

    outputs = torch.cat(outputs, dim=1)
    # remove the last elements of the trajectory to fit the output shape
    t_output = outputs.shape[1]
    full_traj = full_traj[:, :t_output, ...]

    # loss is still a tensor of shape (B, T, H, W, C), averaged only over H and W
    loss = criterion(outputs, full_traj)
    # reduce over H and W
    loss = torch.mean(loss, dim=(2, 3))  # (B, T, C)
    # remove batch dimension
    loss = loss.squeeze(0)  # (T, C)

    # remove batch dimension
    outputs = outputs.squeeze(0)
    full_traj = full_traj.squeeze(0)

    assert not torch.isinf(outputs).any()
    assert not torch.isinf(full_traj).any()
    assert not torch.isinf(loss).any()

    return outputs, full_traj, loss


def load_config(model_path: Path) -> dict:
    """Load the configuration file.

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


def average_predictions(
    model: torch.nn.Module,
    dataset: SuperDataset,
    device: torch.device,
    num_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute multiple rollout predictions and average the losses.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction
    dataset : SuperDataset
        Dataset containing the trajectories
    device : torch.device
        Device to run the model on
    num_samples : int
        Number of samples to average over

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
        _, _, loss = rollout_prediction(model, dataset, device, traj_idx)
        losses.append(loss)
    # crop the losses to the same length
    min_length = min(loss.shape[0] for loss in losses)
    losses = [loss[:min_length] for loss in losses]
    losses = torch.stack(losses, dim=0)

    assert not torch.isnan(losses).any()
    # compute mean and std over the trajectories
    return torch.mean(losses, dim=0), torch.std(losses, dim=0)


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
    model_list = ["m-main-run-all-0001"]

    # base_path = Path("C:/Users/zsa8rk/Coding/Large-Physics-Foundation-Model/logs")
    base_path = Path("/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs")

    dt = 1
    num_samples = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in model_list:
        results_dir = base_path / model_name / "rollout_prediction"
        results_dir.mkdir(exist_ok=True, parents=True)

        model_path = base_path / model_name

        config = load_config(model_path)
        model_config = config["model"]
        model = load_model(model_path / "best_model.pth", device, model_config)

        data_config = config["data"]
        data_config["full_trajectory_mode"] = True
        data_config["max_rollout_steps"] = 30
        data_config["dt_stride"] = dt

        # data_config["datasets"] = ["cylinder_sym_flow_water"]
        datasets: dict = get_datasets(
            data_config,
            split="test",
        )

        for dataset_name, dataset in datasets.items():
            logger.info(f"Rolling out prediction for {dataset_name}")
            # Compute average loss over multiple trajectories
            mean_loss, std_loss = average_predictions(
                model, dataset, device, num_samples=num_samples
            )
            logger.info("   Finished computing average loss")
            plotter = plot_loss(mean_loss, std_loss)
            save_path = results_dir / f"{dataset_name}_loss_dt{dt}.png"
            plotter.save_figure(save_path)

            #########################################################
            # Create video of the rollout prediction
            #########################################################

            # Get predictions for visualization
            rollout_predictions, full_traj, loss = rollout_prediction(
                model, dataset, device
            )
            logger.info("Finished rolling out prediction")

            # rotate x and y axis
            rollout_predictions = rollout_predictions.permute(0, 2, 1, 3)
            full_traj = full_traj.permute(0, 2, 1, 3)

            rollout_predictions = rollout_predictions.cpu().numpy()
            full_traj = full_traj.cpu().numpy()
            loss = loss.cpu().numpy()

            # Create videos for both actual and predicted trajectories
            output_dir = results_dir / "videos"
            output_dir.mkdir(exist_ok=True)

            # Create video of the ground truth and the predicted trajectory
            logger.info("Creating video of rollout prediction")
            create_field_video(
                full_traj,
                rollout_predictions,
                loss,
                output_dir,
                f"{dataset_name}_rollout_dt{dt}",
                fps=2,
            )


if __name__ == "__main__":
    main()
