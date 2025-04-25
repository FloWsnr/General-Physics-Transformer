"""
Use a trained model to predict the next timesteps of a given input.
"""

from pathlib import Path
import torch
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from lpfm.model.transformer.model import get_model
from lpfm.data.dataset_utils import get_datasets
from lpfm.data.phys_dataset import SuperDataset
from lpfm.utils.logger import get_logger
from lpfm.utils.rollout_video import create_field_video

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
) -> torch.Tensor:
    """Rollout prediction for a given dataset.

    The model gets only the first timestep of the trajectory and then needs to predict all
    subsequent timesteps.
    """
    traj_idx = 0
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
            outputs.append(output)

            # Update input
            input = torch.cat([input[:, 1:, ...], output], dim=1)

    outputs = torch.cat(outputs, dim=1)

    # remove batch dimension
    outputs = outputs.squeeze(0)
    full_traj = full_traj.squeeze(0)
    return outputs, full_traj


def next_step_prediction(
    model: torch.nn.Module,
    dataset: SuperDataset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next step prediction for a given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction
    dataset : SuperDataset
        Dataset containing the trajectories
    device : torch.device
        Device to run the model on

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the predicted outputs and the ground truth
    """
    # get first trajectory
    traj_idx = 0
    input, full_traj = dataset[traj_idx]

    input = input.to(device)
    full_traj = full_traj.to(device)

    # add batch dimension
    input = input.unsqueeze(0)
    full_traj = full_traj.unsqueeze(0)

    B, T, H, W, C = full_traj.shape

    outputs = []
    with torch.no_grad():
        for i in range(T):  # T-1 because we predict the next step
            # Predict next timestep
            output = model(input)
            outputs.append(output)

            # Update input
            input = torch.cat(
                [input[:, 1:, ...], full_traj[:, i, ...].unsqueeze(1)], dim=1
            )

    outputs = torch.cat(outputs, dim=1)

    # remove batch dimension
    outputs = outputs.squeeze(0)
    full_traj = full_traj.squeeze(0)

    # Return predictions and ground truth (excluding first timestep)
    return outputs, full_traj


def load_config(model_path: Path) -> dict:
    config_path = model_path / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def main():
    model_path = Path(
        r"C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\logs\ti-main-run-single-0004"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(model_path)
    model_config = config["model"]
    model = load_model(model_path / "best_model.pth", device, model_config)

    data_config = config["data"]
    data_config["full_trajectory_mode"] = True
    data_config["max_rollout_steps"] = 100
    dt = 8
    data_config["dt_stride"] = dt

    # data_config["datasets"] = ["rayleigh_benard"]

    datasets: dict = get_datasets(
        data_config,
        split="test",
    )

    for dataset_name, dataset in datasets.items():
        dataset = SuperDataset(
            {dataset_name: dataset}, out_shape=data_config["out_shape"]
        )
        logger.info(
            f"Rolling out prediction for {dataset.datasets[0].metadata.dataset_name}"
        )
        rollout_predictions, full_traj = rollout_prediction(model, dataset, device)
        logger.info("Finished rolling out prediction")

        logger.info("Rolling out next step prediction")
        next_step_predictions, _ = next_step_prediction(model, dataset, device)
        logger.info("Finished rolling out next step prediction")

        # rotate x and y axis
        rollout_predictions = rollout_predictions.permute(0, 2, 1, 3)
        full_traj = full_traj.permute(0, 2, 1, 3)
        next_step_predictions = next_step_predictions.permute(0, 2, 1, 3)

        rollout_predictions = rollout_predictions.cpu().numpy()
        next_step_predictions = next_step_predictions.cpu().numpy()
        full_traj = full_traj.cpu().numpy()

        # Create videos for both actual and predicted trajectories
        output_dir = model_path / "videos"
        output_dir.mkdir(exist_ok=True)

        # Create video of the ground truth and the predicted trajectory
        logger.info("Creating video of rollout prediction")
        create_field_video(
            full_traj,
            rollout_predictions,
            output_dir,
            f"{dataset.datasets[0].metadata.dataset_name}_rollout_dt{dt}",
        )

        # Create video of the ground truth and the next step prediction
        logger.info("Creating video of next step prediction")
        create_field_video(
            full_traj,
            next_step_predictions,
            output_dir,
            f"{dataset.datasets[0].metadata.dataset_name}_next_step_dt{dt}",
        )


if __name__ == "__main__":
    main()
