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

from metaparc.model.transformer.model import get_model
from metaparc.data.dataset_utils import get_datasets
from metaparc.data.phys_dataset import PhysicsDataset, SuperDataset


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
    model_state_dict = torch.load(model_path, map_location=device)
    model = get_model(model_config)

    # Remove RevIN buffers from state dict since they were removed from the model
    keys_to_remove = [
        k for k in model_state_dict.keys() if "revin.mean" in k or "revin.stdev" in k
    ]
    for k in keys_to_remove:
        del model_state_dict[k]

    model.load_state_dict(model_state_dict, strict=False)
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
    return outputs, full_traj


# def next_step_prediction(
#     model: torch.nn.Module, dataset: PhysicsDataset
# ) -> torch.Tensor:
#     """Next step prediction for a given dataset.

#     The model gets the current timestep and needs to predict the next timestep.
#     """
#     full_traj = dataset[0]
#     input = full_traj[0]

#     outputs = []
#     with torch.no_grad():
#         for i in range(len(input)):
#             output = model(input)


def load_config(model_path: Path) -> dict:
    config_path = model_path / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def main():
    model_path = Path("C:/Users/zsa8rk/Coding/MetaPARC/logs/tokenizer_overlap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(model_path)
    model_config = config["model"]
    model = load_model(model_path / "best_model.pth", device, model_config)

    data_config = config["data"]
    data_config["full_trajectory_mode"] = True

    datasets = get_datasets(
        data_config,
        split="test",
    )

    for dataset in datasets:
        dataset = SuperDataset([dataset], out_shape=data_config["out_shape"])
        rollout_predictions, full_traj = rollout_prediction(model, dataset, device)


if __name__ == "__main__":
    main()
