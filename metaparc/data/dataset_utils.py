from pathlib import Path

import torch
from torch.utils.data import default_collate, DataLoader
from the_well.data.augmentation import (
    Compose,
    RandomAxisFlip,
    RandomAxisRoll,
    RandomAxisPermute,
    # NOTE: Image Resize should come here as well
)

from metaparc.data.phys_dataset import SuperDataset, PhysicsDataset
from metaparc.data.ind_datasets import (
    RayleighBenardDataset,
    ShearFlowDataset,
    TurbulentRadiativeDataset,
    EulerDataset,
    ComsolIncompressibleFlowDataset,
    ComsolHeatedFlowDataset,
)


def get_rng_transforms(p_flip: float) -> Compose:
    """Get a list of transforms to apply to the data."""
    return Compose(
        *[
            RandomAxisFlip(p=p_flip),
            RandomAxisRoll(p=p_flip),
            RandomAxisPermute(p=p_flip),
        ]
    )


def collate_fn(data: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Collate function for the dataset.

    Get input and target field tensors.
    The fields are of shape (Time steps, H, W, C)
    We want to get a batch of shape (B, Time steps, H, W, C).
    Additionally, we replace NaNs with 0.

    Parameters
    ----------
    data : tuple[torch.Tensor, torch.Tensor]
        Tuple of input and target field tensors

    Returns
    -------
    batch : torch.Tensor
        Batch of shape (B, Time steps, H, W, C)
    """

    batch = default_collate(data)  # (B, Time steps, H, W, C)
    x = batch[0]
    y = batch[1]

    # # rearrange to (B, Time steps & C, H, W)
    # x = einops.rearrange(x, "batch time c h w -> batch (time c) h w")
    # y = einops.rearrange(y, "batch time c h w -> batch (time c) h w")

    return x, y


def get_dataloader(data_config: dict, train_config: dict, split: str) -> DataLoader:
    """Get a dataloader for the dataset.

    Parameters
    ----------
    data_config : dict
        Configuration for the dataset.
    train_config : dict
        Configuration for the training.
    split : str
        Split to load ("train", "val", "test")
    """
    datasets = get_datasets(data_config, split)
    train_super_dataset = SuperDataset(datasets, out_shape=data_config["out_shape"])

    dataloader = DataLoader(
        dataset=train_super_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=train_config["num_workers"],
        pin_memory=True,
    )

    return dataloader


def get_datasets(data_config: dict, split: str = "train") -> list[PhysicsDataset]:
    """Get the datasets."""

    split_name = split
    if split == "val":
        split_name = "valid"

    data_dir = Path(data_config["data_dir"])
    if data_config["full_trajectory_mode"]:
        full_traj = data_config["full_trajectory_mode"]
        max_rollout_steps = data_config["max_rollout_steps"]

    n_steps_input = data_config["n_steps_input"]
    n_steps_output = data_config["n_steps_output"]
    dt_stride = data_config["dt_stride"]
    length_limit = data_config["length_limit"]

    datasets = []
    if "shear_flow" in data_config["datasets"]:
        shearflow_dataset = ShearFlowDataset(
            data_dir=data_dir / f"shear_flow/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
        )
        datasets.append(shearflow_dataset)

    if "rayleigh_benard" in data_config["datasets"]:
        rayleigh_benard_dataset = RayleighBenardDataset(
            data_dir=data_dir / f"rayleigh_benard/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
        )
        datasets.append(rayleigh_benard_dataset)

    if "turbulent_radiative_layer_2D" in data_config["datasets"]:
        turbulent_radiative_dataset = TurbulentRadiativeDataset(
            data_dir=data_dir / f"turbulent_radiative_layer_2D/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
        )
        datasets.append(turbulent_radiative_dataset)

    if "euler" in data_config["datasets"]:
        euler_dataset = EulerDataset(
            data_dir=data_dir / f"euler/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
        )
        datasets.append(euler_dataset)

    if "cylinder_pipe_flow_water" in data_config["datasets"]:
        cylinder_pipe_flow_dataset = ComsolIncompressibleFlowDataset(
            data_dir=data_dir / f"cylinder_pipe_flow_water/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
        )
        datasets.append(cylinder_pipe_flow_dataset)

    if "cylinder_sym_flow_water" in data_config["datasets"]:
        cylinder_symmetry_flow_dataset = ComsolIncompressibleFlowDataset(
            data_dir=data_dir / f"cylinder_sym_flow_water/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
        )
        datasets.append(cylinder_symmetry_flow_dataset)

    if "object_periodic_flow_water" in data_config["datasets"]:
        object_periodic_flow_dataset = ComsolIncompressibleFlowDataset(
            data_dir=data_dir / f"object_periodic_flow_water/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
        )
        datasets.append(object_periodic_flow_dataset)

    if "object_sym_flow_water" in data_config["datasets"]:
        object_symmetry_flow_dataset = ComsolIncompressibleFlowDataset(
            data_dir=data_dir / f"object_sym_flow_water/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
        )
        datasets.append(object_symmetry_flow_dataset)

    if "object_sym_flow_air" in data_config["datasets"]:
        object_symmetry_flow_dataset = ComsolIncompressibleFlowDataset(
            data_dir=data_dir / f"object_sym_flow_air/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
        )
        datasets.append(object_symmetry_flow_dataset)

    if "heated_object_pipe_flow_air" in data_config["datasets"]:
        heated_flow_dataset = ComsolHeatedFlowDataset(
            data_dir=data_dir / f"heated_object_pipe_flow_air/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            length_limit=length_limit,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
        )
        datasets.append(heated_flow_dataset)
    return datasets
