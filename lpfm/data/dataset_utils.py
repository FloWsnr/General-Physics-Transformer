from pathlib import Path

import torch
from torch.utils.data import default_collate, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from the_well.data.augmentation import (
    Compose,
    RandomAxisFlip,
)

from lpfm.data.phys_dataset import SuperDataset, PhysicsDataset


def get_rng_transforms(p_flip: float) -> Compose:
    """Get a list of transforms to apply to the data."""
    return Compose(
        *[
            RandomAxisFlip(p=p_flip),
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


def get_dataloader(
    data_config: dict, train_config: dict, split: str, is_distributed: bool = False
) -> DataLoader:
    """Get a dataloader for the dataset.

    Parameters
    ----------
    data_config : dict
        Configuration for the dataset.
    train_config : dict
        Configuration for the training.
    split : str
        Split to load ("train", "val", "test")
    is_distributed : bool
        Whether to use distributed sampling
    """
    seed = train_config["seed"]
    datasets = get_datasets(data_config, split)
    train_super_dataset = SuperDataset(
        datasets,
        out_shape=data_config["out_shape"],
        max_samples_per_ds=data_config["max_samples_per_ds"],
    )
    length = len(train_super_dataset)

    if is_distributed:
        sampler = DistributedSampler(train_super_dataset, seed=seed)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(train_super_dataset, generator=generator)
    dataloader = DataLoader(
        dataset=train_super_dataset,
        batch_size=train_config["batch_size"],
        collate_fn=collate_fn,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=train_config["prefetch_factor"],
    )

    return dataloader


def get_datasets(data_config: dict, split: str = "train") -> dict[str, PhysicsDataset]:
    """Get the datasets."""

    include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }

    split_name = split
    if split == "val":
        split_name = "valid"

    data_dir = Path(data_config["data_dir"])
    if "full_trajectory_mode" in data_config and data_config["full_trajectory_mode"]:
        full_traj = data_config["full_trajectory_mode"]
        max_rollout_steps = data_config["max_rollout_steps"]
    else:
        max_rollout_steps = 10000
        full_traj = False

    n_steps_input = data_config["n_steps_input"]
    n_steps_output = data_config["n_steps_output"]
    dt_stride = data_config["dt_stride"]

    datasets = {}
    dataset_list: list[str] = data_config["datasets"].copy()
    if "shear_flow" in dataset_list:
        shearflow_dataset = PhysicsDataset(
            data_dir=data_dir / f"shear_flow/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["shear_flow"] = shearflow_dataset
        # remove shear_flow from dataset_list
        dataset_list.remove("shear_flow")

    if "rayleigh_benard" in dataset_list:
        rayleigh_benard_dataset = PhysicsDataset(
            data_dir=data_dir / f"rayleigh_benard/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["rayleigh_benard"] = rayleigh_benard_dataset
        # remove rayleigh_benard from dataset_list
        dataset_list.remove("rayleigh_benard")

    if "turbulent_radiative_layer_2D" in dataset_list:
        turbulent_radiative_dataset = PhysicsDataset(
            data_dir=data_dir / f"turbulent_radiative_layer_2D/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["turbulent_radiative_layer_2D"] = turbulent_radiative_dataset
        # remove turbulent_radiative_layer_2D from dataset_list
        dataset_list.remove("turbulent_radiative_layer_2D")

    if "euler_multi_quadrants_periodicBC" in dataset_list:
        euler_dataset = PhysicsDataset(
            data_dir=data_dir / f"euler_multi_quadrants_periodicBC/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["euler_multi_quadrants_periodicBC"] = euler_dataset
        # remove euler_multi_quadrants_periodicBC from dataset_list
        dataset_list.remove("euler_multi_quadrants_periodicBC")

    if "cylinder_pipe_flow_water" in dataset_list:
        cylinder_pipe_flow_dataset = PhysicsDataset(
            data_dir=data_dir / f"cylinder_pipe_flow_water/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["cylinder_pipe_flow_water"] = cylinder_pipe_flow_dataset
        # remove cylinder_pipe_flow_water from dataset_list
        dataset_list.remove("cylinder_pipe_flow_water")

    if "cylinder_sym_flow_water" in dataset_list:
        cylinder_symmetry_flow_dataset = PhysicsDataset(
            data_dir=data_dir / f"cylinder_sym_flow_water/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["cylinder_sym_flow_water"] = cylinder_symmetry_flow_dataset
        # remove cylinder_sym_flow_water from dataset_list
        dataset_list.remove("cylinder_sym_flow_water")

    if "object_periodic_flow_water" in dataset_list:
        object_periodic_flow_dataset = PhysicsDataset(
            data_dir=data_dir / f"object_periodic_flow_water/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["object_periodic_flow_water"] = object_periodic_flow_dataset
        # remove object_periodic_flow_water from dataset_list
        dataset_list.remove("object_periodic_flow_water")

    if "object_sym_flow_water" in dataset_list:
        object_symmetry_flow_dataset = PhysicsDataset(
            data_dir=data_dir / f"object_sym_flow_water/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["object_sym_flow_water"] = object_symmetry_flow_dataset
        # remove object_sym_flow_water from dataset_list
        dataset_list.remove("object_sym_flow_water")

    if "object_sym_flow_air" in dataset_list:
        object_symmetry_flow_dataset = PhysicsDataset(
            data_dir=data_dir / f"object_sym_flow_air/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["object_sym_flow_air"] = object_symmetry_flow_dataset
        # remove object_sym_flow_air from dataset_list
        dataset_list.remove("object_sym_flow_air")

    if "heated_object_pipe_flow_air" in dataset_list:
        heated_flow_dataset = PhysicsDataset(
            data_dir=data_dir / f"heated_object_pipe_flow_air/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["heated_object_pipe_flow_air"] = heated_flow_dataset
        # remove heated_object_pipe_flow_air from dataset_list
        dataset_list.remove("heated_object_pipe_flow_air")

    if "cooled_object_pipe_flow_air" in dataset_list:
        cooled_flow_dataset = PhysicsDataset(
            data_dir=data_dir / f"cooled_object_pipe_flow_air/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["cooled_object_pipe_flow_air"] = cooled_flow_dataset
        # remove cooled_object_pipe_flow_air from dataset_list
        dataset_list.remove("cooled_object_pipe_flow_air")

    if "rayleigh_benard_obstacle" in dataset_list:
        rayleigh_benard_obstacle_dataset = PhysicsDataset(
            data_dir=data_dir / f"rayleigh_benard_obstacle/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["rayleigh_benard_obstacle"] = rayleigh_benard_obstacle_dataset
        # remove rayleigh_benard_obstacle from dataset_list
        dataset_list.remove("rayleigh_benard_obstacle")

    if "twophase_flow" in dataset_list:
        twophase_flow_dataset = PhysicsDataset(
            data_dir=data_dir / f"twophase_flow/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
        )
        datasets["twophase_flow"] = twophase_flow_dataset
        # remove twophase_flow from dataset_list
        dataset_list.remove("twophase_flow")

    if len(dataset_list) > 0:
        raise ValueError(f"Unknown datasets: {dataset_list}")

    return datasets
