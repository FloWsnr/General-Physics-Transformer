from pathlib import Path

import torch
from torch.utils.data import (
    default_collate,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler

from lpfm.data.phys_dataset import SuperDataset, PhysicsDataset


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
    data_config: dict,
    seed: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    data_fraction: float = 1.0,
    split: str = "train",
    is_distributed: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """Get a dataloader for the dataset.

    Parameters
    ----------
    data_config : dict
        Configuration for the datasets.
    seed : int
        Seed for the dataset.
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers.
    prefetch_factor : int
        Prefetch factor.
    split : str
        Split to load ("train", "val", "test")
    data_fraction : float
        Fraction of the dataset to use.
    is_distributed : bool
        Whether to use distributed sampling
    shuffle : bool
        Whether to shuffle the dataset
    """
    datasets = get_dt_datasets(data_config, split=split)

    if data_fraction < 1.0:
        # get the number of samples per dataset
        samples_per_ds = [len(dataset) for dataset in datasets.values()]
        # get the number of samples per dataset using the data_fraction
        num_samples_per_ds = [
            int(data_fraction * samples_per_ds[i]) for i in range(len(samples_per_ds))
        ]
    else:
        num_samples_per_ds = None

    super_dataset = SuperDataset(
        datasets, max_samples_per_ds=num_samples_per_ds, seed=seed
    )

    if is_distributed:
        sampler = DistributedSampler(super_dataset, seed=seed, shuffle=shuffle)
    else:
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = RandomSampler(super_dataset, generator=generator)
        else:
            sampler = SequentialSampler(super_dataset)
    dataloader = DataLoader(
        dataset=super_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=prefetch_factor,
        drop_last=True,
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
    use_normalization = data_config.get("use_normalization", False)
    flip_x = data_config.get("flip_x", False)
    datasets = {}
    dataset_list: list[str] = data_config["datasets"].copy()

    for dataset_name in dataset_list:
        dataset = PhysicsDataset(
            data_dir=data_dir / f"{dataset_name}/data/{split_name}",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
            use_normalization=use_normalization,
            flip_x=flip_x,
        )
        datasets[dataset_name] = dataset

    # Check if all dataset names in the list are in the dictionary
    missing_datasets = [name for name in dataset_list if name not in datasets]
    if missing_datasets:
        print(
            f"Warning: The following datasets were not found: {', '.join(missing_datasets)}"
        )

    return datasets


def get_dt_datasets(
    data_config: dict, split: str = "train"
) -> dict[int, dict[str, PhysicsDataset]]:
    """Get the datasets for different dt strides.
    This is useful for evaluation to control the dt.

    Parameters
    ----------
    data_config : dict
        Configuration for the dataset.
    split : str
        Split to load ("train", "val", "test")

    Returns
    -------
    dt_datasets : dict[int, dict[str, PhysicsDataset]]
        Dictionary of datasets for different dt strides.
    """
    dt_strides_bounds = data_config["dt_stride"]
    if isinstance(dt_strides_bounds, int):
        dt_strides = [dt_strides_bounds]
    elif isinstance(dt_strides_bounds, list):
        dt_strides = list(range(dt_strides_bounds[0], dt_strides_bounds[1] + 1))
    else:
        raise ValueError(f"Invalid dt_stride: {dt_strides_bounds}")
    all_datasets = {}
    for dt_stride in dt_strides:
        dt_data_config = data_config.copy()
        dt_data_config["dt_stride"] = dt_stride
        datasets = get_datasets(dt_data_config, split)
        for dataset_name, dataset in datasets.items():
            all_datasets[f"{dataset_name}_dt_{dt_stride}"] = dataset

    return all_datasets
