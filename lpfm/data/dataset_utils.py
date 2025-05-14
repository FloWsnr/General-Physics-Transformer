from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import (
    default_collate,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
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
    data_config: dict,
    train_config: dict,
    split: str = "train",
    is_distributed: bool = False,
    shuffle: bool = True,
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
    shuffle : bool
        Whether to shuffle the dataset
    """
    seed = train_config["seed"]
    datasets = get_datasets(data_config, split=split)
    super_dataset = SuperDataset(
        datasets,
        max_samples_per_ds=data_config["max_samples_per_ds"],
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
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=train_config["prefetch_factor"],
    )

    return dataloader


def get_dataloader_val(
    data_config: dict,
    train_config: dict,
    is_distributed: bool = False,
) -> DataLoader:
    """Get a dataloader for the validation dataset."""
    seed = train_config["seed"]
    datasets = get_dt_datasets(data_config, split="val")

    # get the number of samples per dataset
    samples_per_ds = [len(dataset) for dataset in datasets.values()]
    # get the number of samples per dataset for the validation set using the val_frac_samples
    val_frac_samples = float(train_config["val_frac_samples"])
    val_num_samples_per_ds = [
        int(val_frac_samples * samples_per_ds[i]) for i in range(len(samples_per_ds))
    ]

    super_dataset = SuperDataset(
        datasets, max_samples_per_ds=val_num_samples_per_ds, seed=seed
    )

    if is_distributed:
        sampler = DistributedSampler(super_dataset, seed=seed, shuffle=False)
    else:
        sampler = SequentialSampler(super_dataset)

    dataloader = DataLoader(
        dataset=super_dataset,
        batch_size=train_config["batch_size"],
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
    use_normalization = data_config.get("use_normalization", False)
    datasets = {}
    dataset_list: list[str] = data_config["datasets"].copy()

    for dataset_name in dataset_list:
        dataset = PhysicsDataset(
            data_dir=data_dir / f"{dataset_name}/data/{split_name}",
            split=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
            full_trajectory_mode=full_traj,
            max_rollout_steps=max_rollout_steps,
            include_field_names=include_field_names,
            use_normalization=use_normalization,
            normalization_path=data_dir / f"{dataset_name}/data/stats.yaml",
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
    dt_strides = data_config["dt_stride"]
    if isinstance(dt_strides, int):
        dt_strides = [dt_strides]
    all_datasets = {}
    for dt_stride in dt_strides:
        data_config["dt_stride"] = dt_stride
        datasets = get_datasets(data_config, split)
        for dataset_name, dataset in datasets.items():
            all_datasets[f"{dataset_name}_dt_{dt_stride}"] = dataset

    return all_datasets
