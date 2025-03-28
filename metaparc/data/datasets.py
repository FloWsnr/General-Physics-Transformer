from pathlib import Path
from typing import Optional

import einops

import torch
from torch.utils.data import default_collate, DataLoader

from the_well.data.datasets import WellDataset
from the_well.data.augmentation import (
    Compose,
    RandomAxisFlip,
    RandomAxisRoll,
    RandomAxisPermute,
    # NOTE: Image Resize should come here as well
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


def collate_fn(data: list[dict]) -> torch.Tensor:
    """Collate function for the dataset.
    Get dict with input and output fields from the WellDataset.
    The fields are of shape (Time steps, H, W, C)
    We want to get a batch of shape (B, Time steps & C, H, W)

    Parameters
    ----------
    data : dict
        Dict with input and output fields from the WellDataset

    Returns
    -------
    batch : torch.Tensor
        Batch of shape (B, Time steps & C, H, W)
    """

    batch = default_collate(data)
    input_fields = batch["input_fields"]
    output_fields = batch["output_fields"]

    # rearrange to (B, Time steps & C, H, W)
    input_fields = einops.rearrange(
        input_fields, "batch time h w c -> batch (time c) h w"
    )
    output_fields = einops.rearrange(
        output_fields, "batch time h w c -> batch (time c) h w"
    )

    # Replace NaNs with 0
    batch["input_fields"] = torch.where(
        torch.isnan(input_fields),
        torch.zeros_like(input_fields),
        input_fields,
    )
    batch["output_fields"] = torch.where(
        torch.isnan(output_fields),
        torch.zeros_like(output_fields),
        output_fields,
    )
    return batch


def get_dataloader(
    dataset: WellDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


class PhysicsDataset(WellDataset):
    """Wrapper around the WellDataset.

    Enables data augmentation.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory (e.g. "data/physics_data/train")
    n_steps_input : int
        Number of input time steps per sample
        By default 1
    n_steps_output : int
        Number of output time steps per sample
        By default 1
    split : str
        Split to load ("train", "val", "test")
        By default "train"
    normalization_path: Optional[Path]
        Path to the normalization file (e.g. "data/physics_data/normalization.yaml")
        By default None
    use_normalization: bool
        Whether to use normalization
        By default False
    dt_stride: int
        Time step stride between samples
        By default 1
    transform: Optional[Compose]
        Transform to apply to the data
        By default None
    """

    def __init__(
        self,
        data_dir: Path,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        split: str = "train",
        normalization_path: Path = Path("stats.yaml"),
        use_normalization: bool = False,
        dt_stride: int = 1,
        transform: Optional[Compose] = None,
    ):
        super().__init__(
            path=str(data_dir),
            well_split_name=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            normalization_path=str(normalization_path),
            use_normalization=use_normalization,
            min_dt_stride=dt_stride,
            max_dt_stride=dt_stride,
            transform=transform,
        )

    def __getitem__(self, index):
        data = super().__getitem__(index)  # returns (time, h, w, c)
        return data


class SuperDataloader:
    """Wrapper around DataLoader.

    Allows to concatenate multiple DataLoaders and randomly sample from them.
    """

    def __init__(self, dataloaders: list[DataLoader]):
        self.dataloaders = dataloaders
        self.iterators = None
        self.length = sum(len(dataloader) for dataloader in dataloaders)

    def __iter__(self):
        # Create fresh iterators for each dataloader
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        return self

    def __next__(self):
        if not self.iterators:
            raise StopIteration

        # Randomly select a dataloader
        loader_idx = torch.randint(0, len(self.dataloaders), (1,)).item()

        try:
            # Get next batch from the selected dataloader
            return next(self.iterators[loader_idx])
        except StopIteration:
            # If this dataloader is exhausted, remove it and try again
            self.iterators.pop(loader_idx)
            self.dataloaders.pop(loader_idx)

            if not self.dataloaders:
                raise StopIteration

            return next(self)

    def __len__(self):
        return self.length
