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

    # Replace NaNs with 0
    x = torch.where(
        torch.isnan(x),
        torch.zeros_like(x),
        x,
    )
    y = torch.where(
        torch.isnan(y),
        torch.zeros_like(y),
        y,
    )
    return x, y


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

    Returns a tuple of input and target field tensors.

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
    channels_first: bool
        Whether to have (time, channels, height, width) or (time, height, width, channels)
        By default (time, height, width, channels)
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
        channels_first: bool = False,
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
        self.channels_first = channels_first

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        data = super().__getitem__(index)  # returns (time, h, w, c)
        x = data["input_fields"]
        y = data["output_fields"]
        if self.channels_first:
            x = einops.rearrange(x, "time h w c -> time c h w")
            y = einops.rearrange(y, "time h w c -> time c h w")
        return x, y


class SuperDataset:
    """Wrapper around a list of datasets.

    Allows to concatenate multiple datasets and randomly sample from them.

    Parameters
    ----------
    datasets : list[WellDataset]
        List of datasets to concatenate
    out_shape : tuple[int, int]
        Output shape (h, w) of the concatenated dataset.
        This is needed to account for the different shapes of the datasets.
    n_channels : int
        Number of channels of the concatenated dataset. Should be the largest number of channels in the datasets.
        Samples with less than n_channels will be padded with zeros channels.
    """

    def __init__(
        self, datasets: list[WellDataset], out_shape: tuple[int, int], n_channels: int
    ):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.out_shape = out_shape
        self.n_channels = n_channels

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        for i, length in enumerate(self.lengths):
            if index < length:
                x, y = self.datasets[i][index]  # (time, h, w, n_channels)
                break
            index -= length

        x = einops.rearrange(x, "time h w c -> time c h w")
        y = einops.rearrange(y, "time h w c -> time c h w")
        # Reshape to out_shape
        x = torch.nn.functional.interpolate(
            x, size=self.out_shape, mode="bilinear", align_corners=False
        )
        y = torch.nn.functional.interpolate(
            y, size=self.out_shape, mode="bilinear", align_corners=False
        )
        x = einops.rearrange(x, "time c h w -> time h w c")
        y = einops.rearrange(y, "time c h w -> time h w c")

        # if x,y has less than n_channels, add channels with zeros
        if x.shape[-1] < self.n_channels:
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        *x.shape[:-1],
                        self.n_channels - x.shape[-1],
                    ),
                ],
                dim=-1,
            )
        if y.shape[-1] < self.n_channels:
            y = torch.cat(
                [
                    y,
                    torch.zeros(
                        *y.shape[:-1],
                        self.n_channels - y.shape[-1],
                    ),
                ],
                dim=-1,
            )

        return x, y


# class SuperDataloader:
#     """Wrapper around DataLoader.

#     Allows to concatenate multiple DataLoaders and randomly sample from them.
#     """

#     def __init__(self, dataloaders: list[DataLoader]):
#         self.dataloaders = dataloaders
#         self.iterators = None
#         self.length = sum(len(dataloader) for dataloader in dataloaders)

#     def __iter__(self):
#         # Create fresh iterators for each dataloader
#         self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
#         return self

#     def __next__(self):
#         if not self.iterators:
#             raise StopIteration

#         # Randomly select a dataloader
#         loader_idx = torch.randint(0, len(self.dataloaders), (1,)).item()

#         try:
#             # Get next batch from the selected dataloader
#             return next(self.iterators[loader_idx])
#         except StopIteration:
#             # If this dataloader is exhausted, remove it and try again
#             self.iterators.pop(loader_idx)
#             self.dataloaders.pop(loader_idx)

#             if not self.dataloaders:
#                 raise StopIteration

#             return next(self)

#     def __len__(self):
#         return self.length
