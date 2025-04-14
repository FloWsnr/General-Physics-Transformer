from typing import Optional
from pathlib import Path

import torch
import einops
from the_well.data.augmentation import Compose

from metaparc.data.well_dataset import WellDataset


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
    include_field_names: dict[str, list[str]]
        Dictionary of field names to include in the dataset.
        The keys are the order of the field (t0, t1, t2) and the values are lists of field names.
        By default {}
    length_limit: Optional[int]
        Limit the number of samples in the dataset
        By default None
    full_trajectory_mode: bool
        Whether to use the full trajectory mode of the well dataset.
        This returns full trajectories instead of individual timesteps.
        By default False
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
        include_field_names: dict[str, list[str]] = {},
        length_limit: Optional[int] = None,
        full_trajectory_mode: bool = False,
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
            include_field_names=include_field_names,
            full_trajectory_mode=full_trajectory_mode,
        )
        self.channels_first = channels_first
        self.length_limit = length_limit

        # Agument the length of the dataset to the length limit
        # That way, a dataloader will only sample the specified number of samples
        if self.length_limit is not None:
            if self.length_limit < len(self):
                self.len = self.length_limit

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        data = super().__getitem__(index)  # returns (time, h, w, c)
        x = data["input_fields"]
        y = data["output_fields"]

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

        if self.channels_first:
            x = einops.rearrange(x, "time h w c -> time c h w")
            y = einops.rearrange(y, "time h w c -> time c h w")
        return x, y


class SuperDataset:
    """Wrapper around a list of datasets.

    Allows to concatenate multiple datasets and randomly sample from them.

    Parameters
    ----------
    datasets : list[PhysicsDataset]
        List of datasets to concatenate
    out_shape : tuple[int, int]
        Output shape (h, w) of the concatenated dataset.
        This is needed to account for the different shapes of the datasets.
    """

    def __init__(self, datasets: list[PhysicsDataset], out_shape: tuple[int, int]):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.out_shape = out_shape

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

        return x, y
