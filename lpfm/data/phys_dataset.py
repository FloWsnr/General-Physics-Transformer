from typing import Optional
from pathlib import Path

import torch
from the_well.data.augmentation import Compose

from lpfm.data.well_dataset import WellDataset


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
        dt_stride: int | list[int] = 1,
        transform: Optional[Compose] = None,
        channels_first: bool = False,
        include_field_names: dict[str, list[str]] = {},
        full_trajectory_mode: bool = False,
        max_rollout_steps: int = 10000,
    ):
        if isinstance(dt_stride, list):
            min_dt_stride = dt_stride[0]
            max_dt_stride = dt_stride[1]
        else:
            min_dt_stride = dt_stride
            max_dt_stride = dt_stride

        super().__init__(
            path=str(data_dir),
            well_split_name=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            normalization_path=str(normalization_path),
            use_normalization=use_normalization,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            transform=transform,
            include_field_names=include_field_names,
            full_trajectory_mode=full_trajectory_mode,
            max_rollout_steps=max_rollout_steps,
        )
        self.channels_first = channels_first

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
        return x, y


class SuperDataset:
    """Wrapper around a list of datasets.

    Allows to concatenate multiple datasets and randomly sample from them.

    Parameters
    ----------
    datasets : dict[str, PhysicsDataset]
        Dictionary of datasets to concatenate
    out_shape : tuple[int, int]
        Output shape (h, w) of the concatenated dataset.
        This is needed to account for the different shapes of the datasets.
    max_samples_per_ds : Optional[int]
        Maximum number of samples to sample from each dataset.
        If None, uses all samples from each dataset.
        By default None.
    seed : Optional[int]
        Random seed for reproducibility.
        By default None.
    """

    def __init__(
        self,
        datasets: dict[str, PhysicsDataset],
        out_shape: tuple[int, int],
        max_samples_per_ds: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.datasets = list(datasets.values())
        self.dataset_names = list(datasets.keys())
        self.out_shape = out_shape
        self.max_samples_per_ds = max_samples_per_ds

        # Initialize random number generator
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # Generate random indices for each dataset if max_samples_per_ds is specified
        self.dataset_indices = []
        for dataset in self.datasets:
            if max_samples_per_ds is not None and len(dataset) > max_samples_per_ds:
                indices = torch.randperm(len(dataset), generator=self.rng)[
                    :max_samples_per_ds
                ]
                self.dataset_indices.append(indices)
            else:
                self.dataset_indices.append(None)

        # Calculate lengths based on either max_samples_per_ds or full dataset length
        self.lengths = [
            min(max_samples_per_ds, len(dataset))
            if max_samples_per_ds is not None
            else len(dataset)
            for dataset in self.datasets
        ]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        for i, length in enumerate(self.lengths):
            if index < length:
                if self.dataset_indices[i] is not None:
                    # Use random index if available
                    actual_index = self.dataset_indices[i][index]
                else:
                    actual_index = index
                x, y = self.datasets[i][actual_index]  # (time, h, w, n_channels)
                break
            index -= length
        return x, y
