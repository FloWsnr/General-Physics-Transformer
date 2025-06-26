from typing import Optional, Any
from pathlib import Path

import torch

from gphyt.data.well_dataset import WellDataset


def zero_field_to_value(x: torch.Tensor, value: float) -> torch.Tensor:
    """Find channels which are all zeros and replace them with a given value.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (T, H, W, C)
    value : float
        Value to replace the zero channels with
    """
    zero_channels = torch.all(x == 0, dim=(0, 1, 2), keepdim=False)
    x[..., zero_channels] = value
    return x


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
    nan_to_zero: bool
        Whether to replace NaNs with 0
        By default True
    flip_x: bool
        Whether to flip the x-axis of the data
        By default False
    flip_y: bool
        Whether to flip the y-axis of the data
        By default False
    """

    def __init__(
        self,
        data_dir: Path,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        use_normalization: bool = False,
        dt_stride: int | list[int] = 1,
        include_field_names: dict[str, list[str]] = {},
        full_trajectory_mode: bool = False,
        max_rollout_steps: int = 10000,
        nan_to_zero: bool = True,
        flip_x: bool = False,
        flip_y: bool = False,
    ):
        self.config = {
            "data_dir": data_dir,
            "n_steps_input": n_steps_input,
            "n_steps_output": n_steps_output,
            "use_normalization": use_normalization,
            "dt_stride": dt_stride,
            "include_field_names": include_field_names,
            "full_trajectory_mode": full_trajectory_mode,
            "max_rollout_steps": max_rollout_steps,
            "nan_to_zero": nan_to_zero,
            "flip_x": flip_x,
            "flip_y": flip_y,
        }

        if isinstance(dt_stride, list):
            min_dt_stride = dt_stride[0]
            max_dt_stride = dt_stride[1]
        else:
            min_dt_stride = dt_stride
            max_dt_stride = dt_stride

        super().__init__(
            path=data_dir,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            include_field_names=include_field_names,
            full_trajectory_mode=full_trajectory_mode,
            max_rollout_steps=max_rollout_steps,
        )
        self.nan_to_zero = nan_to_zero
        self.flip_x = flip_x
        self.flip_y = flip_y

    def copy(self, overwrites: dict[str, Any] = {}) -> "PhysicsDataset":
        """Copy the dataset with optional overwrites.

        Useful for creating a new dataset with slightly different parameters.

        Parameters
        ----------
        overwrites : dict[str, Any]
            Dictionary of overwrites for the config.
        """
        config = self.config.copy()
        config.update(overwrites)
        return PhysicsDataset(
            data_dir=config["data_dir"],
            n_steps_input=config["n_steps_input"],
            n_steps_output=config["n_steps_output"],
            use_normalization=config["use_normalization"],
            dt_stride=config["dt_stride"],
            include_field_names=config["include_field_names"],
            full_trajectory_mode=config["full_trajectory_mode"],
            max_rollout_steps=config["max_rollout_steps"],
            nan_to_zero=config["nan_to_zero"],
            flip_x=config["flip_x"],
            flip_y=config["flip_y"],
        )

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        data = super().__getitem__(index)  # returns (time, h, w, c)
        x = data["input_fields"]
        y = data["output_fields"]

        if self.nan_to_zero:
            x = torch.nan_to_num(x, 0)
            y = torch.nan_to_num(y, 0)

        if self.flip_x:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
            # additionally, velocity vectors need to be flipped
            x[:, :, :, -2] = x[:, :, :, -2] * -1
        if self.flip_y:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
            # additionally, velocity vectors need to be flipped
            x[:, :, :, -1] = x[:, :, :, -1] * -1

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
    max_samples_per_ds : Optional[int | list[int]]
        Maximum number of samples to sample from each dataset.
        If a list, specifies the number of samples for each dataset individually.
        If None, uses all samples from each dataset.
        By default None.

    seed : Optional[int]
        Random seed for reproducibility.
        By default None.
    """

    def __init__(
        self,
        datasets: dict[str, PhysicsDataset],
        max_samples_per_ds: Optional[int | list[int]] = None,
        seed: Optional[int] = None,
    ):
        self.datasets = datasets
        self.dataset_list = list(datasets.values())

        if isinstance(max_samples_per_ds, int):
            self.max_samples_per_ds = [max_samples_per_ds] * len(datasets)
        else:
            self.max_samples_per_ds = max_samples_per_ds

        self.seed = seed

        # Initialize random number generator
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # Generate initial random indices
        self.reshuffle()

    def reshuffle(self):
        """Reshuffle the indices for each dataset.

        This should be called at the start of each epoch to ensure
        a new random subset of samples is used.

        """
        self.dataset_indices = []
        for i, dataset in enumerate(self.dataset_list):
            if (
                self.max_samples_per_ds is not None
                and len(dataset) > self.max_samples_per_ds[i]
            ):
                indices = torch.randperm(len(dataset), generator=self.rng)[
                    : self.max_samples_per_ds[i]
                ]
                self.dataset_indices.append(indices)
            else:
                self.dataset_indices.append(None)

        # Calculate lengths based on either max_samples_per_ds or full dataset length
        self.lengths = [
            min(self.max_samples_per_ds[i], len(dataset))
            if self.max_samples_per_ds is not None
            else len(dataset)
            for i, dataset in enumerate(self.dataset_list)
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
                x, y = self.dataset_list[i][actual_index]  # (time, h, w, n_channels)
                break
            index -= length
        return x, y
