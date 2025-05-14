"""
Copy of the WellDataset class from the_well.data.datasets.py

This is a modified version of the WellDataset class that is used to load the data from the HDF5 files.

By: Florian Wiesner
Date: 2025-04-03
"""

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
)
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


def raw_steps_to_possible_sample_t0s(
    total_steps_in_trajectory: int,
    n_steps_input: int,
    n_steps_output: int,
    dt_stride: int,
):
    """Given the total number of steps in a trajectory returns the number of samples that can be taken from the
      trajectory such that all samples have at least n_steps_input + n_steps_output steps with steps separated
      by dt_stride.

    ex1: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 1
        Possible samples are: [0, 1], [1, 2], [2, 3], [3, 4]
    ex2: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 2
        Possible samples are: [0, 2], [1, 3], [2, 4]
    ex3: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 3
        Possible samples are: [0, 3], [1, 4]
    ex4: total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 1, dt_stride = 2
        Possible samples are: [0, 2, 4]

    """
    elapsed_steps_per_sample = 1 + dt_stride * (
        n_steps_input + n_steps_output - 1
    )  # Number of steps needed for sample
    return max(0, total_steps_in_trajectory - elapsed_steps_per_sample + 1)


def maximum_stride_for_initial_index(
    time_idx: int,
    total_steps_in_trajectory: int,
    n_steps_input: int,
    n_steps_output: int,
):
    """Given the total number of steps in a file and the current step returns the maximum stride
    that can be taken from the file such that all samples have at least n_steps_input + n_steps_output steps with a stride of
      dt_stride
    """
    used_steps_per_sample = n_steps_input + n_steps_output
    return max(
        0,
        int((total_steps_in_trajectory - time_idx - 1) // (used_steps_per_sample - 1)),
    )


# Boundary condition codes
class BoundaryCondition(Enum):
    WALL = 0
    OPEN = 1
    PERIODIC = 2
    SYMMETRIC = 3


def flatten_field_names(metadata, include_constants=True):
    flat_field_names = itertools.chain(*metadata.field_names.values())
    flat_constant_field_names = itertools.chain(*metadata.constant_field_names.values())

    if include_constants:
        return [*flat_field_names, *flat_constant_field_names]
    else:
        return [*flat_field_names]


@dataclass
class WellMetadata:
    """Dataclass to store metadata for each dataset."""

    dataset_name: str
    n_spatial_dims: int
    spatial_resolution: Tuple[int, ...]
    scalar_names: List[str]
    constant_scalar_names: List[str]
    field_names: Dict[int, List[str]]
    constant_field_names: Dict[int, List[str]]
    boundary_condition_types: List[str]
    n_files: int
    n_trajectories_per_file: List[int]
    n_steps_per_trajectory: List[int]
    grid_type: str = "cartesian"

    @property
    def n_scalars(self) -> int:
        return len(self.scalar_names)

    @property
    def n_constant_scalars(self) -> int:
        return len(self.constant_scalar_names)

    @property
    def n_fields(self) -> int:
        return sum(map(len, self.field_names.values()))

    @property
    def n_constant_fields(self) -> int:
        return sum(map(len, self.constant_field_names.values()))

    @property
    def sample_shapes(self) -> Dict[str, List[int]]:
        return {
            "input_fields": [*self.spatial_resolution, self.n_fields],
            "output_fields": [*self.spatial_resolution, self.n_fields],
            "constant_fields": [*self.spatial_resolution, self.n_constant_fields],
            "input_scalars": [self.n_scalars],
            "output_scalars": [self.n_scalars],
            "constant_scalars": [self.n_constant_scalars],
            "space_grid": [*self.spatial_resolution, self.n_spatial_dims],
        }


class TrajectoryData(TypedDict):
    variable_fields: Dict[int, Dict[str, torch.Tensor]]
    constant_fields: Dict[int, Dict[str, torch.Tensor]]
    variable_scalars: Dict[str, torch.Tensor]
    constant_scalars: Dict[str, torch.Tensor]
    boundary_conditions: Optional[torch.Tensor]
    space_grid: Optional[torch.Tensor]
    time_grid: Optional[torch.Tensor]


@dataclass
class TrajectoryMetadata:
    dataset: "WellDataset"
    file_idx: int
    sample_idx: int
    time_idx: int
    time_stride: int


class WellDataset(Dataset):
    """
    Generic dataset for any Well data. Returns data in B x T x H [x W [x D]] x C format.

    Train/Test/Valid is assumed to occur on a folder level.

    Takes in path to directory of HDF5 files to construct dset.

    Args:
        path:
            Path to directory of HDF5 files, one of path or well_base_path+well_dataset_name
            must be specified
        normalization_path:
            Path to normalization constants - assumed to be in same format as constructed data.
        include_filters:
            Only include files whose name contains at least one of these strings
        exclude_filters:
            Exclude any files whose name contains at least one of these strings
        use_normalization:
            Whether to normalize data in the dataset
        max_rollout_steps:
            Maximum number of steps to rollout
        n_steps_input:
            Number of steps to include in each sample
        n_steps_output:
            Number of steps to include in y
        min_dt_stride:
            Minimum stride between samples
        max_dt_stride:
            Maximum stride between samples
        flatten_tensors:
            Whether to flatten tensor valued field into channels

        full_trajectory_mode:
            Overrides to return full trajectory starting from t0 instead of samples
                for long run validation.
        min_std:
            Minimum standard deviation for field normalization. If a field standard
            deviation is lower than this value, it is replaced by this value.

        include_field_names : dict[list[str]]
            Dictionary of field names to include in the dataset in that order.
            The keys are the order of the field (t0, t1, t2) and the values are lists of field names.

        return_grounding_frame : int, by default 0
            How many starting frames to return as the grounding frames.
    """

    def __init__(
        self,
        path: str | Path,
        include_filters: List[str] = [],
        exclude_filters: List[str] = [],
        use_normalization: bool = False,
        max_rollout_steps: int = 10000,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        flatten_tensors: bool = True,
        full_trajectory_mode: bool = False,
        min_std: float = 1e-4,
        include_field_names: Dict[int, List[str]] = {},
        return_grounding_frame: int = 0,
    ):
        super().__init__()

        self.data_path = Path(path).resolve()
        self.normalization_path = self.data_path.parent / "stats.yaml"

        if use_normalization:
            with open(self.normalization_path, mode="r") as f:
                stats = yaml.safe_load(f)

            self.means = {
                field: torch.as_tensor(val, dtype=torch.float32)
                for field, val in stats["mean"].items()
            }
            self.stds = {
                field: torch.clip(
                    torch.as_tensor(val, dtype=torch.float32), min=min_std
                )
                for field, val in stats["std"].items()
            }

        # field_names is a list of field names to include in the dataset in that order.
        # If empty, all fields are included.
        self.include_field_names = include_field_names
        self.return_grounding_frame = return_grounding_frame
        # Copy params
        self.use_normalization = use_normalization
        self.include_filters = include_filters
        self.exclude_filters = exclude_filters
        self.max_rollout_steps = max_rollout_steps
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output  # Gets overridden by full trajectory mode
        self.min_dt_stride = min_dt_stride
        self.max_dt_stride = max_dt_stride
        self.flatten_tensors = flatten_tensors
        self.full_trajectory_mode = full_trajectory_mode

        if self.min_dt_stride < self.max_dt_stride and self.full_trajectory_mode:
            raise ValueError(
                "Full trajectory mode not supported with variable stride lengths"
            )
        # Check the directory has hdf5 that meet our exclusion criteria
        sub_files = list(self.data_path.glob("*.h5")) + list(
            self.data_path.glob("*.hdf5")
        )
        # Check filters - only use file if include_filters are present and exclude_filters are not
        if len(self.include_filters) > 0:
            retain_files = []
            for include_string in self.include_filters:
                retain_files += [f for f in sub_files if include_string in f]
            sub_files = retain_files
        if len(self.exclude_filters) > 0:
            for exclude_string in self.exclude_filters:
                sub_files = [f for f in sub_files if exclude_string not in f]
        assert len(sub_files) > 0, "No HDF5 files found in path {}".format(
            self.data_path
        )
        self.files_paths = sub_files
        self.files_paths.sort()
        # Build multi-index
        self.metadata = self._build_metadata()

    def _build_metadata(self):
        """Builds multi-file indices and checks that folder contains consistent dataset"""
        self.n_files = len(self.files_paths)
        self.n_trajectories_per_file = []
        self.n_steps_per_trajectory = []
        self.n_windows_per_trajectory = []
        self.file_index_offsets = [0]  # Used to track where each file starts
        # Things where we just care every file has same value
        size_tuples = set()
        names = set()
        ndims = set()
        bcs = set()
        lowest_steps = 1e9  # Note - we should never have 1e9 steps
        for index, file in enumerate(self.files_paths):
            with h5.File(file, "r") as f:
                grid_type = f.attrs["grid_type"]
                # Run sanity checks - all files should have same ndims, size_tuple, and names
                trajectories = int(f.attrs["n_trajectories"])
                # Number of steps is always last dim of time
                steps = f["dimensions"]["time"].shape[-1]
                size_tuple = [
                    f["dimensions"][d].shape[-1]
                    for d in f["dimensions"].attrs["spatial_dims"]
                ]
                ndims.add(f.attrs["n_spatial_dims"])
                names.add(f.attrs["dataset_name"])
                size_tuples.add(tuple(size_tuple))
                # Fast enough that I'd rather check each file rather than processing extra files before checking
                assert len(names) == 1, "Multiple dataset names found in specified path"
                assert len(ndims) == 1, "Multiple ndims found in specified path"
                assert len(size_tuples) == 1, (
                    "Multiple resolutions found in specified path"
                )

                # Track lowest amount of steps in case we need to use full_trajectory_mode
                lowest_steps = min(lowest_steps, steps)

                windows_per_trajectory = raw_steps_to_possible_sample_t0s(
                    steps, self.n_steps_input, self.n_steps_output, self.min_dt_stride
                )
                assert windows_per_trajectory > 0, (
                    f"{steps} steps is not enough steps for file {file}"
                    f" to allow {self.n_steps_input} input and {self.n_steps_output} output steps"
                    f" with a minimum stride of {self.min_dt_stride}"
                )
                self.n_trajectories_per_file.append(trajectories)
                self.n_steps_per_trajectory.append(steps)
                self.n_windows_per_trajectory.append(windows_per_trajectory)
                self.file_index_offsets.append(
                    self.file_index_offsets[-1] + trajectories * windows_per_trajectory
                )
                # Check BCs
                for bc in f["boundary_conditions"].keys():
                    bcs.add(f["boundary_conditions"][bc].attrs["bc_type"])

                if index == 0:
                    # Populate scalar names
                    self.scalar_names = []
                    self.constant_scalar_names = []

                    for scalar in f["scalars"].attrs["field_names"]:
                        if f["scalars"][scalar].attrs["time_varying"]:
                            self.scalar_names.append(scalar)
                        else:
                            self.constant_scalar_names.append(scalar)

                    # Populate field names
                    self.field_names = {i: [] for i in range(3)}
                    self.constant_field_names = {i: [] for i in range(3)}

                    for i in range(3):
                        ti = f"t{i}_fields"
                        # if _f[ti][field].attrs["symmetric"]:
                        # itertools.combinations_with_replacement
                        ti_field_dims = [
                            "".join(xyz)
                            for xyz in itertools.product(
                                f["dimensions"].attrs["spatial_dims"],
                                repeat=i,
                            )
                        ]

                        for field in f[ti].attrs["field_names"]:
                            for dims in ti_field_dims:
                                field_name = f"{field}_{dims}" if dims else field

                                if f[ti][field].attrs["time_varying"]:
                                    self.field_names[i].append(field_name)
                                else:
                                    self.constant_field_names[i].append(field_name)
        # Full trajectory mode overrides the above and just sets each sample to "full"
        # trajectory where full = min(lowest_steps_per_file, max_rollout_steps)
        if self.full_trajectory_mode:
            self.n_steps_output = (
                lowest_steps // self.min_dt_stride
            ) - self.n_steps_input
            assert self.n_steps_output > 0, (
                f"Full trajectory mode not supported for dataset {names[0]} with {lowest_steps} minimum steps"
                f" and a minimum stride of {self.min_dt_stride} and {self.n_steps_input} input steps"
            )
            self.n_windows_per_trajectory = [1] * self.n_files
            self.n_steps_per_trajectory = [lowest_steps] * self.n_files
            self.file_index_offsets = np.cumsum([0] + self.n_trajectories_per_file)

        # Just to make sure it doesn't put us in file -1
        self.file_index_offsets[0] = -1
        # Remove file caching
        # self.files: List[h5.File | None] = [None for _ in self.files_paths]  # We open file references as they come
        # Dataset length is last number of samples
        self.len = self.file_index_offsets[-1]
        self.n_spatial_dims = int(ndims.pop())  # Number of spatial dims
        self.size_tuple = tuple(map(int, size_tuples.pop()))  # Size of spatial dims
        self.dataset_name = names.pop()  # Name of dataset
        # BCs
        self.num_bcs = len(bcs)  # Number of boundary condition type included in data
        self.bc_types = list(bcs)  # List of boundary condition types

        return WellMetadata(
            dataset_name=self.dataset_name,
            n_spatial_dims=self.n_spatial_dims,
            grid_type=grid_type,
            spatial_resolution=self.size_tuple,
            scalar_names=self.scalar_names,
            constant_scalar_names=self.constant_scalar_names,
            field_names=self.field_names,
            constant_field_names=self.constant_field_names,
            boundary_condition_types=self.bc_types,
            n_files=self.n_files,
            n_trajectories_per_file=self.n_trajectories_per_file,
            n_steps_per_trajectory=self.n_steps_per_trajectory,
        )

    def _pad_axes(
        self,
        field_data: torch.Tensor,
        use_dims: List[bool],
        time_varying: bool = False,
        tensor_order: int = 0,
    ) -> torch.Tensor:
        """Repeats data over axes not used in storage.

        Parameters
        ----------
        field_data : torch.Tensor
            The tensor to pad
        use_dims : List[bool]
            List indicating which dimensions are used in storage
        time_varying : bool, optional
            Whether the field varies in time, by default False
        tensor_order : int, optional
            Order of the tensor, by default 0

        Returns
        -------
        torch.Tensor
            The padded tensor
        """
        # Look at which dimensions currently are not used and tile based on their sizes
        expand_dims = (1,) if time_varying else ()
        expand_dims = expand_dims + tuple(
            [
                self.size_tuple[i] if not use_dim else 1
                for i, use_dim in enumerate(use_dims)
            ]
        )
        expand_dims = expand_dims + (1,) * tensor_order
        return torch.tile(field_data, expand_dims)

    def _reconstruct_fields(
        self, file: h5.File, sample_idx: int, time_idx: int, n_steps: int, dt: int
    ):
        """Reconstruct space fields starting at index sample_idx, time_idx, with
        n_steps and dt stride."""
        variable_fields = {0: {}, 1: {}, 2: {}}
        constant_fields = {0: {}, 1: {}, 2: {}}
        # Iterate through field types and apply appropriate transforms to stack them
        for i, order_fields in enumerate(["t0_fields", "t1_fields", "t2_fields"]):
            # If field_names is not empty, use only the fields in field_names
            if order_fields in self.include_field_names:
                field_names: list = self.include_field_names[order_fields]
            else:
                # if the order_fields is not in include_field_names, but include_field_names is not empty,
                # we need to skip this order_fields
                if len(self.include_field_names) > 0:
                    field_names = []
                # if the order_fields is not in include_field_names, and include_field_names is empty,
                # we use all the fields in the order_fields
                else:
                    field_names = file[order_fields].attrs["field_names"]

            for field_name in field_names:
                field = file[order_fields][field_name]
                use_dims = field.attrs["dim_varying"]
                # Initialize field_data
                field_data = field
                # Index is built gradually since there can be different numbers of leading fields
                multi_index = ()
                if field.attrs["sample_varying"]:
                    multi_index = multi_index + (sample_idx,)
                if field.attrs["time_varying"]:
                    multi_index = multi_index + (
                        slice(time_idx, time_idx + n_steps * dt, dt),
                    )
                field_data = field_data[multi_index]
                field_data = torch.as_tensor(field_data, dtype=torch.float32)
                # Normalize
                if self.use_normalization:
                    if field_name in self.means:
                        field_data = field_data - self.means[field_name]
                    if field_name in self.stds:
                        field_data = field_data / (self.stds[field_name] + 1e-6)
                # Expand dims
                field_data = self._pad_axes(
                    field_data,
                    use_dims,
                    time_varying=field.attrs["time_varying"],
                    tensor_order=i,
                )

                if field.attrs["time_varying"]:
                    variable_fields[i][field_name] = field_data
                else:
                    constant_fields[i][field_name] = field_data

        return (variable_fields, constant_fields)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Find specific file and local index
        file_idx = int(
            np.searchsorted(self.file_index_offsets, index, side="right") - 1
        )  # which file we are on
        windows_per_trajectory = self.n_windows_per_trajectory[file_idx]
        local_idx = index - max(
            self.file_index_offsets[file_idx], 0
        )  # First offset is -1
        sample_idx = local_idx // windows_per_trajectory
        time_idx = local_idx % windows_per_trajectory

        # Open file, get data, and close file
        with h5.File(self.files_paths[file_idx], "r") as file:
            # If we gave a stride range, decide the largest size we can use given the sample location
            dt = self.min_dt_stride
            if self.max_dt_stride > self.min_dt_stride:
                effective_max_dt = maximum_stride_for_initial_index(
                    time_idx,
                    self.n_steps_per_trajectory[file_idx],
                    self.n_steps_input,
                    self.n_steps_output,
                )
                effective_max_dt = min(effective_max_dt, self.max_dt_stride)
                if effective_max_dt > self.min_dt_stride:
                    # Randint is non-inclusive on the upper bound
                    dt = np.random.randint(self.min_dt_stride, effective_max_dt + 1)
            # Fetch the data
            data = {}

            output_steps = min(self.n_steps_output, self.max_rollout_steps)
            data["variable_fields"], data["constant_fields"] = self._reconstruct_fields(
                file,
                sample_idx,
                time_idx,
                self.n_steps_input + output_steps,
                dt,
            )
            # Concatenate fields and scalars
            for key in ("variable_fields", "constant_fields"):
                data[key] = [
                    field.unsqueeze(-1).flatten(-order - 1)
                    for order, fields in data[key].items()
                    for _, field in fields.items()
                ]

                if data[key]:
                    data[key] = torch.concatenate(data[key], dim=-1)
                else:
                    data[key] = torch.tensor([])

            # Input/Output split
            sample = {
                "input_fields": data["variable_fields"][
                    : self.n_steps_input
                ],  # Ti x H x W x C
                "output_fields": data["variable_fields"][
                    self.n_steps_input :
                ],  # To x H x W x C
                "constant_fields": data["constant_fields"],  # H x W x C
            }

        # Return only non-empty keys - maybe change this later
        return {k: v for k, v in sample.items() if v.numel() > 0}

    def __len__(self):
        return self.len

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.data_path}>"
