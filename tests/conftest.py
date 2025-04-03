import pytest
from pathlib import Path

import h5py
import numpy as np

from the_well.data.datasets import WellDataset


def write_dummy_data(filename: Path):
    """Create dummy data following the Well formating for testing purposes
    Copied from the_well package
    """
    # Create dummy data
    param_a = 0.25
    param_b = 0.75
    dataset_name = "dummy_dataset"
    grid_type = "cartesian"
    n_spatial_dims = 2
    n_trajectories = 2
    dim_x = 32
    dim_y = 32
    dim_t = 10
    n_dim = 2
    x = np.linspace(0, 1, dim_x, dtype=np.float32)
    y = np.linspace(0, 1, dim_y, dtype=np.float32)
    t = np.linspace(0, 1, dim_t, dtype=np.float32)
    x_peridocity_mask = np.zeros_like(x).astype(bool)
    x_peridocity_mask[0] = x_peridocity_mask[-1]
    y_peridocity_mask = np.zeros_like(y).astype(bool)
    y_peridocity_mask[0] = y_peridocity_mask[-1]
    t1_field_values = np.random.rand(n_trajectories, dim_t, dim_x, dim_y, n_dim).astype(
        np.float32
    )
    t0_constant_field_values = np.random.rand(n_trajectories, dim_x, dim_y).astype(
        np.float32
    )
    t0_variable_field_values = np.random.rand(
        n_trajectories, dim_t, dim_x, dim_y
    ).astype(np.float32)

    time_varying_scalar_values = np.random.rand(dim_t)

    # Write the data in the HDF5 file
    with h5py.File(filename, "w") as file:
        # Attributes
        file.attrs["a"] = param_a
        file.attrs["b"] = param_b
        file.attrs["dataset_name"] = dataset_name
        file.attrs["grid_type"] = grid_type
        file.attrs["n_spatial_dims"] = n_spatial_dims
        file.attrs["n_trajectories"] = n_trajectories
        file.attrs["simulation_parameters"] = ["a", "b"]
        # Boundary Conditions
        group = file.create_group("boundary_conditions")
        for key, val in zip(
            ["x_periodic", "y_periodic"], [x_peridocity_mask, y_peridocity_mask]
        ):
            sub_group = group.create_group(key)
            sub_group.attrs["associated_dims"] = key[0]
            sub_group.attrs["associated_fields"] = []
            sub_group.attrs["bc_type"] = "PERIODIC"
            sub_group.attrs["sample_varying"] = False
            sub_group.attrs["time_varying"] = False
            sub_group.create_dataset("mask", data=val)
        # Dimensions
        group = file.create_group("dimensions")
        group.attrs["spatial_dims"] = ["x", "y"]
        for key, val in zip(["time", "x", "y"], [t, x, y]):
            group.create_dataset(key, data=val)
            group[key].attrs["sample_varying"] = False
        # Scalars
        group = file.create_group("scalars")
        group.attrs["field_names"] = ["a", "b", "time_varying_scalar"]
        for key, val in zip(["a", "b"], [param_a, param_b]):
            group.create_dataset(key, data=np.array(val))
            group[key].attrs["time_varying"] = False
            group[key].attrs["sample_varying"] = False
        ## Time varying
        dset = group.create_dataset(
            "time_varying_scalar", data=time_varying_scalar_values
        )
        dset.attrs["time_varying"] = True
        dset.attrs["sample_varying"] = False

        # Fields
        ############### T0 Fields ###############
        group = file.create_group("t0_fields")
        group.attrs["field_names"] = [
            "constant_field",
            "variable_field1",
            "variable_field2",
        ]
        # Add a constant field regarding time
        dset = group.create_dataset("constant_field", data=t0_constant_field_values)
        dset.attrs["dim_varying"] = [True, True]
        dset.attrs["sample_varying"] = True
        dset.attrs["time_varying"] = False

        dset = group.create_dataset("variable_field1", data=t0_variable_field_values)
        dset.attrs["dim_varying"] = [True, True]
        dset.attrs["sample_varying"] = True
        dset.attrs["time_varying"] = True

        dset = group.create_dataset("variable_field2", data=t0_variable_field_values)
        dset.attrs["dim_varying"] = [True, True]
        dset.attrs["sample_varying"] = True
        dset.attrs["time_varying"] = True

        ############### T1 Fields ###############
        # Add a field varying both in time and space
        group = file.create_group("t1_fields")
        group.attrs["field_names"] = ["field1", "field2"]
        dset = group.create_dataset("field1", data=t1_field_values)
        dset.attrs["dim_varying"] = [True, True]
        dset.attrs["sample_varying"] = True
        dset.attrs["time_varying"] = True

        dset = group.create_dataset("field2", data=t1_field_values)
        dset.attrs["dim_varying"] = [True, True]
        dset.attrs["sample_varying"] = True
        dset.attrs["time_varying"] = True

        ############# T2 Fields ###############
        group = file.create_group("t2_fields")
        group.attrs["field_names"] = []


@pytest.fixture(scope="session")
def dummy_datapath(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create dummy data for testing."""
    data_dir = tmp_path_factory.mktemp("train")
    file = data_dir / "dummy_dataset.hdf5"
    write_dummy_data(file)
    return file


@pytest.fixture(scope="session")
def dummy_dataset(dummy_datapath: Path) -> WellDataset:
    """Create a dummy dataset for testing."""
    return WellDataset(path=dummy_datapath.parent)
