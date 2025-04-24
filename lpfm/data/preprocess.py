"""Preprocess the data
Includes interpolating the data to the target grid, rotating the data,
adding all necessary fields and saving the data to disk.
"""

import h5py
import shutil
from pathlib import Path
import numpy as np
import torch
from einops import rearrange


def swap_axes(data: np.ndarray) -> np.ndarray:
    """Swap the axes of the data."""
    # data is of shape (traj, t, h, w, c) or (traj, t, h, w)
    # in this dataset, the x axis is smaller than the y axis
    # so we need to swap the axes
    if data.ndim == 5:
        data = np.swapaxes(data, 2, 3)
    elif data.ndim == 4:
        data = np.swapaxes(data, 2, 3)
    return data


def swap_velocity_components(data: np.ndarray) -> np.ndarray:
    """Swap the velocity components."""
    vel_x = data[..., 0].copy()
    vel_y = data[..., 1].copy()
    data[..., 0] = vel_y
    data[..., 1] = vel_x
    return data


def interpolate_data(data: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Interpolate the data to the target grid."""
    # data is of shape (traj, t, h, w) or (traj, t, h, w, c)
    if data.ndim == 4:
        traj, t, h, w = data.shape
    elif data.ndim == 5:
        traj, t, h, w, c = data.shape

    data = torch.from_numpy(data)
    if data.ndim == 5:
        data_r = rearrange(data, "traj t h w c -> (traj t) c h w")
        interpolated_data = torch.nn.functional.interpolate(
            data_r, size=target_shape, mode="bicubic", align_corners=False
        )
    else:
        interpolated_data = torch.nn.functional.interpolate(
            data, size=target_shape, mode="bicubic", align_corners=False
        )

    if data.ndim == 5:
        interpolated_data = rearrange(
            interpolated_data,
            "(traj t) c h w -> traj t h w c",
            traj=traj,
            t=t,
        )
    return interpolated_data.numpy()


def transform_fn_data(
    name: str,
    data: np.ndarray,
    target_shape: tuple[int, int],
    swap: bool = False,
) -> np.ndarray:
    """Transform the data."""

    fields_names = ["velocity", "pressure", "temperature", "density"]
    for field_name in fields_names:
        if field_name in name:
            if swap:
                data = swap_axes(data)
                if field_name == "velocity":
                    data = swap_velocity_components(data)
            data = interpolate_data(data, target_shape)

    return data


def add_missing_datasets(
    t0_group: h5py.Group, num_traj: int, num_time: int, target_shape: tuple[int, int]
):
    """Add missing fields to the group."""
    names = ["pressure", "temperature", "density"]
    zero_data = np.zeros((num_traj, num_time, *target_shape), dtype=np.float32)
    # get the keys of the group
    keys = list(t0_group.keys())
    # check if the keys are in the keys
    for name in names:
        if name not in keys:
            # create a new dataset
            dset = t0_group.create_dataset(name, data=zero_data)
            dset.attrs["dim_varying"] = [True, True]
            dset.attrs["sample_varying"] = True
            dset.attrs["time_varying"] = True


def handle_boundary_conditions(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    target_shape: tuple[int, int],
    swap: bool,
):
    """Handle the boundary conditions."""

    x_mask = np.zeros(target_shape[0], dtype=np.float32)
    y_mask = np.zeros(target_shape[1], dtype=np.float32)
    x_mask[0] = 1
    x_mask[-1] = 1
    y_mask[0] = 1
    y_mask[-1] = 1
    x_values = np.zeros(target_shape[0], dtype=np.float32)
    y_values = np.zeros(target_shape[1], dtype=np.float32)

    group_data = {"attributes": {}, "data": {}}
    for subgroup_name, sub_group in src_group.items():
        group_data["attributes"][subgroup_name] = {}
        for attr_name, attr_value in sub_group.attrs.items():
            group_data["attributes"][subgroup_name][attr_name] = attr_value

        if "x" in subgroup_name and swap:
            # replace x in name
            subgroup_name_new = subgroup_name.replace("x", "y")
            group_data["data"][subgroup_name_new] = {"mask": y_mask, "values": y_values}
            group_data["attributes"][subgroup_name_new] = group_data["attributes"].pop(
                subgroup_name
            )
            group_data["attributes"][subgroup_name_new]["associated_dims"] = ["y"]
        elif "x" in subgroup_name and not swap:
            group_data["data"][subgroup_name] = {"mask": x_mask, "values": x_values}
            group_data["attributes"][subgroup_name]["associated_dims"] = ["x"]
        elif "y" in subgroup_name and swap:
            # replace y in name
            subgroup_name_new = subgroup_name.replace("y", "x")
            group_data["data"][subgroup_name_new] = {"mask": x_mask, "values": x_values}
            group_data["attributes"][subgroup_name_new] = group_data["attributes"].pop(
                subgroup_name
            )
            group_data["attributes"][subgroup_name_new]["associated_dims"] = ["x"]
        elif "y" in subgroup_name and not swap:
            group_data["data"][subgroup_name] = {"mask": y_mask, "values": y_values}
            group_data["attributes"][subgroup_name]["associated_dims"] = ["y"]

    # add the data to the dst_group
    for subgroup_name, data in group_data["data"].items():
        # create subgroup
        sub_group = dst_group.create_group(subgroup_name)
        # add attributes to subgroup
        for attr_name, attr_value in group_data["attributes"][subgroup_name].items():
            sub_group.attrs[attr_name] = attr_value
        # add data to subgroup
        sub_group.create_dataset("values", data=data["values"])
        sub_group.create_dataset("mask", data=data["mask"])


def add_buoyancy_dataset(t0_group: h5py.Group, target_shape: tuple[int, int]):
    """Add the buoyancy dataset."""

    # get the buoyancy dataset
    buoyancy_dataset = t0_group["buoyancy"]
    # get the data
    data = buoyancy_dataset[()]
    # interpolate the data
    density_data = interpolate_data(data, target_shape)

    dset = t0_group.create_dataset("density", data=density_data)
    dset.attrs["dim_varying"] = [True, True]
    dset.attrs["sample_varying"] = True
    dset.attrs["time_varying"] = True

    # get the pressure dataset
    pressure_dataset = t0_group["pressure"]
    # get the data
    p_data = pressure_dataset[()]
    temp_data = p_data / density_data

    dset = t0_group.create_dataset("temperature", data=temp_data)
    dset.attrs["dim_varying"] = [True, True]
    dset.attrs["sample_varying"] = True
    dset.attrs["time_varying"] = True

    # delete the buoyancy dataset
    del t0_group["buoyancy"]


def add_momentum_dataset(
    t1_group: h5py.Group, t0_group: h5py.Group, target_shape: tuple[int, int]
):
    """Add the momentum dataset."""

    # get the buoyancy dataset
    momentum_dataset = t1_group["momentum"]
    # get the data
    data = momentum_dataset[()]
    # interpolate the data
    momentum_data = interpolate_data(data, target_shape)
    # get the density dataset
    density_dataset = t0_group["density"]
    # get the data
    density_data = density_dataset[()]
    velocity_data = momentum_data / density_data

    dset = t1_group.create_dataset("velocity", data=velocity_data)
    dset.attrs["dim_varying"] = [True, True]
    dset.attrs["sample_varying"] = True
    dset.attrs["time_varying"] = True

    # delete the momentum dataset
    del t1_group["momentum"]


def process_hdf5(
    input_path: Path,
    output_path: Path,
    swap: bool = False,
):
    """Copy HDF5 file contents to a new file

    Parameters
    ----------
    input_path : Path
        Path to the input HDF5 file
    output_path : Path
        Path where the new HDF5 file should be created


    Returns
    -------
    None
    """
    print(f"Processing {input_path}")
    target_shape = (256, 128)

    with h5py.File(input_path, "r") as src_file:
        with h5py.File(output_path, "w") as dst_file:
            num_traj = src_file.attrs["n_trajectories"]
            time = src_file["dimensions"]["time"][()]
            num_time = len(time)

            def copy_attr(src_attr, dst_attr):
                for attr_name, attr_value in src_attr.items():
                    dst_attr[attr_name] = attr_value

            # Copy root attributes
            copy_attr(src_file.attrs, dst_file.attrs)

            # Recursively copy all groups and datasets
            def copy_group(src_group, dst_group):
                for name, item in src_group.items():
                    if isinstance(item, h5py.Group):
                        new_group = dst_group.create_group(name)
                        if name == "boundary_conditions":
                            handle_boundary_conditions(
                                item, new_group, target_shape, swap
                            )
                        else:
                            # Create new group and copy its attributes
                            copy_attr(item.attrs, new_group.attrs)
                            # Recursively copy contents of the group
                            copy_group(item, new_group)

                    elif isinstance(item, h5py.Dataset):
                        if name == "x":
                            new_x = np.arange(0, target_shape[0], dtype=np.float32)
                            # Copy dataset and its attributes
                            new_dataset = dst_group.create_dataset("x", data=new_x)
                            copy_attr(item.attrs, new_dataset.attrs)
                        elif name == "y":
                            new_y = np.arange(0, target_shape[1], dtype=np.float32)
                            # Copy dataset and its attributes
                            new_dataset = dst_group.create_dataset("y", data=new_y)
                            copy_attr(item.attrs, new_dataset.attrs)
                        else:
                            # Get the dataset data
                            data = item[()]
                            # Apply transformation if provided
                            data = transform_fn_data(name, data, target_shape, swap)
                            # Copy dataset and its attributes
                            new_dataset = dst_group.create_dataset(name, data=data)
                            copy_attr(item.attrs, new_dataset.attrs)

            # Start copying from root group
            copy_group(src_file, dst_file)

            # finally add the missing datasets
            t0_group = dst_file["t0_fields"]
            t1_group = dst_file["t1_fields"]
            # add_buoyancy_dataset(t0_group, target_shape)
            # add_momentum_dataset(t1_group, t0_group, target_shape)
            add_missing_datasets(t0_group, num_traj, num_time, target_shape)


if __name__ == "__main__":
    base_path = Path(
        "/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/data/datasets"
    )
    dataset = Path("cylinder_sym_flow_water")
    dataset_dir = base_path / dataset

    # make a safety copy of the whole directory and its contents
    shutil.copytree(dataset_dir, dataset_dir.parent / f"{dataset_dir.stem}_copy")

    swap = False

    for file in list(dataset_dir.glob("**/*.hdf5")):
        new_name = file.parent / f"{file.stem}_new.hdf5"
        process_hdf5(file, new_name, swap)
        # remove old file
        file.unlink()
        # rename new file
        new_name.rename(file)
