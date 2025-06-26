"""Crop the first time steps from the data for specific datasets."""

import h5py
import shutil
from pathlib import Path
import numpy as np


def crop_hdf5(
    input_path: Path,
    output_path: Path,
):
    """Crop the first time steps from the data for specific datasets.

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

    field_names = ["velocity", "pressure", "temperature", "density"]

    def copy_attr(src_attr, dst_attr):
        for attr_name, attr_value in src_attr.items():
            dst_attr[attr_name] = attr_value

    print(f"Processing {input_path}")
    num_time_steps = 20

    with h5py.File(input_path, "r") as src_file:
        with h5py.File(output_path, "w") as dst_file:
            # Copy root attributes
            copy_attr(src_file.attrs, dst_file.attrs)

            # Recursively copy all groups and datasets
            def copy_group(src_group, dst_group):
                for name, item in src_group.items():
                    if isinstance(item, h5py.Group):
                        new_group = dst_group.create_group(name)
                        # Create new group and copy its attributes
                        copy_attr(item.attrs, new_group.attrs)
                        # Recursively copy contents of the group
                        copy_group(item, new_group)

                    elif isinstance(item, h5py.Dataset):
                        print("Processing", name)
                        # Get the dataset data
                        if name == "time":
                            data = item[()]
                            # remove the first num_time_steps time steps
                            data = data[num_time_steps:]
                            # Copy dataset and its attributes
                            new_dataset = dst_group.create_dataset(name, data=data)
                            copy_attr(item.attrs, new_dataset.attrs)
                        elif name in field_names:
                            data = item[()]
                            # remove the first num_time_steps time steps
                            data = data[:, num_time_steps:, ...]
                            # Copy dataset and its attributes
                            new_dataset = dst_group.create_dataset(name, data=data)
                            copy_attr(item.attrs, new_dataset.attrs)
                        else:
                            data = item[()]
                            # Copy dataset and its attributes
                            new_dataset = dst_group.create_dataset(name, data=data)
                            copy_attr(item.attrs, new_dataset.attrs)

            # Start copying from root group
            copy_group(src_file, dst_file)


if __name__ == "__main__":
    base_path = Path(
        r"/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/data/datasets"
    )

    dataset_names = [
        "cylinder_sym_flow_water",
        "cylinder_pipe_flow_water",
        "object_periodic_flow_water",
        "object_sym_flow_water",
        "object_sym_flow_air",
        "heated_object_pipe_flow_air",
        "cooled_object_pipe_flow_air",
    ]
    for dataset_name in dataset_names:
        dataset_dir = base_path / dataset_name

        # make a safety copy of the whole directory and its contents
        # print(f"Copying {dataset_dir} to {dataset_dir.parent / f'{dataset_dir.stem}_copy'}")
        # shutil.copytree(dataset_dir, dataset_dir.parent / f"{dataset_dir.stem}_copy")

        for file in list(dataset_dir.glob("**/*.hdf5")):
            if "new" in file.stem:
                print("Skipping", file)
                continue
            new_name = file.parent / f"{file.stem}_new.hdf5"
            crop_hdf5(file, new_name)
            # remove old file
            file.unlink()
            # rename new file
            new_name.rename(file)
