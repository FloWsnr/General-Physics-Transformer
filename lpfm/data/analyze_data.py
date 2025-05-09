from pathlib import Path
import h5py
import numpy as np
from typing import Dict, List, Any, Tuple


def get_dataset_paths(data_dir: Path) -> List[Path]:
    """Get paths to all dataset directories.

    Parameters
    ----------
    data_dir : Path
        Root directory containing all datasets

    Returns
    -------
    List[Path]
        List of paths to dataset directories
    """
    return [dir for dir in data_dir.iterdir() if dir.is_dir()]


def get_dataset_size(dataset_path: Path) -> float:
    """Calculate total size of dataset in GB.

    Parameters
    ----------
    dataset_path : Path
        Path to dataset directory

    Returns
    -------
    float
        Size of dataset in GB
    """
    train_dataset = dataset_path / "data"
    total_size = sum(
        file.stat().st_size for file in train_dataset.glob("**/*") if file.is_file()
    )
    return total_size / (1024**3)


def get_h5_files(dataset_path: Path) -> List[Path]:
    """Get all HDF5 files in the dataset.

    Parameters
    ----------
    dataset_path : Path
        Path to dataset directory

    Returns
    -------
    List[Path]
        List of paths to HDF5 files
    """
    train_dataset = dataset_path / "data"
    return list(train_dataset.glob("**/*.hdf5"))


def get_dataset_dimensions(h5_file: Path) -> Tuple[int, int, int, List[str], List[str]]:
    """Extract dataset dimensions and field names from HDF5 file.

    Parameters
    ----------
    h5_file : Path
        Path to HDF5 file

    Returns
    -------
    Tuple[int, int, int, List[str], List[str]]
        Tuple containing (n_timesteps, n_x, n_y, field_names_0, field_names_1)
    """
    with h5py.File(h5_file, "r") as f:
        n_timesteps = len(f["dimensions"]["time"])
        n_x = len(f["dimensions"]["x"])
        n_y = len(f["dimensions"]["y"])
        field_names_0 = f["t0_fields"].attrs["field_names"]
        field_names_1 = f["t1_fields"].attrs["field_names"]
    return n_timesteps, n_x, n_y, field_names_0, field_names_1


def initialize_field_stats(field_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Initialize statistics dictionary for fields.

    Parameters
    ----------
    field_names : List[str]
        List of field names

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing initialized statistics for each field
    """
    return {
        name: {
            "min": float("inf"),
            "max": float("-inf"),
            "sum": 0,
            "count": 0,
            "values": [],  # Store all values for median calculation
        }
        for name in field_names
    }


def process_field_data(field_data: np.ndarray, stats: Dict[str, Any]) -> None:
    """Update statistics with field data.

    Parameters
    ----------
    field_data : np.ndarray
        Field data array
    stats : Dict[str, Any]
        Statistics dictionary to update
    """
    # Ignore NaN values in calculations
    valid_data = field_data[~np.isnan(field_data)]
    if valid_data.size > 0:  # Only update if we have valid data
        stats["min"] = min(stats["min"], np.min(valid_data))
        stats["max"] = max(stats["max"], np.max(valid_data))
        stats["sum"] += np.sum(valid_data)
        stats["count"] += valid_data.size
        # Store values for median calculation
        stats["values"].extend(valid_data.flatten().tolist())


def calculate_field_statistics(
    h5_files: List[Path], field_names_0: List[str], field_names_1: List[str]
) -> Tuple[Dict, Dict]:
    """Calculate statistics for all fields across all files.

    Parameters
    ----------
    h5_files : List[Path]
        List of HDF5 files to process
    field_names_0 : List[str]
        Names of t0 fields
    field_names_1 : List[str]
        Names of t1 fields

    Returns
    -------
    Tuple[Dict, Dict]
        Tuple containing statistics for t0 and t1 fields
    """
    field_stats_0 = initialize_field_stats(field_names_0)
    field_stats_1 = initialize_field_stats(field_names_1)

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            for field_name in field_names_0:
                process_field_data(
                    f["t0_fields"][field_name][:], field_stats_0[field_name]
                )
            for field_name in field_names_1:
                process_field_data(
                    f["t1_fields"][field_name][:], field_stats_1[field_name]
                )

    return field_stats_0, field_stats_1


def get_final_statistics(
    field_stats: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Convert raw statistics to final statistics with means and medians.

    Parameters
    ----------
    field_stats : Dict[str, Dict[str, Any]]
        Raw statistics dictionary

    Returns
    -------
    Dict[str, Dict[str, float]]
        Final statistics dictionary with means and medians
    """
    return {
        name: {
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["sum"] / stats["count"],
            "median": float(np.median(stats["values"]))
            if stats["values"]
            else float("nan"),
        }
        for name, stats in field_stats.items()
    }


def analyze_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Analyze a single dataset and return its statistics.

    Parameters
    ----------
    dataset_path : Path
        Path to dataset directory

    Returns
    -------
    Dict[str, Any]
        Dictionary containing dataset statistics
    """
    h5_files = get_h5_files(dataset_path)
    n_traj = sum(int(h5py.File(f, "r").attrs["n_trajectories"]) for f in h5_files)

    n_timesteps, n_x, n_y, field_names_0, field_names_1 = get_dataset_dimensions(
        h5_files[-1]
    )
    field_stats_0, field_stats_1 = calculate_field_statistics(
        h5_files, field_names_0, field_names_1
    )

    return {
        "size": f"{get_dataset_size(dataset_path):.2f} GB",
        "n_traj": n_traj,
        "n_timesteps": n_timesteps,
        "n_x": n_x,
        "n_y": n_y,
        "n_fields_0": len(field_names_0),
        "n_fields_1": len(field_names_1),
        "t0_field_stats": get_final_statistics(field_stats_0),
        "t1_field_stats": get_final_statistics(field_stats_1),
    }


def print_dataset_statistics(dataset_name: str, stats: Dict[str, Any]) -> None:
    """Print dataset statistics in a formatted way.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    stats : Dict[str, Any]
        Dataset statistics
    """
    print(f"{dataset_name}:")
    for key, value in stats.items():
        if key not in ["t0_field_stats", "t1_field_stats"]:
            print(f"  {key}: {value}")

    print("\n  t0_field_stats:")
    for field_name, field_stats in stats["t0_field_stats"].items():
        print(f"    {field_name}:")
        for stat_name, stat_value in field_stats.items():
            print(f"      {stat_name}: {stat_value:.6f}")

    print("\n  t1_field_stats:")
    for field_name, field_stats in stats["t1_field_stats"].items():
        print(f"    {field_name}:")
        for stat_name, stat_value in field_stats.items():
            print(f"      {stat_name}: {stat_value:.6f}")
    print()


def main():
    """Main function to analyze all datasets."""
    data_dir = Path(
        r"C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\data\datasets"
    )
    datasets = get_dataset_paths(data_dir)

    dataset_stats = {}
    for dataset in datasets:
        print(f"Analyzing dataset: {dataset}")
        dataset_stats[dataset.name] = analyze_dataset(dataset)
        print_dataset_statistics(dataset.name, dataset_stats[dataset.name])


if __name__ == "__main__":
    main()
