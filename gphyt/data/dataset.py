import torch
from torch.utils.data import Dataset


def get_dataset(config: dict, split: str = "train") -> Dataset:
    """Factory function to create a dataset based on configuration.

    Creates a SuperDataset containing PhysicsDatasets for each
    dataset specified in the config.

    Parameters
    ----------
    config : dict
        Dataset configuration dictionary.
    split : str
        Data split to use ("train", "val", "valid", "test").

    Returns
    -------
    Dataset
        A SuperDataset containing all configured PhysicsDatasets.
    """
    from gphyt.data.dataset_utils import get_datasets
    from gphyt.data.phys_dataset import SuperDataset

    datasets = get_datasets(config, split)
    if not datasets:
        raise ValueError(f"No datasets found for split '{split}'")

    super_dataset = SuperDataset(
        datasets=list(datasets.values()),
        max_samples=config.get("max_samples_per_ds", None),
    )
    return super_dataset
