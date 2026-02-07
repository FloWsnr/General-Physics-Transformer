"""DataLoader for the Dataset.

By: Florian Wiesner
Date: 2025-09-11
"""

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
    Dataset,
)
import torch


def get_dataloader(
    dataset: Dataset,
    seed: int,
    batch_size: int,
    num_workers: int,
    is_distributed: bool = False,
    shuffle: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    """Get a dataloader for the dataset.

    Uses the correct sampler depending on whether distributed training is used.

    Parameters
    ----------
    dataset : Dataset
        Dataset to load.
    seed : int
        Seed for the dataset.
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers.
    is_distributed : bool
        Whether to use distributed sampling
    shuffle : bool
        Whether to shuffle the dataset
    persistent_workers : bool
        If True, workers are kept alive between batches/epochs. Reduces overhead
        but increases memory usage. Set to False for eval dataloaders to avoid
        OOM when train workers are also persistent.
    """

    if is_distributed:
        sampler = DistributedSampler(dataset, seed=seed, shuffle=shuffle)
    else:
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = RandomSampler(dataset, generator=generator)
        else:
            sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=True,
    )

    return dataloader
