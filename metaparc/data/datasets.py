from pathlib import Path
from typing import Optional
import einops
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, default_collate
from the_well.data.datasets import WellDataset


def get_rng_transforms(p_h_flip: float, p_v_flip: float) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=p_h_flip),
            transforms.RandomVerticalFlip(p=p_v_flip),
        ]
    )


def collate_fn(data: list[dict]) -> torch.Tensor:
    """Collate function for the dataset.
    Get dict with input and output fields from the WellDataset.
    The fields are of shape (Time steps, H, W, C)
    We want to get a batch of shape (B, Time steps & C, H, W)

    Parameters
    ----------
    data : dict
        Dict with input and output fields from the WellDataset

    Returns
    -------
    batch : torch.Tensor
        Batch of shape (B, Time steps & C, H, W)
    """

    batch = default_collate(data)
    input_fields = batch["input_fields"]
    output_fields = batch["output_fields"]

    # rearrange to (B, Time steps & C, H, W)
    input_fields = einops.rearrange(
        input_fields, "batch time h w c -> batch (time c) h w"
    )
    output_fields = einops.rearrange(
        output_fields, "batch time h w c -> batch (time c) h w"
    )

    # Replace NaNs with 0
    batch["input_fields"] = torch.where(
        torch.isnan(input_fields),
        torch.zeros_like(input_fields),
        input_fields,
    )
    batch["output_fields"] = torch.where(
        torch.isnan(output_fields),
        torch.zeros_like(output_fields),
        output_fields,
    )
    return batch


class PhysicsDataset(WellDataset):
    """Wrapper around the WellDataset.

    Enables data augmentation.
    """

    def __init__(
        self,
        data_dir: Path,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        split: str = "train",
    ):
        super().__init__(
            path=str(data_dir),
            well_split_name=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
        )

    def __getitem__(self, index):
        data = super().__getitem__(index)
        return data
