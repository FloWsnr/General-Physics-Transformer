from lpfm.data.phys_dataset import PhysicsDataset
import torch

def get_dataset_sample(dataset: PhysicsDataset, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = dataset[index]
    return x, y




