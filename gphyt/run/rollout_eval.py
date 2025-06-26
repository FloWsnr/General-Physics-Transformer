import torch
import torch.nn as nn
import numpy as np
from lpfm.data.dataset_utils import get_datasets


@torch.inference_mode()
def evaluate_rollout(
    model: nn.Module,
    criterion: nn.Module,
    data_config: dict,
    device: torch.device,
    num_samples: int = 10,
) -> dict[str, float]:
    model.eval()
    losses = {}

    config = data_config.copy()
    config["full_trajectory_mode"] = True
    config["max_rollout_steps"] = 50

    for dt in [1, 4, 8]:
        config["dt_stride"] = dt

        datasets = get_datasets(config, split="val")
        for dataset_name, dataset in datasets.items():
            losses[dataset_name] = {dt: 0}
            acc_loss = 0
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            for traj_idx in indices:
                x, full_traj = dataset[traj_idx]
                x = x.to(device)
                full_traj = full_traj.to(device)

                x = x.unsqueeze(0)
                full_traj = full_traj.unsqueeze(0)

                B, T, H, W, C = full_traj.shape

                outputs = []
                with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                    for i in range(T):
                        output = model(x)
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            break
                        outputs.append(output)

                        x = torch.cat([x[:, 1:, ...], output], dim=1)

                outputs = torch.cat(outputs, dim=1)
                loss = criterion(outputs, full_traj)
                acc_loss += loss.item()

            losses[dataset_name][dt] = acc_loss / num_samples

    return losses
