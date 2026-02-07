"""Utility functions for training and evaluation.

By: Florian Wiesner
Date: 2025-09-15
"""

from pathlib import Path, PurePath
from typing import Optional

import torch
import torch.distributed as dist


def human_format(num: int | float) -> str:
    """Format a number with SI prefixes (K, M, B).

    Parameters
    ----------
    num : int or float
        The number to format.

    Returns
    -------
    str
        Formatted string with SI prefix.
    """
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000
    return f"{num:.2f}P"


def reduce_all_losses(losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reduce the losses across all GPUs."""
    for loss_name, loss in losses.items():
        losses[loss_name] = _reduce_loss(loss)
    return losses


def _reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """Reduce the loss across all GPUs."""
    if dist.is_initialized():
        loss_tensor = loss.clone().detach()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor
    else:
        return loss


@torch.inference_mode()
def compute_metrics(
    x: torch.Tensor,
    target: torch.Tensor,
    metrics: dict[str, torch.nn.Module],
) -> dict[str, torch.Tensor]:
    """Compute the metrics.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    target : torch.Tensor
        The target tensor.
    metrics : dict[str, torch.nn.Module]
        a dictionary of metric names and metric functions

    Returns
    -------
    dict[str, torch.Tensor]
        a dictionary of metric names and metric values
    """
    metrics_values = {}
    for metric_name, metric in metrics.items():
        metric_value = metric(x, target)
        metrics_values[metric_name] = metric_value
    return metrics_values


def gather_across_gpus(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all GPUs and concatenate them.

    In DDP mode, each GPU processes a different subset of the data.
    This function gathers tensors from all GPUs and concatenates them
    along dimension 0 (batch dimension).

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to gather. Should be on the GPU device.
        Shape: (local_batch_size, ...) where local_batch_size is the
        number of samples on this GPU.

    Returns
    -------
    torch.Tensor
        The gathered tensor containing data from all GPUs.
        Shape: (total_batch_size, ...) where total_batch_size is the
        sum of local_batch_size across all GPUs.
        In non-distributed mode, returns the input tensor unchanged.
    """
    if not dist.is_initialized():
        return tensor

    # Get world size (number of GPUs)
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    # Gather tensors from all GPUs
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)

    # Concatenate along batch dimension
    return torch.cat(gathered_tensors, dim=0)


def base_attribute(obj, attr: str):
    """Get the base attribute of an object.

    This function retrieves the base attribute of an object, bypassing any
    wrappers such as DistributedDataParallel (DDP) or torch.compile. It is
    useful for accessing the original model or component when it has been
    wrapped for distributed training or optimization.

    Parameters
    ----------
    obj : Any
        The object from which to retrieve the base attribute.
    attr : str
        The name of the attribute to retrieve.

    Returns
    -------
    Any
        The base attribute of the object.
    """
    # Unwrap DistributedDataParallel if applicable
    if isinstance(obj, torch.nn.parallel.DistributedDataParallel):
        obj = obj.module

    return getattr(obj, attr)
