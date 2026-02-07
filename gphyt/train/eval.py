"""
Detailed evaluation of the model, its predictions, and the losses.
By: Florian Wiesner
Date: 2025-05-01
"""

from pathlib import Path
from typing import Optional
import logging
import math

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from gphyt.train.utils.run_utils import compute_metrics, reduce_all_losses
from gphyt.train.utils.logger import setup_logger


class Evaluator:
    """Thorough evaluation of the model on the full dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    dataloader : DataLoader
        Dataloader to evaluate on
    metrics : dict[str, torch.nn.Module]
        Dictionary of metrics to evaluate
    eval_dir : Path
        Directory to save evaluation results
    eval_fraction : float, optional
        Fraction of validation data to use for evaluation (0.0-1.0), by default 1.0
    amp : bool, optional
        Whether to use automatic mixed precision, by default True
    amp_precision : torch.dtype, optional
        Precision to use for AMP, by default torch.bfloat16
    global_rank : int, optional
        Global rank for distributed training, by default 0
    local_rank : int, optional
        Local rank for distributed training, by default 0
    world_size : int, optional
        World size for distributed training, by default 1
    logger : logging.Logger, optional
        Logger to use, by default None
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        metrics: dict[str, torch.nn.Module],
        eval_dir: Path,
        eval_fraction: float = 1.0,
        n_ar_steps: int = 1,
        amp: bool = True,
        amp_precision: torch.dtype = torch.bfloat16,
        global_rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or setup_logger("Evaluator", rank=global_rank)
        self.global_rank = global_rank
        self.eval_fraction = eval_fraction
        self.n_ar_steps = n_ar_steps
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.ddp_enabled = dist.is_initialized()

        self.model = model
        self.model.eval()
        self.model.to(self.device)

        self.dataloader = dataloader
        self.metrics = metrics
        self.eval_dir = eval_dir
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = amp
        self.amp_precision = amp_precision

    def log_msg(self, msg: str):
        """Log a message."""
        self.logger.info(f"{msg}")

    @torch.inference_mode()
    def eval(self) -> dict[str, torch.Tensor]:
        """Evaluate the model on the full dataset.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of metrics (losses) averaged over the full dataset.
        """
        total_metrics = {}
        for metric_name, _ in self.metrics.items():
            total_metrics[metric_name] = torch.tensor(0.0, device=self.device)

        n_batches = int(len(self.dataloader) * self.eval_fraction)
        log_interval = max(1, 10 ** math.floor(math.log10(max(1, n_batches // 100))))

        for i, data in enumerate(self.dataloader):
            if (i + 1) % log_interval == 0 or i == 0:
                self.log_msg(f"Batch {i + 1}/{n_batches}")

            x = data[0]
            target = data[1]
            x = x.to(self.device)
            target = target.to(self.device)

            if self.n_ar_steps > 1:
                # Autoregressive evaluation loop
                ar_steps = target.shape[1]
                for _ar_step in range(ar_steps):
                    if _ar_step == 0:
                        x_ar = x
                    else:
                        x_ar = torch.cat((x_ar[:, 1:, ...], y), dim=1)

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_precision,
                        enabled=self.use_amp,
                    ):
                        y = self.model(x_ar)

                # Use final output vs final target for metrics
                final_target = target[:, -1:, ...]
                current_metrics = compute_metrics(y, final_target, self.metrics)
            else:
                # Standard single-step forward pass
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_precision,
                    enabled=self.use_amp,
                ):
                    y = self.model(x)

                current_metrics = compute_metrics(y, target, self.metrics)

            current_metrics = reduce_all_losses(current_metrics)
            for metric_name, metric_value in current_metrics.items():
                total_metrics[metric_name] += metric_value.float()

            if i + 1 >= n_batches:
                break

        for metric_name, metric_value in total_metrics.items():
            total_metrics[metric_name] /= n_batches

        return total_metrics
