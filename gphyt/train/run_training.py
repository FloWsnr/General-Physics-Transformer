import argparse
import platform
import os
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp.grad_scaler import GradScaler
import torch._functorch.config as functorch_config

import yaml
from yaml import CLoader

from gphyt.models.model_utils import get_model
from gphyt.models.loss_fns import MAE, MSE, RMSE, NRMSE, VRMSE
from gphyt.models.transformer.loss_fns import RNMSELoss, RVMSELoss

from gphyt.data.dataloader import get_dataloader
from gphyt.data.dataset import get_dataset

from gphyt.train.train_base import Trainer
from gphyt.train.utils.optimizer import get_optimizer
from gphyt.train.utils.lr_scheduler import get_lr_scheduler
from gphyt.train.utils.checkpoint_utils import load_checkpoint
from gphyt.train.utils.wandb_logger import WandbLogger
from gphyt.train.utils.logger import setup_logger


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=CLoader)
    return config


def time_str_to_seconds(time_str: str) -> float:
    return sum(x * int(t) for x, t in zip([3600, 60, 1], time_str.split(":")))


def get_checkpoint_path(output_dir: Path, checkpoint_name: str) -> Path:
    if checkpoint_name == "latest":
        checkpoint_path = output_dir / "latest.pt"
    elif checkpoint_name == "best":
        checkpoint_path = output_dir / "best.pt"
    elif checkpoint_name.isdigit():
        checkpoint_path = output_dir / f"epoch_{checkpoint_name}/checkpoint.pt"
    else:
        raise ValueError(f"Invalid checkpoint name: {checkpoint_name}")

    return checkpoint_path


def _is_nested_config(config: dict) -> bool:
    """Check if config uses GPhyT nested format (has 'training', 'data', 'model' keys)."""
    return "training" in config and "data" in config


@record
def main(
    config_path: Path,
):
    load_dotenv()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    logger = setup_logger("Startup", rank=global_rank)

    config = load_config(config_path)
    output_dir = config_path.parent

    ############################################################
    ###### Parse config (nested GPhyT or flat format) ##########
    ############################################################
    if _is_nested_config(config):
        # GPhyT nested config: config["training"], config["data"], config["model"]
        training_config = config["training"]
        data_config = config["data"]
        model_config = config["model"]

        # Set img_size on model config from data config
        model_config["img_size"] = (
            data_config["n_steps_input"],
            data_config["out_shape"][0],
            data_config["out_shape"][1],
        )

        # Map training fields
        seed = int(training_config["seed"])
        batch_size = int(training_config["batch_size"])
        num_workers = int(training_config["num_workers"])
        total_updates = int(float(training_config["batches"]))
        updates_per_epoch = int(float(training_config["val_every_batches"]))
        cp_every_updates = int(float(training_config["checkpoint_every_batches"]))
        eval_fraction = float(training_config.get("val_frac_samples", 1.0))

        optimizer_config = training_config["optimizer"]
        lr_config = training_config.get("lr_scheduler", None)
        max_grad_norm = training_config.get("grad_clip", None)
        use_amp = training_config.get("amp", True)
        compile_model = training_config.get("compile", False)
        criterion_name = training_config.get("criterion", "MSE")
        n_ar_steps = data_config.get("n_steps_output", 1)
        mem_budget = training_config.get("mem_budget", 1)

        # Time limit
        time_limit = training_config.get("time_limit", None)
        if time_limit is not None:
            if isinstance(time_limit, str):
                time_limit = time_str_to_seconds(time_limit)
            else:
                time_limit = float(time_limit)

        # Wandb config
        wandb_config = config.get("wandb", {})

        # Dataset config is data_config directly
        dataset_config = data_config

        # Checkpoint config
        cp_config = config.get("checkpoint", {})

        # Persistent workers
        persistent_workers = bool(training_config.get("persistent_workers", False))
        persistent_workers_val = bool(
            training_config.get("persistent_workers_val", persistent_workers)
        )

    else:
        # Flat config format (backward compatible)
        training_config = config
        model_config = config["model"]
        dataset_config = config["dataset"]

        seed = int(config["seed"])
        batch_size = int(config["batch_size"])
        num_workers = int(config["num_workers"])
        total_updates = int(float(config["total_updates"]))
        updates_per_epoch = int(float(config["updates_per_epoch"]))
        cp_every_updates = int(float(config["checkpoint_every_updates"]))
        eval_fraction = float(config.get("eval_fraction", 1.0))

        optimizer_config = config["optimizer"]
        lr_config = config.get("lr_scheduler", None)
        max_grad_norm = config.get("max_grad_norm", None)
        use_amp = config.get("use_amp", True)
        compile_model = config.get("compile", False)
        criterion_name = config["model"].get("criterion", "MSE")
        n_ar_steps = 1
        mem_budget = config.get("mem_budget", 1)

        time_limit = config.get("time_limit", None)
        if time_limit is not None:
            if isinstance(time_limit, str):
                time_limit = time_str_to_seconds(time_limit)
            else:
                time_limit = float(time_limit)

        wandb_config = config["wandb"]
        cp_config = config.get("checkpoint", {})

        persistent_workers = bool(config.get("persistent_workers", False))
        persistent_workers_val = bool(
            config.get("persistent_workers_val", persistent_workers)
        )

    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)

    ############################################################
    ###### Wandb ###############################################
    ############################################################
    if global_rank == 0:
        wandb_logger = WandbLogger(
            wandb_config, log_dir=output_dir, rank=global_rank
        )
        wandb_logger.update_config(config)  # Log full config with defaults
    else:
        wandb_logger = None

    samples_trained = 0
    batches_trained = 0
    epoch = 1

    ############################################################
    ###### AMP #################################################
    ############################################################
    amp_precision_str = training_config.get("amp_precision", "bfloat16")
    if amp_precision_str == "bfloat16":
        amp_precision = torch.bfloat16
    elif amp_precision_str == "float16":
        amp_precision = torch.float16
    else:
        print(f"Unknown amp_precision {amp_precision_str}, turning off AMP")
        use_amp = False
        amp_precision = torch.float32

    scaler = GradScaler(device=str(device), enabled=use_amp)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ############################################################
    ###### Load datasets and dataloaders #######################
    ############################################################

    dataset_train = get_dataset(dataset_config, split="train")
    dataset_val = get_dataset(dataset_config, split="valid")

    train_dataloader = get_dataloader(
        dataset=dataset_train,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        is_distributed=dist.is_initialized(),
        shuffle=True,
        persistent_workers=persistent_workers,
    )
    val_dataloader = get_dataloader(
        dataset=dataset_val,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        is_distributed=dist.is_initialized(),
        shuffle=False,
        persistent_workers=persistent_workers_val,
    )

    ############################################################
    ###### Load torch modules ##################################
    ############################################################

    model = get_model(model_config)
    model.to(device)

    if criterion_name.lower() == "mse":
        criterion_fn = MSE()
    elif criterion_name.lower() == "mae":
        criterion_fn = MAE()
    else:
        raise ValueError(f"Unknown criterion {criterion_name}")

    # these are used for evaluation during training (Wandb logging)
    # these are NOT the loss functions used for training (see criterion)
    eval_loss_fns = {
        "MSE": MSE(),
        "MAE": MAE(),
        "RMSE": RMSE(),
        "NRMSE": NRMSE(),
        "VRMSE": VRMSE(),
        "RNMSE": RNMSELoss(dims=(1, 2, 3)),
        "RVMSE": RVMSELoss(dims=(1, 2, 3)),
    }

    ############################################################
    ###### Load checkpoint #####################################
    ############################################################
    checkpoint: Optional[dict] = None
    checkpoint_name: Optional[str] = cp_config.get("checkpoint_name", None)

    if checkpoint_name is not None:
        logger.info(f"Loading checkpoint: {checkpoint_name}")
        checkpoint_path = get_checkpoint_path(output_dir, checkpoint_name)

        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, device)
        else:
            logger.warning(
                f"Checkpoint {checkpoint_path} not found, starting from scratch"
            )

    ############################################################
    ###### Load model weights ##################################
    ############################################################
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    ############################################################
    ###### Compile and distribute model #########################
    ############################################################
    functorch_config.activation_memory_budget = mem_budget
    if compile_model and not platform.system() == "Windows":
        model = torch.compile(model, mode="max-autotune")
        logger.info("Model compiled with torch.compile")
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=device,
        )
        logger.info("Model wrapped with DDP")

    if wandb_logger is not None:
        wandb_logger.watch(model, criterion=criterion_fn)

    ############################################################
    ###### Setup optimizers and lr schedulers ##################
    ############################################################
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    optimizer = get_optimizer(model, optimizer_config)  # type: ignore

    restart = cp_config.get("restart", False)
    if checkpoint is not None and restart:
        samples_trained = checkpoint["samples_trained"]
        batches_trained = checkpoint["batches_trained"]
        epoch = checkpoint["epoch"]

        # Create scheduler BEFORE loading optimizer state dict, so it captures
        # the correct base LR (used for eta_min in cosine annealing, etc.)
        if lr_config is not None:
            lr_scheduler = get_lr_scheduler(
                optimizer,
                lr_config,
                total_batches=total_updates,
                total_batches_trained=0,  # Initial state before loading
            )
            # Load state dicts AFTER scheduler creation
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        grad_scaler_sd = checkpoint.get("grad_scaler_state_dict", None)
        if grad_scaler_sd is not None:
            scaler.load_state_dict(grad_scaler_sd)

    else:
        if lr_config is not None:
            lr_scheduler = get_lr_scheduler(
                optimizer,
                lr_config,
                total_batches=total_updates,
                total_batches_trained=batches_trained,
            )

    ############################################################
    ###### Initialize trainer ##################################
    ############################################################

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion_fn,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scaler=scaler,
        total_updates=total_updates,
        updates_per_epoch=updates_per_epoch,
        checkpoint_every_updates=cp_every_updates,
        eval_fraction=eval_fraction,
        epoch=epoch,
        batches_trained=batches_trained,
        samples_trained=samples_trained,
        loss_fns=eval_loss_fns,
        amp=use_amp,
        amp_precision=amp_precision,
        max_grad_norm=max_grad_norm,
        n_ar_steps=n_ar_steps,
        output_dir=output_dir,
        wandb_logger=wandb_logger,
        time_limit=time_limit,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config_path)

    main(config_path)
