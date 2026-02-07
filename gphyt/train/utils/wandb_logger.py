"""
Wandb logger for handling all wandb-related logging functionality.

Author: Florian Wiesner
Date: 2025-04-07
"""

from typing import Any, Dict, Optional, Literal
from pathlib import Path

from PIL import Image
import wandb
from wandb.sdk.wandb_run import Run

from gphyt.train.utils.logger import setup_logger


class WandbLogger:
    """A class to handle all wandb logging functionality with error handling.

    This class is responsible for initializing wandb, logging metrics, and handling
    any errors that might occur during logging. It ensures that training can continue
    even if wandb logging fails.

    Parameters
    ----------
    wandb_config : WandbConfig
        WandbConfig object containing wandb settings
    log_dir : Optional[Path], optional
        Directory to save wandb logs, by default None
        Usually to store the logs in the same directory as checkpoints
    rank : int, optional
        Global rank for distributed training (logger only logs on rank 0), by default 0
    """

    def __init__(
        self,
        wandb_config: dict,
        log_dir: Optional[Path] = None,
        rank: int = 0,
    ):
        self.config = wandb_config
        self.log_dir = log_dir
        self.logger = setup_logger(
            "WandbLogger",
            rank=rank,
        )
        self.enabled = self.config.get("enabled", False)
        self.run: Optional[Run] = None
        if self.enabled:
            self._initialize_wandb()
        else:
            self.logger.info("Wandb logging is disabled")

    def _initialize_wandb(self) -> None:
        """Initialize wandb with error handling."""
        try:
            # Create a clean wandb ID from the log directory name
            # Use just the directory name, not the full path
            if self.log_dir:
                wandb_id = self.log_dir.name
            else:
                wandb_id = None  # Let wandb generate a unique ID

            wandb.login()
            self.run = wandb.init(
                project=self.config.get("project"),
                entity=self.config.get("entity"),
                config=None,  # Full config is logged via update_config() after defaults
                id=wandb_id,
                dir=self.log_dir,
                tags=self.config.get("tags", []),
                notes=self.config.get("notes", ""),
                resume="allow",
                settings={"init_timeout": 60},
            )
            self.logger.info("Successfully initialized wandb")
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {str(e)}")
            self.run = None

    def log(self, data: Dict[str, Any], folder: str, commit: bool = True) -> None:
        """Log data to wandb with error handling.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data to log
        folder : str
            Wandb folder to log the data to
        commit : bool, optional
            Whether to commit the data immediately, by default True
        """
        if not self.enabled or self.run is None:
            return

        try:
            # add folder to the dict strings
            data = {f"{folder}/{k}": v for k, v in data.items()}
            self.run.log(data, commit=commit)
        except Exception as e:
            self.logger.error(f"Failed to log data to wandb: {str(e)}")

    def watch(
        self,
        model: Any,
        criterion: Any,
        log: Literal["gradients", "parameters"] = "gradients",
        log_freq: int = 100,
    ) -> None:
        """Watch model parameters with error handling.

        Parameters
        ----------
        model : Any
            Model to watch
        criterion : Any
            Loss function
        log : str, optional
            What to log, by default "gradients"
        log_freq : int, optional
            How often to log, by default 100
        """
        if not self.enabled or self.run is None:
            return

        try:
            self.run.watch(
                model,
                criterion=criterion,
                log=log,
                log_freq=log_freq,
            )
        except Exception as e:
            self.logger.error(f"Failed to watch model in wandb: {str(e)}")

    def update_config(self, data: Dict[str, Any]) -> None:
        """Update wandb config with error handling.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of config values to update
        """
        if not self.enabled or self.run is None:
            return

        try:
            self.run.config.update(data, allow_val_change=True)
        except Exception as e:
            self.logger.error(f"Failed to update wandb config: {str(e)}")

    def finish(self) -> None:
        """Finish the wandb run with error handling."""
        if not self.enabled or self.run is None:
            return

        try:
            self.run.finish()
            self.logger.info("Successfully finished wandb run")
        except Exception as e:
            self.logger.error(f"Failed to finish wandb run: {str(e)}")

    def log_image(
        self,
        image_dir: Path,
        folder: str,
        commit: bool = True,
    ) -> None:
        """Log PNG images from directory to wandb with error handling.

        Parameters
        ----------
        image_dir : Path
            Path to the directory containing the images
        folder : str
            Wandb folder to log the data to
        commit : bool, optional
            Whether to commit the data immediately, by default True
        """
        if not self.enabled or self.run is None:
            return

        try:
            data = {}
            for image in image_dir.glob("**/*.png"):
                img = Image.open(image)
                data[f"{folder}/{image.name}"] = wandb.Image(
                    img, file_type="png", mode="RGB"
                )
            self.run.log(data, commit=commit)
        except Exception as e:
            self.logger.error(f"Failed to log images to wandb: {str(e)}")
