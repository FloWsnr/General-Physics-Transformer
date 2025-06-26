"""
Wandb logger for handling all wandb-related logging functionality.

Author: Florian Wiesner
Date: 2025-04-07
"""

from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from dotenv import load_dotenv

import wandb
import wandb.wandb_run
from wandb.sdk.wandb_run import Run

from lpfm.utils.logger import get_logger


class WandbLogger:
    """A class to handle all wandb logging functionality with error handling.

    This class is responsible for initializing wandb, logging metrics, and handling
    any errors that might occur during logging. It ensures that training can continue
    even if wandb logging fails.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing wandb settings
    log_file : str
        Path to the log file
    log_level : str
        Logging level
    """

    def __init__(
        self,
        config: dict,
        log_file: str,
        log_level: str = "INFO",
    ):
        load_dotenv()
        self.config = config
        self.logger = get_logger(
            "WandbLogger",
            log_file=log_file,
            log_level=log_level,
        )
        self.run: Optional[Run] = None
        self._initialize_wandb()

    def _initialize_wandb(self) -> None:
        """Initialize wandb with error handling."""
        try:
            wandb_id = self.config["wandb"]["id"]
            wandb.login()
            self.run = wandb.init(
                project=self.config["wandb"]["project"],
                entity=self.config["wandb"]["entity"],
                config=self.config,
                id=wandb_id,
                tags=self.config["wandb"]["tags"],
                notes=self.config["wandb"]["notes"],
                resume="allow",
                settings=wandb.Settings(init_timeout=120),
            )
            self.logger.info("Successfully initialized wandb")
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {str(e)}")
            self.run = None

    def log(self, data: Dict[str, Any], commit: bool = True) -> None:
        """Log data to wandb with error handling.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data to log
        commit : bool, optional
            Whether to commit the data immediately, by default True
        """
        if self.run is None:
            return

        try:
            self.run.log(data, commit=commit)
        except Exception as e:
            self.logger.error(f"Failed to log data to wandb: {str(e)}")

    def watch(
        self,
        model: Any,
        criterion: Any,
        log: str = "gradients",
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
        if self.run is None:
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
        if self.run is None:
            return

        try:
            self.run.config.update(data, allow_val_change=True)
        except Exception as e:
            self.logger.error(f"Failed to update wandb config: {str(e)}")

    def finish(self) -> None:
        """Finish the wandb run with error handling."""
        if self.run is None:
            return

        try:
            self.run.finish()
            self.logger.info("Successfully finished wandb run")
        except Exception as e:
            self.logger.error(f"Failed to finish wandb run: {str(e)}")

    def log_predictions(
        self,
        image_path: Path,
        name_prefix: str,
    ) -> None:
        """Log predictions to wandb with error handling.

        Parameters
        ----------
        image_path : Path
            Path to the images
        name_prefix : str
            Prefix for the image names
        """
        if self.run is None:
            return

        try:
            data = {}
            for image in image_path.glob("**/*.png"):
                img = Image.open(image)
                data[f"{name_prefix}/{image.name}"] = wandb.Image(
                    img, file_type="png", mode="RGB"
                )
            self.run.log(data)
        except Exception as e:
            self.logger.error(f"Failed to log predictions to wandb: {str(e)}")
