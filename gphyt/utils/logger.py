import logging
from pathlib import Path
import sys
from typing import Optional, TextIO


def setup_logger(
    name: str = "gphyt",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    stream: Optional[TextIO] = sys.stdout,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with a pretty format.

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default "gphyt"
    log_level : int, optional
        Logging level, by default logging.INFO
    log_file : Optional[Path], optional
        Path to log file, by default None
    stream : Optional[TextIO], optional
        Stream to log to, by default sys.stdout
    format_string : Optional[str], optional
        Custom format string, by default None

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Default format string if none provided
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Add stream handler if specified
    if stream:
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(
    name: str = "gphyt",
    log_file: Optional[Path] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Get a logger with a pretty format.

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default "gphyt"
    log_file : Optional[str], optional
        Path to log file, by default None
    log_level : int, optional
        Logging level, by default logging.INFO

    Returns
    -------
    logging.Logger
    """
    return setup_logger(name=name, log_file=log_file, log_level=log_level)
