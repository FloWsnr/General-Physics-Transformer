import logging
from pathlib import Path
import sys
from typing import Optional, TextIO


class RankAwareLogger(logging.Logger):
    """Logger that only logs on a specific rank in distributed training.

    This logger wraps the standard logging.Logger and only produces output
    on the specified rank (default: rank 0). This prevents duplicate log
    messages when running multi-GPU/multi-node training.

    Note: DEBUG level messages are always logged on all ranks to facilitate
    debugging distributed training issues.

    Parameters
    ----------
    name : str
        Name of the logger
    rank : int, optional
        The rank that should produce log output, by default 0
    level : int, optional
        Logging level, by default logging.INFO
    """

    def __init__(self, name: str, rank: int = 0, level: int = logging.INFO):
        super().__init__(name, level)
        self.rank = rank
        self.target_rank = rank

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):
        """Override _log to only log on the target rank, except for DEBUG level.

        DEBUG level logs are always printed on all ranks to help with debugging
        distributed training issues.
        """
        # Always log DEBUG messages on all ranks for debugging purposes
        if level == logging.DEBUG or self.rank == self.target_rank:
            super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel + 1)


def setup_logger(
    name: str = "gphyt",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    stream: Optional[TextIO] = sys.stdout,
    format_string: Optional[str] = None,
    rank: Optional[int] = None,
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
    rank : Optional[int], optional
        If provided, creates a RankAwareLogger that only logs on this rank.
        If None, creates a standard logger that logs on all ranks.
        By default None

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create rank-aware logger if rank is specified
    if rank is not None:
        # Register the custom logger class
        logging.setLoggerClass(RankAwareLogger)
        logger = logging.getLogger(name)
        if isinstance(logger, RankAwareLogger):
            logger.rank = rank
            logger.target_rank = 0  # Only rank 0 should log
        logging.setLoggerClass(logging.Logger)  # Reset to default
    else:
        logger = logging.getLogger(name)

    logger.setLevel(log_level)

    # Prevent propagation to parent loggers (prevents duplicate logs)
    logger.propagate = False

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Default format string if none provided
    if format_string is None:
        format_string = "[%(asctime)s] - %(name)s - %(message)s"

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
