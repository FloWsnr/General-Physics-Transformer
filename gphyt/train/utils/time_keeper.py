"""Utility class for monitoring training time and handling graceful shutdowns.

By: Florian Wiesner
Date: 2025-09-15
"""

import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class TimeEstimates:
    """Time estimates for different training phases.

    Parameters
    ----------
    avg_sec_per_epoch : float
        Average time per epoch in seconds
    avg_sec_per_update_loop : float
        Average time per update loop in seconds
    avg_sec_per_val_loop : float
        Average time per validation loop in seconds
    """

    avg_sec_per_epoch: float
    avg_sec_per_update_loop: float
    avg_sec_per_val_loop: float


class TimeKeeper:
    """Monitors training time and handles graceful shutdowns.

    This class keeps track of remaining time and estimates for different training
    phases. It can determine if there's enough time to complete the next phase
    and trigger graceful shutdown if needed.

    Parameters
    ----------
    time_limit : Optional[float]
        Total time limit in seconds. If None, no time limit is enforced.
    global_rank : int
        Global rank of the process for distributed training
    """

    def __init__(
        self,
        time_limit: Optional[float],
        global_rank: int = 0,
    ):
        self.time_limit = time_limit
        self.global_rank = global_rank
        self.time_start = time.time()
        self.estimates = TimeEstimates(
            avg_sec_per_epoch=0.0,
            avg_sec_per_update_loop=0.0,
            avg_sec_per_val_loop=0.0,
        )

    def update_estimate(self, duration: float, state_key: str, n_phases: int) -> None:
        """Update time estimates for different training phases.

        Parameters
        ----------
        duration : float
            Duration of the current phase in seconds
        state_key : str
            Key in TimeEstimates to update
            ('avg_sec_per_epoch', 'avg_sec_per_update_loop', or 'avg_sec_per_val_loop')
        n_phases : int
            Number of phases to update the average time for
        """
        current_avg = getattr(self.estimates, state_key)
        new_avg = (current_avg * n_phases + duration) / (n_phases + 1)
        setattr(self.estimates, state_key, new_avg)

    def get_remaining_time(self) -> float:
        """Get remaining time in seconds.

        Returns
        -------
        float
            Remaining time in seconds. Returns float('inf') if no time limit is set.
        """
        if self.time_limit is None:
            return float("inf")
        return self.time_limit - (time.time() - self.time_start)

    def should_stop_training(self) -> bool:
        """Check if training should be stopped due to time constraints.

        Returns
        -------
        bool
            True if training should be stopped, False otherwise
        """
        if self.time_limit is None:
            return False

        remaining_time = self.get_remaining_time()
        # Add a 10% safety margin to the time estimates
        estimated_next_epoch = self.estimates.avg_sec_per_epoch * 1.1

        return remaining_time < estimated_next_epoch

    def should_stop_update_loop(self) -> bool:
        """Check if the current update loop should be stopped.

        Returns
        -------
        bool
            True if the update loop should be stopped, False otherwise
        """
        if self.time_limit is None:
            return False

        remaining_time = self.get_remaining_time()
        # Add a 10% safety margin to the time estimates
        estimated_next_loop = self.estimates.avg_sec_per_update_loop * 1.1

        return remaining_time < estimated_next_loop

    def should_stop_validation(self) -> bool:
        """Check if validation should be skipped due to time constraints.

        Returns
        -------
        bool
            True if validation should be skipped, False otherwise
        """
        if self.time_limit is None:
            return False

        remaining_time = self.get_remaining_time()
        # Add a 10% safety margin to the time estimates
        estimated_validation = self.estimates.avg_sec_per_val_loop * 1.1

        return remaining_time < estimated_validation
