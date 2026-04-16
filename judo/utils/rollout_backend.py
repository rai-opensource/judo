# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Base class for rollout backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RolloutBackend(ABC):
    """Abstract base class for rollout backends.

    Rollout backends conduct parallel trajectory simulations
    for sampling-based optimization in the controller.
    """

    num_threads: int

    @abstractmethod
    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Conduct parallel rollouts.

        Args:
            x0: Initial state, shape (nq+nv,). Will be tiled to num_threads internally.
            controls: Control inputs, shape (num_threads, num_timesteps, nu).
            last_policy_output: Previous policy outputs, if applicable.

        Returns:
            Tuple of:
                - states: Rolled out states, shape (num_threads, num_timesteps, nq+nv)
                - sensors: Sensor readings, shape (num_threads, num_timesteps, nsensor)
                - policy_outputs: Final policy outputs, or None if not applicable.
        """

    @abstractmethod
    def update(self, num_threads: int) -> None:
        """Update the number of threads.

        Args:
            num_threads: New number of parallel threads.
        """
