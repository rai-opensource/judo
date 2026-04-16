# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MuJoCo rollout backend with locomotion policy support."""

from pathlib import Path

import numpy as np
from mujoco import MjModel

from judo.tasks.spot.spot_constants import DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME
from judo.utils.rollout_backend import RolloutBackend


class PolicyMJRolloutBackend(RolloutBackend):
    """Rollout backend with C++ mujoco_extensions and ONNX locomotion policy inference.

    For Spot tasks, the command format is a 25-dim vector:
    [base_vel(3), arm(7), legs(12), torso(3)]
    """

    def __init__(
        self,
        model: MjModel,
        num_threads: int,
        policy_path: str | Path,
        physics_substeps: int = 2,
    ) -> None:
        """Initialize the policy rollout backend.

        Args:
            model: MuJoCo model for the scene.
            num_threads: Number of parallel rollout threads.
            policy_path: Path to ONNX locomotion policy.
            physics_substeps: Physics steps per control step.
        """
        self.num_threads = num_threads
        self.model = model
        self.physics_substeps = physics_substeps
        self._policy_path = policy_path

        self._setup_mujoco_extensions(model, policy_path, num_threads)

    def _setup_mujoco_extensions(self, model: MjModel, policy_path: str | Path, num_threads: int) -> None:
        """Setup the mujoco_extensions C++ rollout backend with ONNX policy."""
        try:
            from mujoco_extensions.policy_rollout import create_systems_vector, threaded_rollout  # type: ignore  # noqa: PLC0415, I001
        except ImportError as e:
            raise ImportError("mujoco_extensions is required. Build with: pixi run build") from e

        self._systems = create_systems_vector(
            model,
            str(policy_path),
            num_threads,
        )
        self._threaded_rollout = threaded_rollout

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Conduct parallel rollouts with policy inference.

        Args:
            x0: Initial state, shape (nq+nv,). Will be tiled to num_threads internally.
            controls: Control inputs, shape (num_threads, num_timesteps, cmd_dim).
            last_policy_output: Previous policy outputs, shape (num_threads, 12).
                Required for this backend.

        Returns:
            Tuple of:
                - states: Rolled out states, shape (num_threads, num_timesteps, nq+nv)
                - sensors: Sensor readings, shape (num_threads, num_timesteps, nsensor)
                - policy_outputs: Final policy outputs, shape (num_threads, 12).
        """
        if x0.ndim == 1:
            x0 = np.tile(x0, (self.num_threads, 1))

        if last_policy_output is None:
            last_policy_output = np.zeros((x0.shape[0], 12))

        x0 = np.asarray(x0, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        last_policy_output = np.asarray(last_policy_output, dtype=np.float64)

        states, sensors, policy_outputs = self._threaded_rollout(
            self._systems,
            x0,
            controls,
            last_policy_output,
            self.num_threads,
            self.physics_substeps,
            DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME,
        )

        return np.array(states), np.array(sensors), np.array(policy_outputs)

    def update(self, num_threads: int) -> None:
        """Update the number of threads.

        Recreates C++ systems for new thread count.

        Args:
            num_threads: New number of parallel threads.
        """
        self.num_threads = num_threads
        self._setup_mujoco_extensions(self.model, self._policy_path, num_threads)
