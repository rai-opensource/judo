# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MuJoCo simulation with hierarchical low-level policy support."""

from pathlib import Path

import numpy as np
from mujoco import mj_forward
from omegaconf import DictConfig

from judo.simulation.mj_simulation import MJSimulation
from judo.tasks import get_task_registration
from judo.tasks.spot.spot_constants import DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME, POLICY_OUTPUT_DIM

try:
    from mujoco_extensions.policy_rollout import create_systems_vector, threaded_rollout  # type: ignore
except ImportError as e:
    raise ImportError(
        "mujoco_extensions is not built. Spot locomotion tasks require the C++ extension.\n"
        "Build it with:  pixi run build\n"
        "See README.md for details."
    ) from e


class HierarchicalMJSimulation(MJSimulation):
    """MuJoCo simulation with a hierarchical low-level policy layer.

    For tasks with uses_locomotion_policy=True, this routes control through
    mujoco_extensions threaded_rollout so a lower-level policy can refine the
    high-level command before physics integration.

    The simulation maintains internal policy state (last_policy_output) to
    ensure smooth transitions between timesteps.
    """

    def __init__(
        self,
        init_task: str = "spot_base",
        task_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the hierarchical simulation.

        Args:
            init_task: Name of the task to initialize.
            task_registration_cfg: Optional task registration configuration.
        """
        super().__init__(init_task=init_task, task_registration_cfg=task_registration_cfg)

        self._systems = None
        self._last_policy_output = np.zeros(POLICY_OUTPUT_DIM)

        # Initialize C++ systems if the task uses a hierarchical policy layer.
        if self.task.uses_locomotion_policy:
            policy_path = get_task_registration(self.task.name).locomotion_policy_path
            if policy_path is None:
                raise ValueError(
                    f"Task '{self.task.name}' uses locomotion policy but no locomotion_policy_path is registered."
                )
            self._init_cpp_systems(policy_path)

    def _init_cpp_systems(self, policy_path: str | Path) -> None:
        """Initialize the C++ systems vector for threaded rollout.

        Args:
            policy_path: Path to the ONNX low-level policy file.
        """
        self._systems = create_systems_vector(
            self.task.model,  # Pass the MjModel directly
            str(policy_path),
            1,  # Single system for simulation
        )

    def step(self, command: np.ndarray) -> None:
        """Step the simulation forward.

        Routes to the C++ hierarchical rollout if systems are initialized,
        otherwise falls back to direct actuator control.

        Args:
            command: Control array in task format (task.nu dimensions).
                For hierarchical tasks, this is converted to the low-level
                policy command internally.
        """
        if self._systems is not None:
            if self.paused:
                return
            command = self.task.task_to_sim_ctrl(command)
            self._step_with_hierarchical_policy(command)
        else:
            super().step(command)

    def _step_with_hierarchical_policy(self, command: np.ndarray) -> None:
        """Execute a single step using the hierarchical rollout backend.

        Args:
            command: Command array for the low-level policy.
        """
        # Get current state
        state = np.concatenate([self.task.data.qpos, self.task.data.qvel])

        # Ensure command is 1D
        command = np.asarray(command, dtype=np.float64).flatten()

        # Reshape for threaded rollout:
        # states: (num_threads, nq+nv)
        # commands: (num_threads, num_timesteps, cmd_dim)
        # last_outputs: (num_threads, POLICY_OUTPUT_DIM)
        states = np.array([state], dtype=np.float64)
        commands = np.array([[command]], dtype=np.float64)
        last_outputs = np.array([self._last_policy_output], dtype=np.float64)

        # Run rollout
        self.task.pre_sim_step()
        out_states, out_sensors, policy_outputs = threaded_rollout(
            self._systems,
            states,
            commands,
            last_outputs,
            1,  # num_threads
            self.task.physics_substeps,
            DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME,
        )
        self.task.post_sim_step()

        # Update simulation state from rollout result
        final_state = np.array(out_states[0][-1])
        nq = self.task.model.nq
        self.task.data.qpos[:] = final_state[:nq]
        self.task.data.qvel[:] = final_state[nq:]
        self.task.data.time += self.task.dt

        # Compute derived quantities (xpos, xquat, etc.) for visualization
        mj_forward(self.task.model, self.task.data)

        # Update last policy output for continuity.
        self._last_policy_output = np.array(policy_outputs[0])

    def reset_policy_state(self) -> None:
        """Reset the internal policy state to zeros."""
        self._last_policy_output = np.zeros(POLICY_OUTPUT_DIM)

    def set_task(self, task_name: str) -> None:
        """Set the current task and reinitialize C++ systems if needed.

        Args:
            task_name: Name of the task to set.
        """
        super().set_task(task_name)

        # Reinitialize systems based on the new task's policy layer.
        if self.task.uses_locomotion_policy:
            policy_path = get_task_registration(self.task.name).locomotion_policy_path
            if policy_path is None:
                raise ValueError(
                    f"Task '{self.task.name}' uses locomotion policy but no locomotion_policy_path is registered."
                )
            self._init_cpp_systems(policy_path)
            self._last_policy_output = np.zeros(POLICY_OUTPUT_DIM)
        else:
            raise ValueError(
                f"Task '{self.task.name}' does not use a locomotion policy. "
                "Use MJSimulation instead of HierarchicalMJSimulation for this task."
            )

    @property
    def last_policy_output(self) -> np.ndarray:
        """Return the last low-level policy output."""
        return self._last_policy_output.copy()
