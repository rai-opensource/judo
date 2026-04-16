# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec

from judo import MODEL_PATH
from judo.utils.assets import download_and_extract_meshes


@dataclass
class TaskConfig:
    """Base task configuration dataclass."""


ConfigT = TypeVar("ConfigT", bound=TaskConfig)


class Task(ABC, Generic[ConfigT]):
    """Task definition."""

    config_t: type[ConfigT]

    def __init__(self, model_path: Path | str = "", sim_model_path: Path | str | None = None) -> None:
        """Initialize the Mujoco task."""
        if not model_path:
            raise ValueError("Model path must be provided.")
        download_and_extract_meshes(extract_root=str(MODEL_PATH), repo="bdaiinstitute/judo", asset_name="meshes.zip")
        self.config = self.config_t()
        self.spec = MjSpec.from_file(str(model_path))
        self._process_spec()
        self.model = self.spec.compile()
        self.data = MjData(self.model)
        self.model_path = model_path
        self.sim_model = self.model if sim_model_path is None else MjModel.from_xml_path(str(sim_model_path))

    def _process_spec(self) -> None:
        """Hook for subclasses to modify spec before compile. No-op by default."""

    @property
    def time(self) -> float:
        """Returns the current simulation time."""
        return self.data.time

    @time.setter
    def time(self, value: float) -> None:
        """Sets the current simulation time."""
        self.data.time = value

    @abstractmethod
    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Abstract reward function for task.

        Args:
            states: The rolled out states (after the initial condition). Shape=(num_rollouts, T, nq + nv).
            sensors: The rolled out sensors readings. Shape=(num_rollouts, T, total_num_sensor_dims).
            controls: The rolled out controls. Shape=(num_rollouts, T, nu).
            config: The current task config (passed in from the top-level controller).
            system_metadata: Any additional metadata from the system that is useful for computing the reward. For
                example, in the cube rotation task, the system could pass in new goal cube orientations to the
                controller here.

        Returns:
            rewards: The reward for each rollout. Shape=(num_rollouts,).
        """

    @property
    def nu(self) -> int:
        """Number of control inputs. The same as the MjModel for this task."""
        return self.model.nu

    @property
    def locomotion_policy_path(self) -> str | None:
        """Path to locomotion policy for this task, or None if not used.

        Override in tasks that use a learned locomotion policy
        (e.g., Spot tasks that run an ONNX policy at 50Hz).
        """
        return None

    @property
    def uses_locomotion_policy(self) -> bool:
        """Whether this task uses a locomotion policy for simulation."""
        return self.locomotion_policy_path is not None

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """Mujoco actuator limits for this task."""
        limits = self.model.actuator_ctrlrange
        limited: np.ndarray = self.model.actuator_ctrllimited.astype(bool)  # type: ignore
        limits[~limited] = np.array([-np.inf, np.inf], dtype=limits.dtype)  # if not limited, set to inf
        return limits  # type: ignore

    def reset(self) -> None:
        """Reset behavior for task. Sets config + velocities to zeros."""
        self.data.qpos = np.zeros_like(self.data.qpos)
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the task was completed successfully. Override in subclasses."""
        return False

    @property
    def physics_substeps(self) -> int:
        """Number of physics steps per control step."""
        return 1

    @property
    def dt(self) -> float:
        """Effective timestep (physics_substeps * model timestep)."""
        return self.model.opt.timestep * self.physics_substeps

    def pre_rollout(self, curr_state: np.ndarray) -> None:
        """Pre-rollout behavior for task (does nothing by default).

        Args:
            curr_state: Current state of the task. Shape=(nq + nv,).
        """

    def post_rollout(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Post-rollout behavior for task (does nothing by default).

        Same inputs as in reward function.
        """

    def pre_sim_step(self) -> None:
        """Pre-simulation step behavior for task (does nothing by default)."""

    def post_sim_step(self) -> None:
        """Post-simulation step behavior for task (does nothing by default)."""

    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns metadata from the simulation.

        We need this function because the simulation thread runs separately from the controller thread, but there are
        task objects in both. This function is used to pass information about the simulation's version of the task to
        the controller's version of the task.

        For example, the LeapCube task has a goal quaternion that is updated in the simulation thread based on whether
        the goal was reached (which the controller thread doesn't know about). When a new goal is set, it must be passed
        to the controller thread via this function.
        """
        return {}

    def optimizer_warm_start(self) -> np.ndarray:
        """Returns a warm start for the optimizer.

        This is used to provide an initial guess for the optimizer when optimizing the task before any iterations.
        """
        return np.zeros(self.nu)

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        """Convert task control format to simulation control format.

        Override in subclasses that use a different control format than the simulator
        (e.g., Spot tasks that convert to 25-dim locomotion policy commands).

        Args:
            controls: Controls in task format, shape (..., nu).

        Returns:
            Controls in simulation format (same as input by default).
        """
        return controls

    def get_sensor_start_index(self, sensor_name: str) -> int:
        """Returns the starting index of a sensor in the 'sensors' array given the sensor's name.

        Args:
            sensor_name: The name of the sensor to get the index of.
        """
        return self.model.sensor(sensor_name).adr[0]

    def get_joint_position_start_index(self, joint_name: str) -> int:
        """Returns the starting index of a joint's position in the 'states' array given the joint's name.

        Args:
            joint_name: The name of the joint to get the starting index in the position of the state array.
        """
        return self.model.jnt_qposadr[self.model.joint(joint_name).id]

    def get_joint_velocity_start_index(self, joint_name: str) -> int:
        """Returns the starting index of a joint's velocity in the 'states' array given the joint's name.

        NOTE: This is the index of the joint's velocity in the state array, which is after the position indices!

        Args:
            joint_name: The name of the joint to get the starting index in the state array of.
        """
        return self.model.nq + self.model.jnt_dofadr[self.model.joint(joint_name).id]
