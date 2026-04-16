# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotTireRoll task - roll a tire to a goal location.

Adapted from starfish/dexterity/tasks/spot_tire_roll.py.
Shares reward structure with SpotWheelRimRoll.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
    TIRE_RADIUS,
    Z_AXIS,
)
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml" / "spot_tire" / "robot.xml")


@dataclass
class SpotTireRollConfig(SpotBaseConfig):
    """Configuration for the SpotTireRoll task."""

    fall_penalty: float = 5000.0
    tire_fallen_threshold: float = 0.1
    w_goal: float = 60.0
    w_torso_proximity: float = 1.0
    torso_goal_offset: float = 1.0
    w_gripper_proximity: float = 1.0
    gripper_goal_offset: float = 0.15
    gripper_goal_altitude: float = 0.05
    w_tire_linear_velocity: float = 10.0
    w_tire_angular_velocity: float = 0.30
    w_controls: float = 0.0
    goal_distance_threshold: float = 0.5
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, TIRE_RADIUS]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )


class SpotTireRoll(SpotBase[SpotTireRollConfig]):
    """Task getting Spot to roll a tire to a desired goal location."""

    name: str = "spot_tire_roll"
    config_t: type[SpotTireRollConfig] = SpotTireRollConfig  # type: ignore[assignment]
    config: SpotTireRollConfig

    def __init__(
        self,
        model_path: str = XML_PATH,
        config: SpotTireRollConfig | None = None,
    ) -> None:
        """Initialize the SpotTireRoll task."""
        super().__init__(model_path=model_path, use_arm=True, use_gripper=True, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("tire_joint")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")

        # Velocity index for tire
        self.object_vel_idx = self.model.jnt_dofadr[self.model.joint("tire_joint").id]

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the tire roll task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        tire_linear_velocity = qvel[..., self.object_vel_idx : self.object_vel_idx + 3]
        tire_angular_velocity = qvel[..., self.object_vel_idx + 3 : self.object_vel_idx + 6]
        gripper_pos = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]
        object_y_axis = sensors[..., self.object_y_axis_idx : self.object_y_axis_idx + 3]

        tire_to_goal = self.config.goal_pos - object_pos
        tire_to_goal_norm = np.linalg.norm(tire_to_goal, axis=-1, keepdims=True)
        tire_to_goal_direction = tire_to_goal / (1e-2 + tire_to_goal_norm)

        gripper_goal = object_pos - self.config.gripper_goal_offset * tire_to_goal_direction
        gripper_goal[..., 2] = self.config.gripper_goal_altitude
        torso_goal = object_pos - self.config.torso_goal_offset * tire_to_goal_direction

        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        tire_fallen_reward = -self.config.fall_penalty * np.abs(
            np.dot(object_y_axis, Z_AXIS) > self.config.tire_fallen_threshold
        ).sum(axis=-1)

        goal_reward = -self.config.w_goal * np.linalg.norm(object_pos - self.config.goal_pos, axis=-1).mean(-1)

        torso_proximity_reward = -self.config.w_torso_proximity * np.linalg.norm(body_pos - torso_goal, axis=-1).mean(
            -1
        )

        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            gripper_goal - gripper_pos, axis=-1
        ).mean(-1)

        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        tire_linear_velocity_reward = -self.config.w_tire_linear_velocity * np.linalg.norm(
            tire_linear_velocity, axis=-1
        ).mean(-1)

        tire_angular_velocity_reward = -self.config.w_tire_angular_velocity * np.linalg.norm(
            tire_angular_velocity, axis=-1
        ).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        return (
            spot_fallen_reward
            + tire_fallen_reward
            + goal_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + controls_reward
            + tire_linear_velocity_reward
            + tire_angular_velocity_reward
        )

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the tire is at goal, upright, and Spot is still standing."""
        object_pos = data.qpos[self.object_pose_idx : self.object_pose_idx + 3]
        tire_y_axis = data.sensordata[self.object_y_axis_idx : self.object_y_axis_idx + 3]
        at_goal = np.linalg.norm(object_pos - self.config.goal_pos) <= self.config.goal_distance_threshold
        upright = np.abs(tire_y_axis[2]) <= 0.1
        return bool(at_goal and upright and super().success(model, data, metadata))

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the tire roll task."""
        standing_pose = np.array([0, 0, STANDING_HEIGHT])
        robot_radius = 1.0
        reset_pose = (np.random.rand(7) - 0.5) * 3.0
        reset_pose[2] = TIRE_RADIUS
        reset_pose[3:] = [1, 0, 0, 0]
        while np.linalg.norm(reset_pose[:3] - standing_pose) < robot_radius:
            reset_pose = (np.random.rand(7) - 0.5) * 3.0
            reset_pose[2] = TIRE_RADIUS
            reset_pose[3:] = [1, 0, 0, 0]
        return np.array([*standing_pose, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos, *reset_pose])
