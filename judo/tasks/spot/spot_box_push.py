# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotBoxPush task - push a box to a goal location.

Adapted from starfish/dexterity/tasks/spot_box_push.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import BOX_HALF_LENGTH, LEGS_STANDING_POS, STANDING_HEIGHT, Z_AXIS
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml" / "spot_box" / "robot.xml")

RADIUS_MIN = 1.0
RADIUS_MAX = 2.0


@dataclass
class SpotBoxPushConfig(SpotBaseConfig):
    """Configuration for the SpotBoxPush task."""

    w_goal: float = 60.0
    w_orientation: float = 15.0
    w_torso_proximity: float = 0.1
    w_gripper_proximity: float = 4.0
    orientation_threshold: float = 0.5
    fall_penalty: float = 2500.0
    w_controls: float = 0.0
    goal_distance_threshold: float = 0.5
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, BOX_HALF_LENGTH]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )


class SpotBoxPush(SpotBase[SpotBoxPushConfig]):
    """Task getting Spot to push a box to a desired goal location."""

    name: str = "spot_box_push"
    config_t: type[SpotBoxPushConfig] = SpotBoxPushConfig  # type: ignore[assignment]
    config: SpotBoxPushConfig

    def __init__(
        self,
        model_path: str = XML_PATH,
        config: SpotBoxPushConfig | None = None,
    ) -> None:
        """Initialize the SpotBoxPush task."""
        super().__init__(model_path=model_path, use_arm=True, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("box_joint")
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the box push task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        object_y_axis = sensors[..., self.object_y_axis_idx : self.object_y_axis_idx + 3]
        gripper_pos = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]

        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        goal_reward = -self.config.w_goal * np.linalg.norm(object_pos - self.config.goal_pos[None, None], axis=-1).mean(
            -1
        )

        box_orientation_reward = -self.config.w_orientation * np.abs(
            np.dot(object_y_axis, Z_AXIS) > self.config.orientation_threshold
        ).sum(axis=-1)

        torso_proximity_reward = self.config.w_torso_proximity * np.linalg.norm(body_pos - object_pos, axis=-1).mean(-1)

        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            gripper_pos - object_pos, axis=-1
        ).mean(-1)

        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        return (
            spot_fallen_reward
            + goal_reward
            + box_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + controls_reward
        )

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the box reached the goal and Spot is still standing."""
        object_pos = data.qpos[self.object_pose_idx : self.object_pose_idx + 3]
        at_goal = np.linalg.norm(object_pos - self.config.goal_pos) <= self.config.goal_distance_threshold
        return bool(at_goal and super().success(model, data, metadata))

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the box push task."""
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_xy = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + np.random.randn(2)
        reset_object_pose = np.array([*object_xy, BOX_HALF_LENGTH, 1, 0, 0, 0])
        return np.array(
            [
                *np.random.randn(2),
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
                *reset_object_pose,
            ]
        )
