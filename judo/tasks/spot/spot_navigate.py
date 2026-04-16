# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotNavigate task - navigate Spot to a goal location.

Adapted from starfish/dexterity/tasks/spot_navigate.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from judo.tasks.spot.spot_base import XML_PATH, SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT
from judo.utils.fields import np_1d_field


@dataclass
class SpotNavigateConfig(SpotBaseConfig):
    """Configuration for the SpotNavigate task."""

    w_goal: float = 60.0
    fall_penalty: float = 2500.0
    w_controls: float = 0.0
    goal_distance_threshold: float = 0.5
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, STANDING_HEIGHT]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )


class SpotNavigate(SpotBase[SpotNavigateConfig]):
    """Task getting Spot to navigate to a desired goal location."""

    name: str = "spot_navigate"
    config_t: type[SpotNavigateConfig] = SpotNavigateConfig  # type: ignore[assignment]
    config: SpotNavigateConfig

    def __init__(
        self,
        model_path: str = XML_PATH,
        config: SpotNavigateConfig | None = None,
    ) -> None:
        """Initialize the SpotNavigate task."""
        super().__init__(model_path=model_path, use_arm=False, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the navigate task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]

        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        goal_reward = -self.config.w_goal * np.linalg.norm(body_pos - self.config.goal_pos[None, None], axis=-1).mean(
            -1
        )

        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return spot_fallen_reward + goal_reward + controls_reward

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if Spot reached the goal and is still standing."""
        body_pos = data.qpos[self.body_pose_idx : self.body_pose_idx + 3]
        at_goal = np.linalg.norm(body_pos - self.config.goal_pos) <= self.config.goal_distance_threshold
        return bool(at_goal and super().success(model, data, metadata))

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the navigate task with random initial xy position."""
        xy = np.random.uniform(-2.0, 2.0, size=2)
        return np.array([*xy, STANDING_HEIGHT, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos])
