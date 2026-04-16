# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import quadratic_norm
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml/cylinder_push.xml")
GOAL_DISTANCE_THRESHOLD = 0.3


@slider("w_pusher_proximity", 0.0, 5.0, 0.1)
@dataclass
class CylinderPushConfig(TaskConfig):
    """Reward configuration for the cylinder push task."""

    w_pusher_proximity: float = 0.5
    w_pusher_velocity: float = 0.0
    w_cart_position: float = 0.1
    pusher_goal_offset: float = 0.25
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0]),
        names=["x", "y"],
        mins=[-1.0, -1.0],
        maxs=[1.0, 1.0],
        steps=[0.01, 0.01],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
        xyz_vis_defaults=[0.0, 0.0, 0.0],
    )


class CylinderPush(Task[CylinderPushConfig]):
    """Defines the cylinder push balancing task."""

    name: str = "cylinder_push"
    config_t: type[CylinderPushConfig] = CylinderPushConfig

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the cylinder push task."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the cylinder push reward from MJPC.

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The cylinder push reward has four terms:
            * `pusher_reward`, penalizing the distance between the pusher and the cart.
            * `velocity_reward` penalizing squared linear velocity of the pusher.
            * `goal_reward`, penalizing the distance from the cart to the goal.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        batch_size = states.shape[0]

        pusher_pos = states[..., 0:2]
        cart_pos = states[..., 2:4]
        pusher_vel = states[..., 4:6]
        cart_goal = self.config.goal_pos[0:2]

        cart_to_goal = cart_goal - cart_pos
        cart_to_goal_norm = np.linalg.norm(cart_to_goal, axis=-1, keepdims=True)
        cart_to_goal_direction = cart_to_goal / cart_to_goal_norm

        pusher_goal = cart_pos - self.config.pusher_goal_offset * cart_to_goal_direction

        pusher_proximity = quadratic_norm(pusher_pos - pusher_goal)
        pusher_reward = -self.config.w_pusher_proximity * pusher_proximity.sum(-1)

        velocity_reward = -self.config.w_pusher_velocity * quadratic_norm(pusher_vel).sum(-1)

        goal_proximity = quadratic_norm(cart_pos - cart_goal)
        goal_reward = -self.config.w_cart_position * goal_proximity.sum(-1)

        assert pusher_reward.shape == (batch_size,)
        assert velocity_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)

        return pusher_reward + velocity_reward + goal_reward

    def success(self, model: mujoco.MjModel, data: mujoco.MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the cart is close to the goal position."""
        cart_pos = data.qpos[2:4]
        goal_pos = self.config.goal_pos[0:2]
        return bool(np.linalg.norm(cart_pos - goal_pos) < GOAL_DISTANCE_THRESHOLD)

    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        theta = 2 * np.pi * np.random.rand(2)
        self.data.qpos = np.array(
            [
                np.cos(theta[0]),
                np.sin(theta[0]),
                2 * np.cos(theta[1]),
                2 * np.sin(theta[1]),
            ]
        )
        self.data.qvel = np.zeros(4)
        mujoco.mj_forward(self.model, self.data)
