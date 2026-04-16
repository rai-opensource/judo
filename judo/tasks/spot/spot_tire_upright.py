# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotTireUpright task - get Spot to upright a tire that's lying flat.

This module mirrors the logic of starfish/dexterity/tasks/spot_tire_upright.py
but adapted for judo's standalone simulation framework.
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
    TIRE_HALF_WIDTH,
    TIRE_RADIUS,
)
from judo.tasks.spot.spot_utils import apply_quat_to_vec

XML_PATH = str(MODEL_PATH / "xml" / "spot_tire" / "robot.xml")


@dataclass
class SpotTireUprightConfig(SpotBaseConfig):
    """Configuration for the SpotTireUpright task.

    Values match starfish/dexterity/tasks/spot_tire_upright.py.
    """

    # Reward weights (from starfish)
    orientation_error_smoothing_width: float = 1.0
    w_tire_orientation: float = 200.0
    w_gripper_proximity: float = 10.0
    w_foot_proximity: float = 5.0
    w_torso_proximity: float = 5.0
    gripper_too_inside_tire_penalty: float = 150.0
    gripper_not_above_tire_penalty: float = 100.0
    w_controls: float = 2.0

    # Override base config
    fall_penalty: float = 10_000.0


class SpotTireUpright(SpotBase[SpotTireUprightConfig]):
    """Task for getting Spot to upright a tire that's lying flat.

    The goal is to manipulate the tire from a lying flat position to an
    upright position (y-axis horizontal). Spot can use its arm gripper
    and front legs to accomplish this task.

    This class mirrors starfish/dexterity/tasks/spot_tire_upright.py.
    """

    name: str = "spot_tire_upright"
    config_t: type[SpotTireUprightConfig] = SpotTireUprightConfig  # type: ignore[assignment]
    config: SpotTireUprightConfig

    def __init__(
        self,
        model_path: str = XML_PATH,
        config: SpotTireUprightConfig | None = None,
    ) -> None:
        """Initialize the SpotTireUpright task.

        Args:
            model_path: Path to the MuJoCo XML model.
            config: Optional task configuration.
        """
        # Starfish uses use_legs=True for this task
        super().__init__(
            model_path=model_path,
            use_arm=True,
            use_gripper=False,
            use_legs=True,
            use_torso=False,
            config=config,
        )

        # Get sensor indices
        self._setup_sensor_indices()

    def _setup_sensor_indices(self) -> None:
        """Setup sensor indices for reward computation."""
        # Object pose index (tire joint)
        self.object_pose_idx = self.get_joint_position_start_index("tire_joint")

        # Sensor indices
        self.tire_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")
        self.fl_pos_idx = self.get_sensor_start_index("fl_pos")
        self.fr_pos_idx = self.get_sensor_start_index("fr_pos")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the tire uprighting task.

        This matches the reward function in starfish/dexterity/tasks/spot_tire_upright.py.

        Args:
            states: Rolled out states, shape (num_rollouts, T, nq+nv).
            sensors: Sensor readings, shape (num_rollouts, T, nsensor).
            controls: Control inputs, shape (num_rollouts, T, nu).
            system_metadata: Optional metadata from the system.

        Returns:
            Rewards for each rollout, shape (num_rollouts,).
        """
        batch_size = states.shape[0]

        # (batch, horizon, size)
        qpos = states[..., : self.model.nq]

        #######
        ### Proximity rewards
        #######

        # Compute unit vector pointing from tire to torso
        W_p_tire = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        W_p_torso = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        W_p_tire_torso = W_p_torso - W_p_tire
        W_u_tire_torso = W_p_tire_torso / (np.linalg.norm(W_p_tire_torso, axis=-1, keepdims=True) + 1e-8)

        # Compute the desired gripper position
        W_p_gripper_des = W_p_tire + (TIRE_RADIUS - 0.05) * W_u_tire_torso
        W_p_gripper_des[..., 2] = TIRE_HALF_WIDTH + 0.1

        # Compute the distance from gripper to desired gripper position
        W_p_gripper = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]
        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            W_p_gripper - W_p_gripper_des, axis=-1
        ).mean(axis=-1)

        # Compute the desired right foot position (using quaternion rotation like starfish)
        pi_over_4_yaw_quat = np.array([np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)])
        W_u_tire_torso_rotated = apply_quat_to_vec(quat=pi_over_4_yaw_quat, vec=W_u_tire_torso)
        W_p_right_foot_des = W_p_tire + TIRE_RADIUS * W_u_tire_torso_rotated
        W_p_right_foot_des[..., 2] = 0.1

        # Compute the desired left foot position
        pi_over_4_yaw_quat_neg = np.array([np.cos(np.pi / 8), 0, 0, np.sin(-np.pi / 8)])
        W_u_tire_torso_rotated = apply_quat_to_vec(quat=pi_over_4_yaw_quat_neg, vec=W_u_tire_torso)
        W_p_left_foot_des = W_p_tire + TIRE_RADIUS * W_u_tire_torso_rotated
        W_p_left_foot_des[..., 2] = 0.1

        # Compute the distance from right foot to desired foot position
        W_p_foot = sensors[..., self.fr_pos_idx : self.fr_pos_idx + 3]
        right_foot_proximity_reward = -self.config.w_foot_proximity * np.linalg.norm(
            W_p_foot - W_p_right_foot_des, axis=-1
        ).mean(axis=-1)

        # Compute the distance from left foot to desired foot position
        W_p_foot = sensors[..., self.fl_pos_idx : self.fl_pos_idx + 3]
        left_foot_proximity_reward = -self.config.w_foot_proximity * np.linalg.norm(
            W_p_foot - W_p_left_foot_des, axis=-1
        ).mean(axis=-1)

        # Use the maximum reward of both feet
        foot_proximity_reward = np.maximum(right_foot_proximity_reward, left_foot_proximity_reward)

        # Compute distance from torso to desired torso position
        W_p_torso_des = W_p_tire + 0.75 * W_u_tire_torso
        W_p_torso_des[..., 2] = STANDING_HEIGHT  # Enforce standing height

        # Compute the distance from torso to desired torso position
        torso_proximity_reward = -self.config.w_torso_proximity * np.linalg.norm(
            W_p_torso - W_p_torso_des, axis=-1
        ).mean(axis=-1)

        #######
        ### Goal rewards
        #######

        # Compute orientation reward
        tire_y_axis = sensors[..., self.tire_y_axis_idx : self.tire_y_axis_idx + 3]
        orientation_error = np.abs(tire_y_axis[..., 2])  # 0 to 1
        orientation_error_smooth = np.exp(orientation_error / self.config.orientation_error_smoothing_width)  # 1 to e
        orientation_reward = -self.config.w_tire_orientation * orientation_error_smooth.mean(axis=-1)

        #######
        ### Penalize bad behavior
        #######

        # Don't put the gripper too close to tire center
        gripper_distance_from_tire = np.linalg.norm(W_p_gripper - W_p_tire, axis=-1)
        gripper_inside_tire_reward = -self.config.gripper_too_inside_tire_penalty * (
            gripper_distance_from_tire < (TIRE_RADIUS * 0.5)
        ).mean(axis=-1)

        # Don't put the gripper below the tire if too far from the tire center
        gripper_height = W_p_gripper[..., 2]
        gripper_not_above_tire = gripper_height < 2 * TIRE_HALF_WIDTH + 0.05
        gripper_too_far_from_tire = gripper_distance_from_tire > TIRE_RADIUS
        gripper_not_above_tire_reward = -self.config.gripper_not_above_tire_penalty * (
            np.logical_and(gripper_not_above_tire, gripper_too_far_from_tire)
        ).mean(axis=-1)

        # Check if any state in the rollout has spot fallen
        body_height = qpos[..., self.body_pose_idx + 2]
        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Compute a penalty to prefer small commands (full controls, like starfish)
        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        # Validate shapes
        assert orientation_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert foot_proximity_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_inside_tire_reward.shape == (batch_size,)
        assert gripper_not_above_tire_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        reward = (
            orientation_reward
            + gripper_proximity_reward
            + foot_proximity_reward
            + torso_proximity_reward
            + gripper_inside_tire_reward
            + gripper_not_above_tire_reward
            + spot_fallen_reward
            + controls_reward
        )
        return reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and tire - tire starts lying flat.

        Returns:
            Initial qpos array with random robot and tire positions.
        """
        for _ in range(100):
            # Random tire position
            tire_x = np.random.uniform(-2, 2)
            tire_y = np.random.uniform(-2, 2)
            tire_z = TIRE_HALF_WIDTH  # Lying flat (using TIRE_HALF_WIDTH like starfish)

            # Random tire orientation (lying flat with random yaw)
            if np.random.random() < 0.5:
                # +90 degree roll
                tire_quat = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])
            else:
                # -90 degree roll
                tire_quat = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0])

            # Apply random yaw
            random_yaw = np.random.uniform(0, 2 * np.pi)
            yaw_quat = np.array([np.cos(random_yaw / 2), 0, 0, np.sin(random_yaw / 2)])

            # Quaternion multiplication: yaw_quat * tire_quat
            w1, x1, y1, z1 = yaw_quat
            w2, x2, y2, z2 = tire_quat
            tire_quat_final = np.array(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ]
            )

            # Random robot position with random yaw
            robot_x = np.random.uniform(-2, 2)
            robot_y = np.random.uniform(-2, 2)
            robot_z = STANDING_HEIGHT
            random_yaw_robot = np.random.uniform(0, 2 * np.pi)
            robot_quat = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])

            # Check that robot and tire are sufficiently far apart
            robot_pos = np.array([robot_x, robot_y, robot_z])
            tire_pos = np.array([tire_x, tire_y, tire_z])
            if np.linalg.norm(robot_pos[:2] - tire_pos[:2]) > 1.0:  # Match starfish inner_radius
                return np.array(
                    [
                        *robot_pos,
                        *robot_quat,
                        *LEGS_STANDING_POS,
                        *self.reset_arm_pos,
                        *tire_pos,
                        *tire_quat_final,
                    ]
                )

        # Fallback: use default positions
        tire_pose = np.array([2.0, 0.0, TIRE_HALF_WIDTH, np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        return np.array(
            [
                0.0,
                0.0,
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
                *tire_pose,
            ]
        )

    def success(
        self,
        model: MjModel,
        data: MjData,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Check if the tire is upright (y-axis horizontal).

        Args:
            model: MuJoCo model.
            data: MuJoCo data.
            metadata: Optional task metadata.

        Returns:
            True if the tire is upright.
        """
        # Get tire y-axis sensor data
        tire_y_axis = data.sensordata[self.tire_y_axis_idx : self.tire_y_axis_idx + 3]

        # Check if y-axis is horizontal (z-component close to 0) and Spot is still standing
        upright = np.abs(tire_y_axis[2]) <= 0.1
        return bool(upright and super().success(model, data, metadata))
