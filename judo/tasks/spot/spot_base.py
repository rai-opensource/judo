# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Base task class for Spot locomotion and manipulation.

This module mirrors the structure of starfish/dexterity/tasks/spot_base.py
but adapted for judo's standalone simulation framework.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import mujoco
import numpy as np
from mujoco import MjData, MjModel

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.tasks.spot.spot_constants import (
    ARM_CMD_INDS,
    ARM_JOINT_NAMES,
    ARM_SOFT_LOWER_JOINT_LIMITS,
    ARM_SOFT_UPPER_JOINT_LIMITS,
    ARM_STOWED_POS,
    ARM_UNSTOWED_POS,
    BASE_SOFT_LIMITS,
    BASE_VEL_CMD_INDS,
    FRONT_LEG_CMD_INDS,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    LEG_JOINT_NAMES_BOSDYN,
    LEG_SOFT_LOWER_JOINT_LIMITS,
    LEG_SOFT_UPPER_JOINT_LIMITS,
    LEGS_STANDING_POS,
    LEGS_STANDING_POS_RL,
    SPOT_LOCOMOTION_POLICY_PATH,
    STANDING_HEIGHT,
    STANDING_HEIGHT_CMD,
    TORSO_CMD_INDS,
    TORSO_LOWER,
    TORSO_UPPER,
)

XML_PATH = str(MODEL_PATH / "xml" / "spot_primitive" / "robot.xml")


@dataclass
class GoalPositions:
    """Goal positions for Spot tasks."""

    origin: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.0]))
    blue_cross: np.ndarray = field(default_factory=lambda: np.array([2.77, 0.71, 0.3]))
    black_cross: np.ndarray = field(default_factory=lambda: np.array([1.5, -1.5, 0.275]))


@dataclass
class SpotBaseConfig(TaskConfig):
    """Base configuration for Spot tasks.

    Values match starfish/dexterity/tasks/spot_base.py.
    """

    fall_penalty: float = 2500.0
    spot_fallen_threshold: float = 0.35  # Torso height in meters (starfish uses 35 cm)
    w_goal: float = 60.0
    w_controls: float = 0.0


ConfigT = TypeVar("ConfigT", bound=SpotBaseConfig)


class SpotBase(Task[ConfigT], Generic[ConfigT]):
    """Flexible base task for Spot locomotion/skills.

    This class mirrors starfish/dexterity/tasks/spot_base.py but adapted
    for judo's standalone simulation framework.

    Controls are a compact vector mapped to the 25-dim policy command:
    - Base only:                        [base_vel(3)]
    - Base + Arm:                       [base_vel(3), arm_cmd(7)]
    - Base + Legs:                      [base_vel(3), front_leg_cmd(6), leg_selection(1)]
    - Base + Arm + Legs:                [base_vel(3), arm_cmd(7), front_leg_cmd(6), leg_selection(1)]
    - With Torso:                       [..., torso_cmd(3)]  (appended to any of the above)

    The mapping to the 25-dim policy command is done in task_to_sim_ctrl.
    """

    name: str = "spot_base"
    config_t: type[SpotBaseConfig] = SpotBaseConfig  # type: ignore[assignment]

    def _process_spec(self) -> None:
        """Replace mesh and texture paths with correct resolved paths.

        MjSpec resolves mesh file paths relative to the included file's directory,
        ignoring the parent model's meshdir. We fix this by rewriting:
        - Spot robot meshes → mujoco_menagerie assets (from robot_descriptions)
        - Object meshes (tire, wheel_rim, etc.) → MODEL_PATH/meshes/objects/...
        """
        from robot_descriptions import spot_mj_description  # noqa: PLC0415

        menagerie_dir = Path(spot_mj_description.PACKAGE_PATH)
        menagerie_assets = menagerie_dir / "assets"
        meshes_root = MODEL_PATH / "meshes"
        for mesh in self.spec.meshes:
            if "spot/meshes/" in mesh.file:
                basename = Path(mesh.file).name
                mesh.file = str(menagerie_assets / basename)
            elif "objects/" in mesh.file:
                idx = mesh.file.index("objects/")
                mesh.file = str(meshes_root / mesh.file[idx:])
        for texture in self.spec.textures:
            if "spot/textures/" in texture.file:
                texture.file = str(menagerie_dir / "spot.png")

    @property
    def physics_substeps(self) -> int:  # type: ignore[override]
        """Number of physics steps per control step."""
        return 2

    @property
    def locomotion_policy_path(self) -> str:
        """Path to Spot locomotion policy."""
        return str(SPOT_LOCOMOTION_POLICY_PATH)

    def __init__(
        self,
        model_path: str = XML_PATH,
        use_arm: bool = True,
        use_gripper: bool = False,
        use_legs: bool = False,
        use_torso: bool = False,
        config: SpotBaseConfig | None = None,
    ) -> None:
        """Initialize the Spot base task.

        Args:
            model_path: Path to the MuJoCo XML model.
            use_arm: Whether to include arm control in the action space.
            use_gripper: Whether to include gripper control (requires use_arm).
            use_legs: Whether to include leg manipulation (front legs only).
            use_torso: Whether to include torso control (roll, pitch, height).
            config: Optional task configuration.
        """
        super().__init__(model_path=model_path)
        if config is not None:
            self.config = config

        self.use_arm = use_arm
        self.use_gripper = use_gripper
        self.use_legs = use_legs
        self.use_torso = use_torso

        # Selection indices (set by set_command_values)
        self.leg_selection_index: int | None = None
        self.gripper_selection_index: int | None = None

        self.set_command_values()

        # Default 25-dim policy command
        self.default_policy_command = np.array(
            [0, 0, 0] + list(ARM_STOWED_POS) + [0] * 12 + [0, 0, STANDING_HEIGHT_CMD]
        )

        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.reset()

    @property
    def nu(self) -> int:
        """Number of control inputs for this task."""
        return len(self.default_command)

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """Control bounds for the task action space."""
        BASE_LOWER = -BASE_SOFT_LIMITS
        BASE_UPPER = BASE_SOFT_LIMITS
        GRIPPER_LOWER = GRIPPER_OPEN_POS if self.use_gripper else GRIPPER_CLOSED_POS
        GRIPPER_UPPER = GRIPPER_CLOSED_POS
        ARM_LOWER = np.concatenate((ARM_SOFT_LOWER_JOINT_LIMITS[:-1], [GRIPPER_LOWER]))
        ARM_UPPER = np.concatenate((ARM_SOFT_UPPER_JOINT_LIMITS[:-1], [GRIPPER_UPPER]))
        LEGS_LOWER = LEG_SOFT_LOWER_JOINT_LIMITS[0:6]
        LEGS_UPPER = LEG_SOFT_UPPER_JOINT_LIMITS[0:6]
        LEG_SELECTION_LOWER = -np.ones(1)
        LEG_SELECTION_UPPER = np.ones(1)
        GRIPPER_SELECTION_LOWER = -np.ones(1)
        GRIPPER_SELECTION_UPPER = np.ones(1)
        TORSO_BOUNDS_LOWER = TORSO_LOWER
        TORSO_BOUNDS_UPPER = TORSO_UPPER

        if not self.use_arm and not self.use_legs:  # Base only
            lower_components = [BASE_LOWER]
            upper_components = [BASE_UPPER]
        elif self.use_arm and not self.use_legs:  # Base and arm
            lower_components = [BASE_LOWER, ARM_LOWER]
            upper_components = [BASE_UPPER, ARM_UPPER]
            if self.use_gripper:
                lower_components.append(GRIPPER_SELECTION_LOWER)
                upper_components.append(GRIPPER_SELECTION_UPPER)
        elif not self.use_arm and self.use_legs:  # Base and legs
            lower_components = [BASE_LOWER, LEGS_LOWER, LEG_SELECTION_LOWER]
            upper_components = [BASE_UPPER, LEGS_UPPER, LEG_SELECTION_UPPER]
        else:  # Base, arm, and legs
            lower_components = [BASE_LOWER, ARM_LOWER]
            upper_components = [BASE_UPPER, ARM_UPPER]
            if self.use_gripper:
                lower_components.append(GRIPPER_SELECTION_LOWER)
                upper_components.append(GRIPPER_SELECTION_UPPER)
            lower_components.extend([LEGS_LOWER, LEG_SELECTION_LOWER])
            upper_components.extend([LEGS_UPPER, LEG_SELECTION_UPPER])

        # Add torso if enabled
        if self.use_torso:
            lower_components.append(TORSO_BOUNDS_LOWER)
            upper_components.append(TORSO_BOUNDS_UPPER)

        lower_bound = np.concatenate(lower_components)
        upper_bound = np.concatenate(upper_components)

        return np.stack([lower_bound, upper_bound], axis=-1)

    def set_command_values(self) -> None:
        """Update default_command and command_mask based on enabled features."""
        # Reset selection indices
        self.leg_selection_index = None
        self.gripper_selection_index = None

        command_values: list[float]
        if not self.use_arm and not self.use_legs:  # Base only
            command_values = [0, 0, 0]
            command_mask = BASE_VEL_CMD_INDS
        elif self.use_arm and not self.use_legs:  # Base and arm
            command_values = [0, 0, 0, *ARM_UNSTOWED_POS]
            command_mask = BASE_VEL_CMD_INDS + ARM_CMD_INDS
            if self.use_gripper:
                command_values.append(0.0)  # gripper selection
                self.gripper_selection_index = len(command_values) - 1
        elif not self.use_arm and self.use_legs:  # Base and legs
            command_values = [0, 0, 0, *LEGS_STANDING_POS[0:6], 0]
            command_mask = BASE_VEL_CMD_INDS + FRONT_LEG_CMD_INDS
            self.leg_selection_index = len(command_values) - 1
        else:  # Base, arm, and legs
            command_values = [0, 0, 0, *ARM_UNSTOWED_POS]
            command_mask = BASE_VEL_CMD_INDS + ARM_CMD_INDS + FRONT_LEG_CMD_INDS
            if self.use_gripper:
                command_values.append(0.0)  # gripper selection
                self.gripper_selection_index = len(command_values) - 1
            command_values.extend([*LEGS_STANDING_POS[0:6], 0])
            self.leg_selection_index = len(command_values) - 1

        # Add torso if enabled
        if self.use_torso:
            command_values.extend([0, 0, STANDING_HEIGHT])  # roll, pitch, height
            command_mask = command_mask + TORSO_CMD_INDS

        self.default_command = np.array(command_values)
        self.command_mask = np.array(command_mask)

    def apply_selection_mask(self, controls: np.ndarray) -> np.ndarray:
        """Activate or deactivate leg and gripper commands based on selection values.

        leg selection:
        -1.0 to -0.5: manipulation with left leg
        -0.5 to +0.5: no leg manipulation
        +0.5 to +1.0: manipulation with right leg

        gripper selection:
        -1.0 to 0.0: gripper closed
        0.0 to +1.0: gripper open

        Args:
            controls: Control array with selection indices.

        Returns:
            Controls with selection indices removed and masks applied.
        """
        if self.leg_selection_index is None and self.gripper_selection_index is None:
            return controls

        added_dim = False
        if controls.ndim == 1:
            controls = np.expand_dims(controls, axis=0)
            added_dim = True

        controls = controls.copy()

        # Collect selection indices to remove
        selection_indices = []
        if self.gripper_selection_index is not None:
            selection_indices.append(self.gripper_selection_index)
        if self.leg_selection_index is not None:
            selection_indices.append(self.leg_selection_index)

        # Apply gripper mask
        if self.use_arm and self.use_gripper and self.gripper_selection_index is not None:
            selection = controls[..., self.gripper_selection_index]
            mask_gripper = selection < 0.0
            # Gripper is at index 9 in controls (base=3 + arm_joints=6 = 9)
            controls[mask_gripper, 9] = GRIPPER_CLOSED_POS

        # Apply leg mask
        if self.use_legs and self.leg_selection_index is not None:
            selection = controls[..., self.leg_selection_index]
            mask_fl = selection < -0.5
            mask_fr = selection > 0.5
            mask_neither = ~(mask_fl | mask_fr)

            # Determine where leg commands start
            leg_start_idx = 3  # after base
            if self.use_arm:
                leg_start_idx += 7  # arm commands
                if self.use_gripper:
                    leg_start_idx += 1  # gripper selection

            # Last 6 entries before leg selection are leg commands (FL: 3, FR: 3)
            controls[mask_fl, leg_start_idx + 3 : leg_start_idx + 6] = 0.0  # Zero FR
            controls[mask_fr, leg_start_idx : leg_start_idx + 3] = 0.0  # Zero FL
            controls[mask_neither, leg_start_idx : leg_start_idx + 6] = 0.0  # Zero both

        # Remove selection indices
        controls_full = np.delete(controls, selection_indices, axis=-1)

        if added_dim:
            controls_full = controls_full.squeeze(axis=0)

        return controls_full

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        """Map compact controls to 25-dim policy command for C++ rollout.

        Layout of 25-dim policy command:
        [0:3]   torso_vel_cmd (base velocity)
        [3:10]  arm_cmd (7 arm joints)
        [10:22] leg_cmd (4 legs x 3, used for override)
        [22:25] torso_pos_cmd (roll, pitch, height)

        Args:
            controls: Compact control array of shape (..., nu).

        Returns:
            Policy command array of shape (..., 25).
        """
        controls = np.asarray(controls)
        added_dim = False
        if controls.ndim == 1:
            controls = controls[None]
            added_dim = True

        # Apply selection mask first
        controls = self.apply_selection_mask(controls)

        T = controls.shape[1] if controls.ndim == 3 else 1
        if controls.ndim == 2:
            controls = controls[:, None, :]
            T = 1

        # Initialize from default_policy_command so uncontrolled dimensions
        # keep their defaults (e.g. arm stays at ARM_STOWED_POS when use_arm=False)
        out = np.broadcast_to(self.default_policy_command, (controls.shape[0], controls.shape[1], 25)).copy()

        # Index calculations after selection mask removal
        base_end = 3
        arm_end = base_end + (7 if self.use_arm else 0)
        legs_end = arm_end + (6 if self.use_legs else 0)
        torso_end = legs_end + (3 if self.use_torso else 0)

        # Base velocity
        out[..., 0:3] = controls[..., 0:base_end]

        # Arm commands
        if self.use_arm:
            out[..., 3:10] = controls[..., base_end:arm_end]

        # Leg override commands (front legs only)
        if self.use_legs:
            leg_block = controls[..., arm_end:legs_end]  # shape (..., 6)
            fl_cmd = leg_block[..., 0:3]
            fr_cmd = leg_block[..., 3:6]

            # Place into policy command leg slots [10:22]
            # Groups: FL(10:13), FR(13:16), HL(16:19), HR(19:22)
            out[..., 10:13] = fl_cmd
            out[..., 13:16] = fr_cmd

        # Torso commands (roll, pitch, height)
        if self.use_torso:
            out[..., 22:25] = controls[..., legs_end:torso_end]

        if added_dim:
            out = out.squeeze(axis=0)
        if T == 1 and out.ndim == 3:
            out = out[:, 0, :]

        return out

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Base reward function (returns zeros).

        Override in subclasses for task-specific rewards.

        Args:
            states: Rolled out states, shape (num_rollouts, T, nq+nv).
            sensors: Sensor readings, shape (num_rollouts, T, nsensor).
            controls: Control inputs, shape (num_rollouts, T, nu).
            system_metadata: Optional metadata from the system.

        Returns:
            Rewards for each rollout, shape (num_rollouts,).
        """
        return np.zeros(states.shape[0])

    @property
    def reset_arm_pos(self) -> np.ndarray:
        """Reset position of the arm based on use_arm setting."""
        return ARM_UNSTOWED_POS if self.use_arm else ARM_STOWED_POS

    @property
    def reset_pose(self) -> np.ndarray:
        """Default reset pose for the robot (using RL training defaults)."""
        return np.array(
            [
                0.0,
                0.0,
                STANDING_HEIGHT,  # position
                1,
                0,
                0,
                0,  # quaternion (identity)
                *LEGS_STANDING_POS_RL,  # Use RL default positions
                *self.reset_arm_pos,
            ]
        )

    def optimizer_warm_start(self) -> np.ndarray:
        """Warm start from default command (arm unstowed, legs standing)."""
        return self.default_command.copy()

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check that Spot is still standing. Subclasses should AND with super().success()."""
        body_height = data.qpos[self.body_pose_idx + 2]
        return bool(body_height > self.config.spot_fallen_threshold)

    def reset(self) -> None:
        """Reset the simulation to the default pose."""
        self.data.qpos[:] = self.reset_pose
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_action_components(self) -> list[str]:
        """Get names of each component in the action command vector.

        Matches starfish/dexterity/tasks/spot_base.py.
        """
        action_components = ["spot/base.vx", "spot/base.vy", "spot/base.vtheta"]
        if self.use_arm:
            action_components.extend([f"spot/{joint}" for joint in ARM_JOINT_NAMES])
        if self.use_legs:
            # FR then FL ordering (matches starfish)
            fr_joints = LEG_JOINT_NAMES_BOSDYN[3:6]  # fr_hx, fr_hy, fr_kn
            fl_joints = LEG_JOINT_NAMES_BOSDYN[0:3]  # fl_hx, fl_hy, fl_kn
            action_components.extend([f"spot/{joint}" for joint in fr_joints + fl_joints])
            action_components.append("spot/leg_selection")
        if self.use_torso:
            action_components.extend(["spot/torso.roll", "spot/torso.pitch", "spot/torso.height"])
        return action_components
