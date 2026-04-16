# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Constants for Spot tasks.

This file contains joint limits, default positions, and command indices
used by Spot locomotion and manipulation tasks.
Values are synchronized with starfish/dexterity/spot_interface/spot/constants.py.
"""

import numpy as np

from judo import MODEL_PATH

# Locomotion policy path (Xinghao's v1 policy from starfish)
SPOT_LOCOMOTION_POLICY_PATH = MODEL_PATH / "policies" / "spot_locomotion.onnx"

# Default rollout cutoff time (125ms for 8Hz MPC)
DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME: float = 0.125

# Number of legs
N_LEGS = 4
N_LEG_JOINTS = 3
POLICY_OUTPUT_DIM = N_LEGS * N_LEG_JOINTS  # 12 leg actuator commands

### Joint Names
LEG_JOINT_NAMES_BOSDYN = [
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
]

ARM_JOINT_NAMES = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]

### Initial Configurations
GRIPPER_CLOSED_POS = 0.0
GRIPPER_OPEN_POS = -1.54

LEGS_STANDING_POS = np.array(
    [
        0.12,
        0.72,
        -1.45,  # FL
        -0.12,
        0.72,
        -1.45,  # FR
        0.12,
        0.72,
        -1.45,  # HL
        -0.12,
        0.72,
        -1.45,  # HR
    ]
)

# RL training default positions (used for policy normalization)
LEGS_STANDING_POS_RL = np.array(
    [
        0.12,
        0.5,
        -1.0,  # FL
        -0.12,
        0.5,
        -1.0,  # FR
        0.12,
        0.5,
        -1.0,  # HL
        -0.12,
        0.5,
        -1.0,  # HR
    ]
)

ARM_STOWED_POS = np.array([0, -3.11, 3.13, 1.56, 0, -1.56, GRIPPER_CLOSED_POS])

ARM_UNSTOWED_POS = np.array([0, -0.9, 1.8, 0, -0.9, 0, GRIPPER_CLOSED_POS])

### Heights
STANDING_HEIGHT = 0.52
STANDING_HEIGHT_CMD = STANDING_HEIGHT  # Match starfish

### Soft Joint Limits (for optimization)
LEG_SOFT_LOWER_JOINT_LIMITS = np.array([-0.6, -0.8, -2.7] * N_LEGS)
LEG_SOFT_UPPER_JOINT_LIMITS = np.array([0.6, 1.65, -0.3] * N_LEGS)
ARM_SOFT_LOWER_JOINT_LIMITS = ARM_UNSTOWED_POS - np.array([1.0, 1.0, 0.8, np.pi / 2, 0.7, np.pi / 4, 0])
ARM_SOFT_UPPER_JOINT_LIMITS = ARM_UNSTOWED_POS + np.array([1.0, 0.8, 0.6, np.pi / 2, 0.9, np.pi / 4, 0])

### Locomotion Policy Constants
RL_LOCOMOTION_COMMAND_LENGTH = 25

# Isaac<->MuJoCo joint index mappings
ISAAC_TO_MUJOCO_INDICES_12 = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
ISAAC_TO_MUJOCO_INDICES_19 = [1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18]
MUJOCO_TO_ISAAC_INDICES_12 = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
MUJOCO_TO_ISAAC_INDICES_19 = [12, 0, 3, 6, 9, 13, 1, 4, 7, 10, 14, 2, 5, 8, 11, 15, 16, 17, 18]

# Default joint offsets for locomotion policy (Isaac order, 19 joints)
LOCOMOTION_DEFAULT_JOINTS_OFFSET = (
    0.0,
    0.12,
    -0.12,
    0.12,
    -0.12,
    -0.9,
    0.5,
    0.5,
    0.5,
    0.5,
    1.8,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    0.0,
    -0.9,
    0.0,
    -1.54,
)

# Default leg offsets for locomotion policy (Isaac order, 12 legs)
LOCOMOTION_DEFAULT_LEGS_OFFSET = (0.12, -0.12, 0.12, -0.12, 0.5, 0.5, 0.5, 0.5, -1.0, -1.0, -1.0, -1.0)

### Command Indices
# Command structure: [base_vel(3), arm(7), legs(12), torso(3)] = 25 dimensions
BASE_VEL_CMD_INDS = [0, 1, 2]
ARM_CMD_INDS = [3, 4, 5, 6, 7, 8, 9]
FRONT_LEG_CMD_INDS = [10, 11, 12, 13, 14, 15]  # FL then FR
TORSO_CMD_INDS = [22, 23, 24]  # roll, pitch, height

### Control Limits
BASE_SOFT_LIMITS = 0.7 * np.ones(3)

# Torso control limits (roll, pitch, height) - from starfish
TORSO_LOWER = np.array([-0.0, -1.0, 0.3])
TORSO_UPPER = np.array([+0.0, +1.0, 1.0])

### Object constants
# Synced from starfish/dexterity/spot_interface/spot/constants.py
Z_AXIS = np.array([0.0, 0.0, 1.0])

# Tire
TIRE_RADIUS = 0.33
TIRE_HALF_WIDTH = 0.17

# Box
BOX_HALF_LENGTH = 0.254
