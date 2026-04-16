# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""GPU-batched locomotion controller for Spot hierarchical control."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import warp as wp

from judo.tasks.spot.spot_constants import (
    ISAAC_TO_MUJOCO_INDICES_12,
    LOCOMOTION_DEFAULT_JOINTS_OFFSET,
    LOCOMOTION_DEFAULT_LEGS_OFFSET,
    MUJOCO_TO_ISAAC_INDICES_19,
    POLICY_OUTPUT_DIM,
)


@wp.kernel
def build_observation_kernel(
    qpos: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    qvel: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    cmd: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    prev_actions: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    joints_offset: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    mj_to_isaac: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    obs: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Build 84-dim observation from state, command, and previous actions.

    Per-thread (one thread per batch element):
      obs[ 0: 3] = quat_rotate_inverse(quat, lin_vel)
      obs[ 3: 6] = angular velocity
      obs[ 6: 9] = quat_rotate_inverse(quat, [0, 0, -1])
      obs[ 9:34] = cmd (25 dims)
      obs[34:53] = mujoco->isaac reordered joints - offset (19 dims)
      obs[53:72] = mujoco->isaac reordered joint velocities (19 dims)
      obs[72:84] = previous actions (12 dims)
    """
    i = wp.tid()

    # Quaternion (scalar-first: w, x, y, z)
    qw = qpos[i, 3]
    qx = qpos[i, 4]
    qy = qpos[i, 5]
    qz = qpos[i, 6]

    ww = qw * qw
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz

    # quat_rotate_inverse(quat, lin_vel) -> obs[0:3]
    vx = qvel[i, 0]
    vy = qvel[i, 1]
    vz = qvel[i, 2]
    obs[i, 0] = (ww + xx - yy - zz) * vx + 2.0 * (xy + wz) * vy + 2.0 * (xz - wy) * vz
    obs[i, 1] = 2.0 * (xy - wz) * vx + (ww - xx + yy - zz) * vy + 2.0 * (yz + wx) * vz
    obs[i, 2] = 2.0 * (xz + wy) * vx + 2.0 * (yz - wx) * vy + (ww - xx - yy + zz) * vz

    # Angular velocity -> obs[3:6]
    obs[i, 3] = qvel[i, 3]
    obs[i, 4] = qvel[i, 4]
    obs[i, 5] = qvel[i, 5]

    # quat_rotate_inverse(quat, [0, 0, -1]) -> obs[6:9]
    obs[i, 6] = -2.0 * (xz - wy)
    obs[i, 7] = -2.0 * (yz + wx)
    obs[i, 8] = -(ww - xx - yy + zz)

    # Command (25 dims) -> obs[9:34]
    for j in range(25):
        obs[i, 9 + j] = cmd[i, j]

    # Mujoco->Isaac reorder + offset subtraction for joints -> obs[34:53]
    for j in range(19):
        idx = mj_to_isaac[j]
        obs[i, 34 + j] = qpos[i, 7 + idx] - joints_offset[j]

    # Mujoco->Isaac reorder for joint velocities -> obs[53:72]
    for j in range(19):
        idx = mj_to_isaac[j]
        obs[i, 53 + j] = qvel[i, 6 + idx]

    # Previous actions -> obs[72:84]
    for j in range(12):
        obs[i, 72 + j] = prev_actions[i, j]


@wp.kernel
def compute_targets_kernel(
    actions: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    cmd: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    legs_offset: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    isaac_to_mj: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    action_scale: float,
    target_q: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Post-process policy actions into 19-dim target_q.

    Per-thread:
      1. Scale actions and apply offset, then isaac->mujoco reorder for legs
      2. Per-leg override: if any of 3 cmd joints != 0, use cmd instead
      3. Arm passthrough: target_q[12:19] = cmd[3:10]
    """
    i = wp.tid()

    # Scale + offset + isaac->mujoco reorder for legs
    # _isaac_to_mujoco_batch: output[k] = input[ISAAC_TO_MUJOCO_INDICES_12[k]]
    for k in range(12):
        j = isaac_to_mj[k]
        target_q[i, k] = actions[i, j] * action_scale + legs_offset[j]

    # Per-leg override: if any of 3 cmd joints are nonzero, use cmd
    # leg_joint_command = cmd[:, 10:22]
    for leg in range(4):
        any_nonzero = int(0)
        for dof in range(3):
            if cmd[i, 10 + leg * 3 + dof] != 0.0:
                any_nonzero = 1
        if any_nonzero == 1:
            for dof in range(3):
                target_q[i, leg * 3 + dof] = cmd[i, 10 + leg * 3 + dof]

    # Arm passthrough: target_q[12:19] = cmd[3:10]
    for j in range(7):
        target_q[i, 12 + j] = cmd[i, 3 + j]


class BatchedSpotLocomotion:
    """Batched GPU-accelerated locomotion controller for Spot.

    This controller implements the learned locomotion policy that converts
    high-level commands to target joint positions.

    The controller:
    - Takes high-level commands (base velocity, arm/leg positions) from MPC
    - Runs learned locomotion policy at ~56Hz
    - Produces target joint positions for PD control
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        action_scale: float = 0.2,
    ) -> None:
        """Initialize the batched locomotion controller.

        Args:
            model_path: Path to the pre-trained locomotion policy (.onnx file).
            device: The warp device string (e.g. "cuda:0").
            action_scale: Scaling factor for the action. Defaults to 0.2.
        """
        self._device = device
        self.action_scale = action_scale

        # Parse device string to get CUDA device id
        if ":" in device:
            self._device_id = int(device.split(":")[1])
        else:
            self._device_id = 0

        # Load ONNX model and make batch dimension dynamic (on-disk file
        # keeps batch=1 for the C++ rollout code; Python needs dynamic batch).
        onnx_model = onnx.load(str(model_path))
        for tensor in list(onnx_model.graph.input) + list(onnx_model.graph.output):
            tensor.type.tensor_type.shape.dim[0].ClearField("dim_value")
            tensor.type.tensor_type.shape.dim[0].dim_param = "batch"

        self._ort_session = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=[("CUDAExecutionProvider", {"device_id": self._device_id})],
        )
        self._ort_input_name = self._ort_session.get_inputs()[0].name
        self._ort_output_name = self._ort_session.get_outputs()[0].name

        # Constant arrays on GPU
        self._joints_offset = wp.array(
            np.array(LOCOMOTION_DEFAULT_JOINTS_OFFSET, dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )
        self._legs_offset = wp.array(
            np.array(LOCOMOTION_DEFAULT_LEGS_OFFSET, dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )
        self._mj_to_isaac = wp.array(
            np.array(MUJOCO_TO_ISAAC_INDICES_19, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        self._isaac_to_mj = wp.array(
            np.array(ISAAC_TO_MUJOCO_INDICES_12, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

    def _run_policy(self, obs: wp.array, actions: wp.array) -> None:
        """Run the locomotion policy via ONNX Runtime with GPU I/O binding.

        Writes ORT output directly into the pre-allocated ``actions`` warp array.

        Args:
            obs: Observation array on GPU (batch_size, 84).
            actions: Pre-allocated output array on GPU (batch_size, POLICY_OUTPUT_DIM).
        """
        io_binding = self._ort_session.io_binding()
        io_binding.bind_input(
            name=self._ort_input_name,
            device_type="cuda",
            device_id=self._device_id,
            element_type=np.float32,
            shape=tuple(obs.shape),
            buffer_ptr=obs.ptr,
        )
        io_binding.bind_output(
            name=self._ort_output_name,
            device_type="cuda",
            device_id=self._device_id,
            element_type=np.float32,
            shape=tuple(actions.shape),
            buffer_ptr=actions.ptr,
        )
        self._ort_session.run_with_iobinding(io_binding)

    def compute_batch(
        self,
        cmd: wp.array,
        qpos: wp.array,
        qvel: wp.array,
        previous_actions: wp.array | None,
    ) -> tuple[wp.array, wp.array]:
        """Get target joint positions from high-level command (batched interface).

        Args:
            cmd: High-level command (batch_size, 25).
            qpos: Joint positions (batch_size, nq).
            qvel: Joint velocities (batch_size, nv).
            previous_actions: Previous actions (batch_size, POLICY_OUTPUT_DIM) or None.

        Returns:
            Tuple of (target_q, new_previous_actions) as warp arrays.
        """
        batch_size = cmd.shape[0]

        if previous_actions is None:
            previous_actions = wp.zeros((batch_size, POLICY_OUTPUT_DIM), dtype=wp.float32, device=self._device)

        obs = wp.zeros((batch_size, 84), dtype=wp.float32, device=self._device)
        wp.launch(
            build_observation_kernel,
            dim=batch_size,
            inputs=[qpos, qvel, cmd, previous_actions, self._joints_offset, self._mj_to_isaac],
            outputs=[obs],
            device=self._device,
        )

        # Ensure observation kernel completes before ORT reads the buffer
        wp.synchronize()

        actions = wp.zeros((batch_size, POLICY_OUTPUT_DIM), dtype=wp.float32, device=self._device)
        self._run_policy(obs, actions)

        target_q = wp.zeros((batch_size, 19), dtype=wp.float32, device=self._device)
        wp.launch(
            compute_targets_kernel,
            dim=batch_size,
            inputs=[actions, cmd, self._legs_offset, self._isaac_to_mj, self.action_scale],
            outputs=[target_q],
            device=self._device,
        )

        return target_q, actions

    def reset(self) -> None:
        """Reset internal state (no-op since previous_actions is now external)."""

    @property
    def target_frequency(self) -> float:
        """Returns the target frequency of the low-level controller."""
        return 50.0  # Hz
