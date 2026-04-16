# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MPC setup (sim/controller creation) and HDF5 persistence."""

import copy
from pathlib import Path

import h5py
import numpy as np
import warp as wp
from mujoco import MjModel

from judo.controller import BatchedControllers as JudoBatchedController
from judo.controller import ControllerConfig
from judo.controller.batched_spot_locomotion import BatchedSpotLocomotion
from judo.optimizers import Optimizer
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.policy_mj_simulation import PolicyMJSimulation
from judo.tasks import Task as JudoTask
from judo.utils.mjwarp_rollout_backend import MJWarpRolloutBackend
from run_mpc.mpc_config import PublicMPCConfig, SizeData, make_size_data

MJWARP_MIN_FRICTION = 0.01  # mujoco_warp minimum sliding friction; values below this cause NaN in contact solving


def clamp_for_mjwarp(model: MjModel, min_friction: float = MJWARP_MIN_FRICTION) -> None:
    """Apply model fixups for mujoco_warp compatibility."""
    # Disable fluid dynamics (not supported by mujoco_warp)
    model.opt.density = 0
    # Clamp geom friction to mujoco_warp minimum to avoid NaN in contact solving
    np.maximum(model.geom_friction, min_friction, out=model.geom_friction)


def make_locomotion_controller(use_spot: bool, policy_path: str | None, device: str) -> BatchedSpotLocomotion | None:
    """Creates low-level locomotion controller if Spot task is used."""
    if not use_spot or policy_path is None:
        return None

    return BatchedSpotLocomotion(
        model_path=policy_path,
        device=device,
    )


def setup_mpc(
    config: PublicMPCConfig,
    json_configs: dict,
    task: JudoTask,
    optimizer: Optimizer,
    controller_cfg: ControllerConfig,
    num_parallel: int,
) -> tuple[list[MJSimulation], JudoBatchedController, SizeData]:
    """Create simulations, GPU rollout backend, and batched controllers."""
    use_spot = task.uses_locomotion_policy
    policy_path = str(config.locomotion_policy_path) if use_spot else None
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    sims: list[MJSimulation] = []

    for _ in range(num_parallel):
        if use_spot:
            sim = PolicyMJSimulation(init_task=json_configs["task"])
        else:
            sim = MJSimulation(init_task=json_configs["task"])
        sim.task.config = copy.deepcopy(task.config)
        clamp_for_mjwarp(sim.task.model)
        sims.append(sim)

    # Create shared GPU rollout backend for batched execution
    locomotion_controller = make_locomotion_controller(use_spot, policy_path, device)
    rollout_backend = MJWarpRolloutBackend(
        model=task.model,
        num_threads=optimizer.config.num_rollouts,
        num_problems=num_parallel,
        locomotion_controller=locomotion_controller,
        device=device,
        check_nan=config.check_nan,
    )
    batched_controllers = JudoBatchedController(controller_cfg, task, optimizer, rollout_backend=rollout_backend)
    size_data = make_size_data(sims[0], batched_controllers.controllers[0], config)

    return sims, batched_controllers, size_data


def save_results_to_h5(
    output_path: Path,
    all_results: list[dict[str, np.ndarray | int]],
    size_data: SizeData,
    config: PublicMPCConfig,
    config_path: Path,
    json_configs: dict,
    num_parallel: int,
) -> None:
    """Write all MPC results to an HDF5 file."""
    with h5py.File(str(output_path), "w") as f:
        # Create datasets for all fields we're logging.
        qpos_dataset = f.create_dataset(
            "qpos",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps, size_data.nq),
            chunks=(size_data.chunk_size, size_data.max_num_task_steps, size_data.nq),
            dtype="float64",
        )
        qvel_dataset = f.create_dataset(
            "qvel",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps, size_data.nv),
            chunks=(size_data.chunk_size, size_data.max_num_task_steps, size_data.nv),
            dtype="float64",
        )
        control_dataset = f.create_dataset(
            "control",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps, size_data.nu),
            chunks=(size_data.chunk_size, size_data.max_num_task_steps, size_data.nu),
            dtype="float64",
        )
        sensor_dataset = f.create_dataset(
            "sensor",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps, size_data.nsensordata),
            chunks=(size_data.chunk_size, size_data.max_num_task_steps, size_data.nsensordata),
            dtype="float64",
        )
        reward_dataset = f.create_dataset(
            "reward",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps),
            dtype="float",
        )
        trajectory_length_dataset = f.create_dataset(
            "trajectory_length", shape=(size_data.num_trajectories,), chunks=(size_data.chunk_size), dtype="int"
        )

        if config.store_rollouts:
            rollout_states_dataset = f.create_dataset(
                "rollout_states",
                shape=(
                    size_data.num_trajectories,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                    size_data.num_timesteps,
                    size_data.nq + size_data.nv,
                ),
                chunks=(
                    size_data.chunk_size,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                    size_data.num_timesteps,
                    size_data.nq + size_data.nv,
                ),
                dtype="float64",
            )
            rollout_controls_dataset = f.create_dataset(
                "rollout_controls",
                shape=(
                    size_data.num_trajectories,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                    size_data.num_timesteps,
                    size_data.nu,
                ),
                chunks=(
                    size_data.chunk_size,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                    size_data.num_timesteps,
                    size_data.nu,
                ),
                dtype="float64",
            )
            rollout_rewards_dataset = f.create_dataset(
                "rollout_rewards",
                shape=(
                    size_data.num_trajectories,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                ),
                chunks=(
                    size_data.chunk_size,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                ),
                dtype="float64",
            )

        if config.store_viapoints:
            control_viapoints_dataset = f.create_dataset(
                "control_viapoints",
                shape=(size_data.num_trajectories, size_data.max_num_mpc_steps, size_data.num_nodes, size_data.nu),
                chunks=(size_data.chunk_size, size_data.max_num_mpc_steps, size_data.num_nodes, size_data.nu),
                dtype="float64",
            )

        # Store goal_pos per trajectory if present
        has_goal = "goal_pos" in all_results[0]
        if has_goal:
            goal_dim = len(np.asarray(all_results[0]["goal_pos"]))
            goal_pos_dataset = f.create_dataset(
                "goal_pos",
                shape=(size_data.num_trajectories, goal_dim),
                dtype="float64",
            )

        # Store configuration data.
        f.attrs["config_path"] = str(config_path)
        f.attrs["num_parallel"] = num_parallel
        f.attrs["task"] = json_configs["task"]

        # Write results
        for traj_idx, result in enumerate(all_results):
            trajectory_length_dataset[traj_idx] = result["task_step"]
            qpos_dataset[traj_idx] = result["qpos"]
            qvel_dataset[traj_idx] = result["qvel"]
            control_dataset[traj_idx] = result["control"]
            sensor_dataset[traj_idx] = result["sensor"]
            reward_dataset[traj_idx] = result["reward"]
            if config.store_rollouts:
                rollout_states_dataset[traj_idx] = result["rollout_states"]  # pyright: ignore[reportPossiblyUnboundVariable]
                rollout_controls_dataset[traj_idx] = result["rollout_controls"]  # pyright: ignore[reportPossiblyUnboundVariable]
                rollout_rewards_dataset[traj_idx] = result["rollout_rewards"]  # pyright: ignore[reportPossiblyUnboundVariable]
            if config.store_viapoints:
                control_viapoints_dataset[traj_idx] = result["control_viapoints"]  # pyright: ignore[reportPossiblyUnboundVariable]
            if has_goal:
                goal_pos_dataset[traj_idx] = result["goal_pos"]  # pyright: ignore[reportPossiblyUnboundVariable]
