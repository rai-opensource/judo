# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MPC configuration dataclasses and config-loading utilities."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dacite
import numpy as np

from judo.controller import Controller as JudoController
from judo.controller import ControllerConfig
from judo.optimizers import (
    Optimizer,
    OptimizerConfig,
    get_registered_optimizers,
)
from judo.simulation.mj_simulation import MJSimulation
from judo.tasks import (
    Task as JudoTask,
)
from judo.tasks import (
    TaskConfig as JudoTaskConfig,
)
from judo.tasks import (
    get_registered_tasks as get_registered_judo_tasks,
)
from judo.tasks.spot.spot_constants import SPOT_LOCOMOTION_POLICY_PATH
from judo.utils.timer import Timer


def decode_config(obj: dict) -> dict:
    """Custom JSON decoder to handle NumPy arrays and Path strings."""
    for key, value in obj.items():
        if isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            obj[key] = np.array(value)
        if isinstance(value, str) and ("/" in value or "\\" in value):
            obj[key] = Path(value)
    return obj


@dataclass
class PublicMPCConfig:
    """Config class for MPC runs."""

    config_path: Path  # Path to .json containing task/controller config for MPC run.
    dataset_output_path: Path | None = None  # Path to .h5 dataset of trajectories.
    num_trajectories: int = 10  # Number of MPC runs to perform.
    num_parallel: int = 1  # Number of trajectories to run in parallel (batched MPC).
    max_num_task_steps: int = 750  # Maximum number of task steps to run per MPC run.
    chunk_size: int = 1  # Size of the memory chunk for the h5 dataset (defaults to 1 -> 1 trajectory per chunk).
    control_freq: float | None = None  # MPC control frequency in Hz. If None, use the controller config's frequency.
    store_rollouts: bool = True  # Store open loop control inputs and state trajectories.
    store_viapoints: bool = True  # Store open loop control of the best spline.
    require_success: bool = True  # Only keep trajectories where task.success() returns True.
    visualize: bool = False
    locomotion_policy_path: Path = SPOT_LOCOMOTION_POLICY_PATH
    check_nan: bool = False  # Check for NaN in mujoco_warp rollout outputs (for debugging).


@dataclass
class SizeData:
    """Sizes needed to calculate things in the dataset."""

    max_num_task_steps: int
    nq: int
    nv: int
    nu: int
    max_num_mpc_steps: int
    nsensordata: int
    num_nodes: int
    num_rollouts: int
    num_trajectories: int
    chunk_size: int
    num_timesteps: int
    sim_steps_per_mpc_step: int


def make_size_data(sim: MJSimulation, controller: JudoController, config: PublicMPCConfig) -> SizeData:
    """Calculates the sizes for everything."""
    nu = sim.task.nu
    if config.control_freq is None:
        control_freq = controller.controller_cfg.control_freq
    else:
        control_freq = config.control_freq
    sim_steps_per_mpc_step = int(1.0 / control_freq / sim.task.dt)
    max_num_mpc_steps = np.ceil(config.max_num_task_steps / sim_steps_per_mpc_step).astype(int)
    nq = sim.task.model.nq
    nv = sim.task.model.nv
    nsensordata = sim.task.model.nsensordata
    num_nodes = controller.optimizer_cfg.num_nodes
    num_rollouts = controller.optimizer_cfg.num_rollouts
    return SizeData(
        max_num_mpc_steps=max_num_mpc_steps,
        nq=nq,
        nv=nv,
        nsensordata=nsensordata,
        num_nodes=num_nodes,
        num_rollouts=num_rollouts,
        num_trajectories=config.num_trajectories,
        chunk_size=config.chunk_size,
        num_timesteps=controller.num_timesteps,
        sim_steps_per_mpc_step=sim_steps_per_mpc_step,
        max_num_task_steps=config.max_num_task_steps,
        nu=nu,
    )


def load_configs_from_json_data(json_data: Any) -> tuple[JudoTask, Optimizer, ControllerConfig]:
    """Loads the configs from a json file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

    available_tasks = get_registered_judo_tasks()
    available_optimizers = get_registered_optimizers()
    task_entry = available_tasks.get(json_data["task"])
    assert task_entry is not None, f"Task {json_data['task']} is not registered!"
    optimizer_entry = available_optimizers.get(json_data["optimizer"])
    assert optimizer_entry is not None, f"Optimizer {json_data['optimizer']} is not registered!"

    task_cls, task_config_cls = task_entry
    task_config: JudoTaskConfig = dacite.from_dict(task_config_cls, json_data["task_config"])
    task: JudoTask = task_cls()
    task.config = task_config

    optimizer_cls, optimizer_config_cls = optimizer_entry
    optimizer_config: OptimizerConfig = dacite.from_dict(optimizer_config_cls, json_data["optimizer_config"])
    optimizer: Optimizer = optimizer_cls(optimizer_config, task.nu)

    controller_cfg: ControllerConfig = dacite.from_dict(ControllerConfig, json_data["controller_config"])

    return (
        task,
        optimizer,
        controller_cfg,
    )


@dataclass
class MPCTimers:
    """Timers for measuring MPC performance."""

    optimization: Timer
    sim_step: Timer
    reward_compute: Timer
    h5_write: Timer

    @classmethod
    def create(cls) -> "MPCTimers":
        """Create a new set of MPC timers."""
        return cls(
            optimization=Timer("Opt", unit="ms"),
            sim_step=Timer("Sim", unit="ms"),
            reward_compute=Timer("Rew", unit="ms"),
            h5_write=Timer("h5p", unit="ms"),
        )

    def print_all(self) -> None:
        """Print timing statistics for all MPC operations."""
        logging.info("=== Performance Statistics ===")
        self.optimization.print_stats(logging.info)
        self.sim_step.print_stats(logging.info)
        self.reward_compute.print_stats(logging.info)
        self.h5_write.print_stats(logging.info)
