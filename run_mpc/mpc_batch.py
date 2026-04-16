# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Batched MPC execution loop and trajectory storage."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from judo.app.structs import MujocoState
from judo.controller import BatchedControllers as JudoBatchedController
from judo.controller import Controller as JudoController
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.policy_mj_simulation import PolicyMJSimulation
from judo.visualizers.visualizer import Visualizer
from run_mpc.mpc_config import MPCTimers, PublicMPCConfig, SizeData


def _get_previous_actions(sims: list[MJSimulation]) -> list[np.ndarray | None]:
    """Get previous actions from sims for hierarchical control sync."""
    return [sim.last_policy_output if isinstance(sim, PolicyMJSimulation) else None for sim in sims]


def update_visualization(visualizer: Visualizer, sim_state: MujocoState, traces: np.ndarray) -> None:
    """Update the viser visualization with current sim state and traces."""
    visualizer.data.xpos[:] = sim_state.xpos
    visualizer.data.xquat[:] = sim_state.xquat
    visualizer.viser_model.set_data(visualizer.data)
    sensor_rollout_size = traces.shape[1]
    num_trace_sensors = traces.shape[2]
    all_traces_rollout_size = sensor_rollout_size * num_trace_sensors
    visualizer.viser_model.set_traces(traces, all_traces_rollout_size)


@dataclass
class _BatchStorage:
    """Pre-allocated NaN-padded arrays for recording batch trajectory data."""

    qpos: np.ndarray
    qvel: np.ndarray
    control: np.ndarray
    sensor: np.ndarray
    reward: np.ndarray
    rollout_states: np.ndarray | None = None
    rollout_controls: np.ndarray | None = None
    rollout_rewards: np.ndarray | None = None
    control_timesteps: np.ndarray | None = None
    control_viapoints: np.ndarray | None = None

    @staticmethod
    def allocate(num_parallel: int, size_data: SizeData, config: PublicMPCConfig) -> "_BatchStorage":
        sd = size_data
        storage = _BatchStorage(
            qpos=np.full((num_parallel, sd.max_num_task_steps, sd.nq), np.nan, dtype="float64"),
            qvel=np.full((num_parallel, sd.max_num_task_steps, sd.nv), np.nan, dtype="float64"),
            control=np.full((num_parallel, sd.max_num_task_steps, sd.nu), np.nan, dtype="float64"),
            sensor=np.full((num_parallel, sd.max_num_task_steps, sd.nsensordata), np.nan, dtype="float64"),
            reward=np.full((num_parallel, sd.max_num_task_steps), np.nan, dtype="float64"),
        )
        if config.store_viapoints:
            storage.control_viapoints = np.full(
                (num_parallel, sd.max_num_mpc_steps, sd.num_nodes, sd.nu), np.nan, dtype="float64"
            )
        if config.store_rollouts:
            storage.rollout_states = np.full(
                (num_parallel, sd.max_num_mpc_steps, sd.num_rollouts, sd.num_timesteps, sd.nq + sd.nv),
                np.nan,
                dtype="float64",
            )
            storage.rollout_controls = np.full(
                (num_parallel, sd.max_num_mpc_steps, sd.num_rollouts, sd.num_timesteps, sd.nu),
                np.nan,
                dtype="float64",
            )
            storage.rollout_rewards = np.full(
                (num_parallel, sd.max_num_mpc_steps, sd.num_rollouts),
                -1,
                dtype="int",
            )
            storage.control_timesteps = np.full(
                (num_parallel, sd.max_num_mpc_steps),
                -1,
                dtype="int",
            )
        return storage

    def record_task_step(self, i: int, task_step: int, data: Any, task_control: np.ndarray, reward: float) -> None:
        """Record sim state, control, and reward for one sim at one task step."""
        self.qpos[i, task_step] = data.qpos
        self.qvel[i, task_step] = data.qvel
        self.control[i, task_step] = task_control
        self.sensor[i, task_step] = data.sensordata
        self.reward[i, task_step] = reward

    def record_mpc_step(self, mpc_step: int, task_step: int, controllers: list[JudoController]) -> None:
        """Record rollout / viapoint data for all controllers at one MPC step."""
        for i, ctrl in enumerate(controllers):
            if self.rollout_states is not None:
                self.control_timesteps[i, mpc_step] = task_step  # type: ignore[index]
                self.rollout_states[i, mpc_step] = ctrl.states
                self.rollout_controls[i, mpc_step] = ctrl.rollout_controls  # type: ignore[index]
                self.rollout_rewards[i, mpc_step] = ctrl.rewards  # type: ignore[index]
            if self.control_viapoints is not None:
                self.control_viapoints[i, mpc_step] = ctrl.spline.y

    def package_results(self, max_num_task_steps: int) -> list[dict[str, np.ndarray | int]]:
        """Package per-trajectory result dicts from stored arrays."""
        num_parallel = self.qpos.shape[0]
        results: list[dict[str, np.ndarray | int]] = []
        for i in range(num_parallel):
            result: dict[str, np.ndarray | int] = {
                "task_step": max_num_task_steps,
                "qpos": self.qpos[i],
                "qvel": self.qvel[i],
                "control": self.control[i],
                "sensor": self.sensor[i],
                "reward": self.reward[i],
            }
            if self.control_timesteps is not None:
                result["control_timesteps"] = self.control_timesteps[i]
                result["rollout_states"] = self.rollout_states[i]  # type: ignore[index]
                result["rollout_controls"] = self.rollout_controls[i]  # type: ignore[index]
                result["rollout_rewards"] = self.rollout_rewards[i]  # type: ignore[index]
            if self.control_viapoints is not None:
                result["control_viapoints"] = self.control_viapoints[i]
            results.append(result)
        return results


def run_mpc_batch(
    sims: list[MJSimulation],
    batched_controllers: JudoBatchedController,
    config: PublicMPCConfig,
    size_data: SizeData,
    timers: MPCTimers,
    vis: Visualizer | None = None,
) -> list[dict[str, np.ndarray | int]]:
    """Runs MPC for one or more problems using batched controllers.

    Typically for Spot tasks:
        physics_step = model.opt.timestep = 0.01 s
        task_step = task.dt = physics_step * num_physics_substeps = 0.01 * 2 = 0.02 s
        mpc_step = 1 / control_freq = 1 / 10 = 0.1 s
        tasks_steps_per_mpc_step = mpc_step / task_step  = 0.1 / 0.02 = 5

    We record at the task step rate (not physics step rate) since downstream
    BC and RL applications take decisions at the task step rate.
    """
    num_parallel = len(sims)
    controllers = batched_controllers.controllers

    # Reset all simulations and controllers, sample new initial conditions + goals
    for sim, ctrl in zip(sims, controllers, strict=True):
        sim.task.reset()
        if isinstance(sim, PolicyMJSimulation):
            sim.reset_policy_state()
        ctrl.reset()
        ctrl.update_states(sim.sim_state)

    # First optimization pass
    batched_controllers.set_init_previous_actions(_get_previous_actions(sims))
    timers.optimization.tic()
    batched_controllers.update_action()
    timers.optimization.toc()

    storage = _BatchStorage.allocate(num_parallel, size_data, config)

    mpc_step = 0
    for task_step in tqdm(range(config.max_num_task_steps), desc=f"Task steps ({num_parallel} parallel)", leave=False):
        # Compute control and record state for all simulations
        for i, (sim, ctrl) in enumerate(zip(sims, controllers, strict=True)):
            task_control = ctrl.compute(sim.task.data)

            timers.reward_compute.tic()
            reward = sim.task.reward(
                states=np.concatenate([sim.task.data.qpos, sim.task.data.qvel], axis=0)[None, None],
                sensors=sim.task.data.sensordata[None, None],
                controls=task_control[None, None],
            )[0]
            timers.reward_compute.toc()

            storage.record_task_step(i, task_step, sim.task.data, task_control, reward)

        # Re-plan at MPC frequency
        if task_step % size_data.sim_steps_per_mpc_step == 0:
            batched_controllers.update_states([sim.sim_state for sim in sims])
            batched_controllers.set_init_previous_actions(_get_previous_actions(sims))

            timers.optimization.tic()
            batched_controllers.update_action()
            timers.optimization.toc()

            storage.record_mpc_step(mpc_step, task_step, controllers)
            mpc_step += 1

        # Advance physics
        timers.sim_step.tic()
        for i, sim in enumerate(sims):
            sim.step(storage.control[i, task_step])
        timers.sim_step.toc()

        if vis is not None and num_parallel == 1:
            update_visualization(vis, sims[0].sim_state, controllers[0].traces)
            time.sleep(sims[0].timestep)

    results = storage.package_results(config.max_num_task_steps)
    for i, sim in enumerate(sims):
        results[i]["success"] = sim.task.success(sim.task.model, sim.task.data)
        if hasattr(sim.task.config, "goal_pos"):
            results[i]["goal_pos"] = sim.task.config.goal_pos.copy()
    return results
