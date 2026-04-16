# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import copy
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from mujoco import MjData
from omegaconf import DictConfig
from scipy.interpolate import interp1d

from judo.app.structs import MujocoState, SplineData
from judo.app.utils import register_optimizers_from_cfg, register_tasks_from_cfg
from judo.config import OverridableConfig
from judo.gui import slider
from judo.optimizers import Optimizer, OptimizerConfig, get_registered_optimizers
from judo.tasks import Task, TaskConfig, get_registered_tasks
from judo.utils.mj_rollout_backend import MJRolloutBackend
from judo.utils.normalization import (
    IdentityNormalizer,
    Normalizer,
    NormalizerType,
    make_normalizer,
    normalizer_registry,
)
from judo.utils.policy_mj_rollout_backend import PolicyMJRolloutBackend

if TYPE_CHECKING:
    from judo.utils.mjwarp_rollout_backend import MJWarpRolloutBackend
from judo.utils.rollout_backend import RolloutBackend
from judo.utils.timer import Timer
from judo.visualizers.utils import get_trace_sensors


@slider("horizon", 0.1, 10.0, bounded=True)
@slider("control_freq", 0.25, 50.0)
@dataclass
class ControllerConfig(OverridableConfig):
    """Base controller config."""

    horizon: float = 1.0
    spline_order: Literal["zero", "linear", "cubic"] = "linear"
    control_freq: float = 20.0
    max_opt_iters: int = 1
    max_num_traces: int = 5
    action_normalizer: Literal["none", "min_max", "running"] = "none"


class Controller:
    """The controller object."""

    def __init__(
        self,
        controller_config: ControllerConfig,
        task: Task,
        optimizer: Optimizer,
        rollout_backend: "Literal['mujoco'] | MJWarpRolloutBackend" = "mujoco",
    ) -> None:
        """Initialize the controller.

        Args:
            controller_config: The controller configuration.
            task: The task to use.
            optimizer: The optimizer to use.
            rollout_backend: The backend to use for rollouts. Either "mujoco" to create
                a new CPU backend, or an existing RolloutBackend instance (e.g. GPU warp).
        """
        self._controller_cfg = controller_config
        self.task = task
        self.optimizer = optimizer

        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()

        self.model = self.task.model

        # Initialize rollout backend (auto-select policy backend if task requires it)
        if not isinstance(rollout_backend, str):
            # MJWarpRolloutBackend instance (avoid isinstance check to skip mujoco_warp import)
            self.rollout_backend = rollout_backend
        elif self.task.uses_locomotion_policy:
            assert self.task.locomotion_policy_path is not None
            self.rollout_backend: RolloutBackend = PolicyMJRolloutBackend(
                model=self.model,
                num_threads=self.optimizer_cfg.num_rollouts,
                policy_path=self.task.locomotion_policy_path,
                physics_substeps=self.task.physics_substeps,
            )
        else:
            self.rollout_backend = MJRolloutBackend(
                model=self.model,
                num_threads=self.optimizer_cfg.num_rollouts,
            )

        self._last_policy_output = None
        self.action_normalizer = self._init_action_normalizer()

        # a container for any metadata from the system that we want to pass to the task
        self.system_metadata = {}

        self.states = np.zeros((self.optimizer_cfg.num_rollouts, self.num_timesteps, self.model.nq + self.model.nv))
        self.current_state = np.concatenate([self.task.data.qpos, self.task.data.qvel])
        self.sensors = np.zeros((self.optimizer_cfg.num_rollouts, self.num_timesteps, self.model.nsensordata))
        # Use task.nu for control dimensions (may differ from model.nu for Spot tasks)
        self.rollout_controls = np.zeros((self.optimizer_cfg.num_rollouts, self.num_timesteps, self.task.nu))
        self.rewards = np.zeros((self.optimizer_cfg.num_rollouts,))
        self.reset()

        self.traces = None
        self.trace_sensors = get_trace_sensors(self.model)
        self.num_trace_elites = min(self.max_num_traces, len(self.rewards))
        self.num_trace_sensors = len(self.trace_sensors)
        self.sensor_rollout_size = self.num_timesteps - 1
        self.all_traces_rollout_size = self.sensor_rollout_size * self.num_trace_sensors

    @property
    def horizon(self) -> float:
        """Helper function to recalculate the horizon for simulation."""
        return self.controller_cfg.horizon

    @property
    def nu(self) -> int:
        """Helper function to get the number of control inputs."""
        return self.task.nu

    @property
    def max_num_traces(self) -> int:
        """Helper function to recalculate the max number of traces for simulation."""
        return self.controller_cfg.max_num_traces

    @property
    def max_opt_iters(self) -> int:
        """Helper function to recalculate the max number of optimization iterations for simulation."""
        return self.controller_cfg.max_opt_iters

    @property
    def spline_order(self) -> str:
        """Helper function to recalculate the spline order for simulation."""
        return self.controller_cfg.spline_order

    @property
    def spline_data(self) -> SplineData:
        """Helper function to get the spline data."""
        return SplineData(self.times, self.nominal_knots)

    @property
    def action_normalizer_type(self) -> NormalizerType:
        """Helper function to get the type of action normalizer."""
        return self.controller_cfg.action_normalizer

    @property
    def num_timesteps(self) -> int:
        """Helper function to recalculate the number of timesteps for simulation."""
        return np.ceil(self.horizon / self.task.dt).astype(int)

    @property
    def rollout_times(self) -> np.ndarray:
        """Helper function to calculate the rollout times based on the horizon length."""
        return self.task.dt * np.arange(self.num_timesteps)

    @property
    def spline_timesteps(self) -> np.ndarray:
        """Helper function to create new timesteps for spline queries."""
        return np.linspace(0, self.horizon, self.optimizer_cfg.num_nodes, endpoint=True)

    @property
    def optimizer_cfg(self) -> OptimizerConfig:
        """Helper function to get the optimizer config."""
        return self.optimizer.config

    @optimizer_cfg.setter
    def optimizer_cfg(self, optimizer_cfg: OptimizerConfig) -> None:
        """Helper function to set the optimizer config."""
        self.optimizer.config = optimizer_cfg

    @property
    def optimizer_cls(self) -> type:
        """Returns the optimizer class."""
        return self.optimizer.__class__

    @property
    def optimizer_config_cls(self) -> type:
        """Returns the optimizer config class."""
        return self.optimizer.config.__class__

    @property
    def task_config(self) -> TaskConfig:
        """Returns the task config, which is uniquely defined by the task."""
        return self.task.config

    @task_config.setter
    def task_config(self, task_cfg: TaskConfig) -> None:
        """Sets the task config."""
        self.task.config = task_cfg

    @property
    def time(self) -> float:
        """Returns the current simulation time."""
        return self.task.time

    @time.setter
    def time(self, value: float) -> None:
        """Sets the current simulation time."""
        self.task.time = value

    @property
    def controller_cfg(self) -> ControllerConfig:
        """Returns the controller config."""
        return self._controller_cfg

    @controller_cfg.setter
    def controller_cfg(self, controller_cfg: ControllerConfig) -> None:
        """Sets the controller config."""
        self._controller_cfg = controller_cfg
        self.action_normalizer = self._init_action_normalizer()

    def update_action(self) -> None:
        """Update controller actions from current state/time.

        This method runs the full optimization loop for a single controller.
        For batched multi-controller optimization, use BatchedControllers instead.
        """
        self._pre_optimization()

        # run optimization loop
        i = 0
        while i < self.max_opt_iters and not self.optimizer.stop_cond():
            self.rollout_controls = self._sample_controls()
            self._pre_rollout()

            # Convert task-space controls to sim-space for rollout backend
            sim_controls = self.task.task_to_sim_ctrl(self.rollout_controls)

            # Roll out dynamics
            self.states, self.sensors, policy_output = self.rollout_backend.rollout(
                self.current_state,
                sim_controls,
                self._last_policy_output,
            )
            if policy_output is not None:
                self._last_policy_output = policy_output

            self._post_rollout()
            self._update_iteration()
            i += 1

        self._post_optimization()

    def _pre_optimization(self) -> None:
        """Setup phase before optimization loop. Called once per update_action."""
        assert self.current_state.shape == (self.model.nq + self.model.nv,), "Current state must be of shape (nq + nv,)"
        assert self.optimizer_cfg.num_rollouts > 0, "Need at least one rollout!"

        if self.optimizer_cfg.num_nodes < 4 and self.spline_order == "cubic":
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            self.optimizer_cfg.num_nodes = 4

        # Adjust time + move policy forward.
        self._new_times = self.time + self.spline_timesteps
        nominal_knots = self.spline(self._new_times)
        self._nominal_knots_normalized = self.action_normalizer.normalize(nominal_knots)

        # Resizing rollout backend due to changes in num_rollouts (e.g., from GUI)
        if self.rollout_backend.num_threads != self.optimizer_cfg.num_rollouts:
            # Preserve num_problems when updating shared backend (warp), or just num_threads (CPU)
            num_problems = getattr(self.rollout_backend, "num_problems", None)
            if num_problems is not None:
                self.rollout_backend.update(self.optimizer_cfg.num_rollouts, num_problems)  # type: ignore[call-arg]
            else:
                self.rollout_backend.update(self.optimizer_cfg.num_rollouts)
            if self._last_policy_output is not None:
                self._last_policy_output = None

        normalizer_cls = normalizer_registry.get(self.action_normalizer_type)
        if normalizer_cls is None:
            warnings.warn(
                f"Invalid action normalizer type '{self.action_normalizer_type}'. "
                f"Available types: {list(normalizer_registry.keys())}. "
                "Falling back to 'none' normalizer.",
                stacklevel=2,
            )
            normalizer_cls = IdentityNormalizer

        # force the normalizer to be re-initialized when the type changes in GUI
        # TODO(yunhai): check for changes in the normalizer config and update when appropriate
        if not isinstance(self.action_normalizer, normalizer_cls):
            self.action_normalizer = self._init_action_normalizer()

        # call entrypoint prior to optimization
        self.optimizer.pre_optimization(self.times, self._new_times)

    def _sample_controls(self) -> np.ndarray:
        """Sample control knots and prepare rollout controls for this iteration."""
        # sample controls and clamp to action bounds
        self._candidate_knots_normalized = self.optimizer.sample_control_knots(self._nominal_knots_normalized)
        self._candidate_knots_normalized = np.clip(
            self._candidate_knots_normalized,
            self.action_normalizer.normalize(self.task.actuator_ctrlrange[:, 0]),
            self.action_normalizer.normalize(self.task.actuator_ctrlrange[:, 1]),
        )
        self.candidate_knots = self.action_normalizer.denormalize(self._candidate_knots_normalized)

        # Evaluate rollout controls at sim timesteps.
        candidate_splines = make_spline(self._new_times, self.candidate_knots, self.spline_order)
        rollout_controls = candidate_splines(self.time + self.rollout_times)
        return rollout_controls

    def _pre_rollout(self) -> None:
        """Pre-rollout phase: prepare task for rollout."""
        self.task.pre_rollout(self.current_state)

    def _post_rollout(self) -> None:
        """Post-rollout phase: process rollout results and compute rewards."""
        self.task.post_rollout(
            self.states,
            self.sensors,
            self.rollout_controls,
            self.system_metadata,
        )
        self.rewards = self.task.reward(
            self.states,
            self.sensors,
            self.rollout_controls,
            self.system_metadata,
        )

    def _update_iteration(self) -> None:
        """Update phase: update nominal knots and normalizer for next iteration."""
        # Update nominal knots for next optimization iteration
        self._nominal_knots_normalized = self.optimizer.update_nominal_knots(
            self._candidate_knots_normalized, self.rewards
        )
        # Update action normalizer
        self.action_normalizer.update(self.candidate_knots)

    def _post_optimization(self) -> None:
        """Finalization phase after optimization loop. Called once per update_action."""
        # Update nominal controls and spline.
        self.nominal_knots = self.action_normalizer.denormalize(self._nominal_knots_normalized)
        self.times = self._new_times
        self.update_spline(self.times, self.nominal_knots)
        self.update_traces()

    def action(self, time: float) -> np.ndarray:
        """Current best action of policy."""
        return self.spline(time)

    def compute(self, data: MjData) -> np.ndarray:
        """Compute control action for given MjData state.

        Args:
            data: MuJoCo data object with current state.

        Returns:
            Control action at the current time.
        """
        return self.action(data.time)

    def update_spline(self, times: np.ndarray, controls: np.ndarray) -> None:
        """Update the spline with new timesteps / controls."""
        self.spline = make_spline(times, controls, self.spline_order)

    def reset(self) -> None:
        """Reset the controls, candidate controls and the spline to their default values."""
        self.task.reset()
        if self.optimizer_cfg.num_nodes < 4 and self.spline_order == "cubic":
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            self.optimizer_cfg.num_nodes = 4
        self.nominal_knots = np.tile(self.task.optimizer_warm_start(), (self.optimizer_cfg.num_nodes, 1))
        self.candidate_knots = np.tile(self.nominal_knots, (self.optimizer_cfg.num_rollouts, 1, 1))
        self.times = self.task.data.time + self.spline_timesteps
        self.update_spline(self.times, self.nominal_knots)
        # Reset policy output state for locomotion policy tasks
        if self.task.uses_locomotion_policy:
            self._last_policy_output = None

    def update_traces(self) -> None:
        """Update traces by extracting data from sensors readings.

        We need to have num_spline_points - 1 line segments. Sensors will initially be of shape
        (num_rollout x num_timesteps x nsensordata) and needs to end up being in shape
        (num_elite * num_trace_sensors * size of a single rollout x 2 (first and last point of spline) x 3 (3d pos))
        """
        # Resize traces if forced by config change.
        self.sensor_rollout_size = self.num_timesteps - 1
        self.all_traces_rollout_size = self.sensor_rollout_size * self.num_trace_sensors
        if self.num_trace_elites != min(self.max_num_traces, self.optimizer_cfg.num_rollouts):
            self.num_trace_elites = min(self.max_num_traces, self.optimizer_cfg.num_rollouts)
        sensors = np.repeat(self.sensors, 2, axis=1)

        # Order the actions from best to worst so that the first `num_trace_sensors` x `num_nodes` traces
        # correspond to the best rollout and are using a special colors
        elite_actions = np.argsort(self.rewards)[-self.num_trace_elites :][::-1]

        total_traces_rollouts = int(self.num_trace_elites * self.num_trace_sensors * self.sensor_rollout_size)
        # Calculates list of the elite indicies
        trace_inds = [self.model.sensor_adr[id] + pos for id in self.trace_sensors for pos in range(3)]

        # Filter out the non-elite indices we don't care about
        sensors = sensors[elite_actions, :, :]
        # Remove everything but the trace sensors we care about, leaving htis column as size num_trace_sensors * 3
        sensors = sensors[:, :, trace_inds]
        # Remove the first and last part the trajectory to form line segments properly
        # Array will be doubled and look something like: [(0, 0), (1, 1), (4, 4)]
        # We want it to look like: [(0, 1), (1, 4)]
        sensors = sensors[:, 1:-1, :]

        # We doubled it so the number of entries is going to be the size of the rollout * 2
        separated_sensors_size = (self.num_trace_elites, self.sensor_rollout_size, 2, 3)

        # Each block of (i, self.sensor_rollout_size) needs to be interleaved together into a stack of
        # [block(i, ), block (i + 1, ), ..., block(i + n)]
        elites = np.zeros((self.num_trace_sensors * self.num_trace_elites, self.sensor_rollout_size, 2, 3))
        for sensor in range(self.num_trace_sensors):
            s1 = np.reshape(sensors[:, :, sensor * 3 : (sensor + 1) * 3], separated_sensors_size)
            elites[sensor :: self.num_trace_sensors] = s1
        self.traces = np.reshape(elites, (total_traces_rollouts, 2, 3))

    def update_states(self, state_msg: MujocoState) -> None:
        """Updates the states."""
        self.current_state = np.concatenate([state_msg.qpos, state_msg.qvel])
        self.time = state_msg.time
        self.system_metadata = state_msg.sim_metadata

    def _init_action_normalizer(self) -> Normalizer:
        """Initialize the action normalizer."""
        action_normalizer_kwargs = {}
        if self.action_normalizer_type == "min_max":
            action_normalizer_kwargs["min"] = self.task.actuator_ctrlrange[:, 0]
            action_normalizer_kwargs["max"] = self.task.actuator_ctrlrange[:, 1]
        elif self.action_normalizer_type == "running":
            action_normalizer_kwargs["init_std"] = 1.0  # TODO(yunhai): make this configurable
        return make_normalizer(self.action_normalizer_type, self.model.nu, **action_normalizer_kwargs)


class BatchedControllers:
    """Coordinates multiple controllers sharing a single RolloutBackend.

    This class manages batched rollouts across multiple controllers, executing
    a single GPU rollout for all controllers at each optimization iteration.

    Usage:
        # Create shared backend with num_threads per problem and num_problems
        num_rollouts = 64  # rollouts per controller
        num_problems = 3   # number of controllers
        backend = RolloutBackend(model, num_threads=num_rollouts, num_problems=num_problems)

        # Create batched controller coordinator
        batched = BatchedControllers(config, task, optimizer, backend)

        # Update all controllers with batched rollouts
        batched.update_action()
    """

    def __init__(
        self,
        controller_config: ControllerConfig,
        task: Task,
        optimizer: Optimizer,
        rollout_backend: "MJWarpRolloutBackend",
    ) -> None:
        """Initialize the batched controllers.

        Args:
            controller_config: Configuration for all controllers.
            task: Template task instance (new instances created from its class and model_path).
            optimizer: Template optimizer instance (deep copied for each controller).
            rollout_backend: The shared WarpRolloutBackend instance. Should be initialized with
                num_problems equal to len(controllers).
        """
        self.num_problems = rollout_backend.num_problems
        self.controllers = []
        for _ in range(self.num_problems):
            new_task = task.__class__(model_path=task.model_path)
            new_task.config = copy.deepcopy(task.config)
            controller = Controller(
                controller_config=controller_config,
                task=new_task,
                optimizer=copy.deepcopy(optimizer),
                rollout_backend=rollout_backend,
            )
            self.controllers.append(controller)
        self.rollout_backend = rollout_backend

        # Validate that all controllers use the shared backend
        for i, ctrl in enumerate(self.controllers):
            if ctrl.rollout_backend is not rollout_backend:
                raise ValueError(
                    f"Controller {i} does not use the shared rollout_backend. "
                    "All controllers must share the same RolloutBackend instance."
                )

        # Validate num_problems matches number of controllers
        if rollout_backend.num_problems != len(self.controllers):
            raise ValueError(
                f"RolloutBackend num_problems ({rollout_backend.num_problems}) does not match "
                f"number of controllers ({len(self.controllers)}). "
                f"Initialize backend with num_problems={len(self.controllers)}."
            )

        # Initialize timers for update_action() breakdown
        self.timer_sample_controls = Timer("Sample Controls", unit="ms")
        self.timer_rollout = Timer("Batched Rollout", unit="ms")
        self.timer_rewards = Timer("Rewards       ", unit="ms")
        self.timer_update_iter = Timer("Update Iter   ", unit="ms")
        self.timer_post_opt = Timer("Post Opt      ", unit="ms")

        # Policy output state for hierarchical control (warp array, kept on GPU between rollouts)
        self._last_policy_output = None

    def update_action(self) -> None:
        """Update all controllers with coordinated batched rollouts.

        This method runs the optimization loop across all controllers,
        batching their rollouts into a single GPU execution per iteration.
        """
        # Pre-optimization for all controllers
        for ctrl in self.controllers:
            ctrl._pre_optimization()

        # Run optimization loop - all controllers iterate together
        max_iters = min(ctrl.max_opt_iters for ctrl in self.controllers)
        for _ in range(max_iters):
            # Sample controls for all controllers
            self.timer_sample_controls.tic()
            for ctrl in self.controllers:
                ctrl.rollout_controls = ctrl._sample_controls()
            self.timer_sample_controls.toc()

            # Pre-rollout for all controllers
            for ctrl in self.controllers:
                ctrl._pre_rollout()

            # Execute batched rollout for all controllers
            self.timer_rollout.tic()
            self._execute_batched_rollout()
            self.timer_rollout.toc()

            # Post-rollout (compute rewards) for all controllers
            self.timer_rewards.tic()
            for ctrl in self.controllers:
                ctrl._post_rollout()
            self.timer_rewards.toc()

            # Update iteration for all controllers
            self.timer_update_iter.tic()
            for ctrl in self.controllers:
                ctrl._update_iteration()
            self.timer_update_iter.toc()

        # Post-optimization for all controllers
        self.timer_post_opt.tic()
        for ctrl in self.controllers:
            ctrl._post_optimization()
        self.timer_post_opt.toc()

    def _execute_batched_rollout(self) -> None:
        """Execute a single batched rollout for all controllers."""
        # Collect x0 from all controllers: shape (num_problems, x0_dim)
        x0_stacked = np.stack([ctrl.current_state for ctrl in self.controllers], axis=0)

        # Convert task-space controls to sim-space, then batch: shape (num_problems * num_threads, horizon, model.nu)
        controls_batched = np.concatenate(
            [ctrl.task.task_to_sim_ctrl(ctrl.rollout_controls) for ctrl in self.controllers], axis=0
        )

        # Execute the batched rollout (policy output tensor is passed through directly)
        all_states, all_sensors, self._last_policy_output = self.rollout_backend.rollout(
            x0_stacked,
            controls_batched,
            self._last_policy_output,
        )

        # Distribute states/sensors back to each controller
        num_threads = self.rollout_backend.num_threads
        for i, ctrl in enumerate(self.controllers):
            start_idx = i * num_threads
            end_idx = (i + 1) * num_threads
            ctrl.states = all_states[start_idx:end_idx]
            ctrl.sensors = all_sensors[start_idx:end_idx]

    def reset(self) -> None:
        """Reset all controllers."""
        for ctrl in self.controllers:
            ctrl.reset()
        self._last_policy_output = None

    def print_timer_stats(self) -> None:
        """Print timing statistics for update_action() breakdown."""
        self.timer_sample_controls.print_stats()
        self.timer_rollout.print_stats()
        self.timer_rewards.print_stats()
        self.timer_update_iter.print_stats()
        self.timer_post_opt.print_stats()
        if hasattr(self.rollout_backend, "print_timer_stats"):
            self.rollout_backend.print_timer_stats()

    def reset_timers(self) -> None:
        """Reset all timers."""
        self.timer_sample_controls.reset()
        self.timer_rollout.reset()
        self.timer_rewards.reset()
        self.timer_update_iter.reset()
        self.timer_post_opt.reset()
        if hasattr(self.rollout_backend, "reset_timers"):
            self.rollout_backend.reset_timers()

    def update_states(self, state_msgs: list) -> None:
        """Update states for all controllers.

        Args:
            state_msgs: List of MujocoState messages, one per controller.
        """
        if len(state_msgs) != len(self.controllers):
            raise ValueError(f"Expected {len(self.controllers)} state messages, got {len(state_msgs)}")
        for ctrl, state_msg in zip(self.controllers, state_msgs, strict=False):
            ctrl.update_states(state_msg)

    def set_init_previous_actions(self, previous_actions_list: list[np.ndarray | None]) -> None:
        """Sync previous_actions from sims to the batched policy output state.

        Broadcasts per-problem actions to all threads.

        Args:
            previous_actions_list: List of previous actions arrays, one per controller/problem.
        """
        if all(pa is None for pa in previous_actions_list):
            self._last_policy_output = None
        else:
            pa_np = np.stack([pa for pa in previous_actions_list if pa is not None], axis=0)
            pa_broadcast = np.repeat(pa_np, self.rollout_backend.num_threads, axis=0)
            import warp as wp  # noqa: PLC0415

            self._last_policy_output = wp.array(pa_broadcast, dtype=wp.float32, device=self.rollout_backend.device)


def make_spline(times: np.ndarray, controls: np.ndarray, spline_order: str) -> interp1d:
    """Helper function for creating spline objects.

    Args:
        times: array of times for knot points, shape (T,).
        controls: (possibly batched) array of controls to interpolate, shape (..., T, m).
        spline_order: Order to use for interpolation. Same as parameter for scipy.interpolate.interp1d.
        extrapolate: Whether to allow extrapolation queries. Default true (for re-initialization).
    """
    # fill values for "before" and "after" spline extrapolation.
    fill_value = (controls[..., 0, :], controls[..., -1, :])
    return interp1d(
        times,
        controls,
        kind=spline_order,
        axis=-2,
        copy=False,
        fill_value=fill_value,  # interp1d is incorrectly typed # type: ignore
        bounds_error=False,
    )


def make_controller(
    init_task: str,
    init_optimizer: str,
    task_registration_cfg: DictConfig | None = None,
    optimizer_registration_cfg: DictConfig | None = None,
    rollout_backend: "Literal['mujoco'] | MJWarpRolloutBackend" = "mujoco",
) -> Controller:
    """Make a controller.

    Args:
        init_task: The task name to use.
        init_optimizer: The optimizer name to use.
        task_registration_cfg: Optional task registration config.
        optimizer_registration_cfg: Optional optimizer registration config.
        rollout_backend: Either a backend type string ("mujoco") to create a new backend,
            or an existing WarpRolloutBackend instance to share with other controllers.

    Returns:
        The created Controller instance.
    """
    available_optimizers = get_registered_optimizers()
    available_tasks = get_registered_tasks()
    if task_registration_cfg is not None:
        register_tasks_from_cfg(task_registration_cfg)
    if optimizer_registration_cfg is not None:
        register_optimizers_from_cfg(optimizer_registration_cfg)

    task_entry = available_tasks.get(init_task)
    optimizer_entry = available_optimizers.get(init_optimizer)

    assert task_entry is not None, f"Task {init_task} not found in task registry."
    assert optimizer_entry is not None, f"Optimizer {init_optimizer} not found in optimizer registry."

    # instantiate the task/optimizer/controller
    task_cls, _ = task_entry
    task = task_cls()

    optimizer_cls, optimizer_config_cls = optimizer_entry
    optimizer_cfg = optimizer_config_cls()
    optimizer_cfg.set_override(init_task)
    optimizer = optimizer_cls(optimizer_cfg, task.nu)

    controller_cfg = ControllerConfig()
    controller_cfg.set_override(init_task)

    return Controller(
        controller_config=controller_cfg,
        task=task,
        optimizer=optimizer,
        rollout_backend=rollout_backend,
    )
