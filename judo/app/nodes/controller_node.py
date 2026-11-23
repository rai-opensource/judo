# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import copy
import dataclasses
import time
from typing import Any, Callable, TypeGuard

import numpy as np
from omegaconf import DictConfig

from judo.app.structs import GuiState, MujocoState, SplineData
from judo.controller import Controller, make_controller
from judo.unlocked.const import const_cast


class ControllerNode:
    """Controller node. Plans at ``control_freq`` from the latest simulation state."""

    def __init__(
        self,
        init_task: str = "cylinder_push",
        init_optimizer: str = "cem",
        task_registration_cfg: DictConfig | None = None,
        optimizer_registration_cfg: DictConfig | None = None,
        make_controller_fn: Callable | None = None,
    ) -> None:
        """Store construction args. The controller itself is built in :meth:`setup`, after the process forks.

        Args:
            init_task: Name of the task to initialize.
            init_optimizer: Name of the optimizer to initialize (e.g., "cem", "ps", "mppi").
            task_registration_cfg: Optional config for task registration overrides.
            optimizer_registration_cfg: Optional config for optimizer registration overrides.
            make_controller_fn: Optional factory function to create Controller instances.
                Defaults to judo.controller.make_controller. Allows custom controller creation.
        """
        self._init_task = init_task
        self._init_optimizer = init_optimizer
        self._make_controller_fn = make_controller_fn or make_controller
        self._task_registration_cfg = task_registration_cfg
        self._optimizer_registration_cfg = optimizer_registration_cfg
        self._paused = False
        # what the controller currently runs, compared with each incoming GuiState
        self._task_name = init_task
        self._optimizer_name = init_optimizer
        self._applied_revisions: dict[str, int] = {}
        # Last control period sent to the clock node. ``step`` publishes the period to the clock only when control_freq
        # changes (startup, GUI config edit, task switch), so the clock updates its tick rate and triggers the controller
        # at the new frequency; unchanged periods are not republished. None forces the first publish.
        self._published_period: float | None = None

    def setup(self) -> None:
        """Build the controller (runs once the node's process has started)."""
        self.controller = self._build_controller(self._init_task, self._init_optimizer)

    def _build_controller(self, task_name: str, optimizer_name: str) -> Controller:
        """Build controller using the task's registered rollout backend."""
        return self._make_controller_fn(
            init_task=task_name,
            init_optimizer=optimizer_name,
            task_registration_cfg=self._task_registration_cfg,
            optimizer_registration_cfg=self._optimizer_registration_cfg,
        )

    def _apply_gui_state(self, state: GuiState) -> None:
        """Apply what changed in ``state`` since the last applied one: task first, then optimizer, reset, configs."""
        if state.task != self._task_name:
            if state.task not in self.controller.available_tasks:
                raise ValueError(f"Task {state.task} not found in task registry.")
            self.controller = self._build_controller(state.task, state.optimizer)
            self._task_name, self._optimizer_name = state.task, state.optimizer
        elif state.optimizer != self._optimizer_name:
            optimizer_entry = self.controller.available_optimizers.get(state.optimizer)
            if optimizer_entry is None:
                raise ValueError(f"Optimizer {state.optimizer} not found in optimizer registry.")
            optimizer_cls, optimizer_config_cls = optimizer_entry
            optimizer_config = optimizer_config_cls()
            optimizer_config.set_override(self.controller.task.name)
            self.controller.optimizer = optimizer_cls(optimizer_config, self.controller.task.nu)
            self._optimizer_name = state.optimizer
        self._paused = state.sim_paused

        moved = state.moved(self._applied_revisions)
        self._applied_revisions = dict(state.revisions)
        if "task_reset" in moved:
            self.controller.reset()
        # A config the GUI has not rebuilt yet for a new task/optimizer has different fields and is skipped; the
        # rebuilt one follows with a newer revision. Configs are copied out of the shared-memory message.
        if "controller_config" in moved and _fits(state.controller_config, type(self.controller.controller_cfg)):
            self.controller.controller_cfg = type(self.controller.controller_cfg)(
                **copy.deepcopy(state.controller_config)
            )
        if "optimizer_config" in moved and _fits(state.optimizer_config, self.controller.optimizer_config_cls):
            self.controller.optimizer_cfg = self.controller.optimizer_config_cls(
                **copy.deepcopy(state.optimizer_config)
            )
        if "task_config" in moved and _fits(state.task_config, type(self.controller.task_config)):
            self.controller.task_config = type(self.controller.task_config)(**copy.deepcopy(state.task_config))

    def step(
        self,
        tick: int,
        states: MujocoState,
        gui_state: GuiState | None = None,
    ) -> tuple[SplineData, tuple[np.ndarray, int] | None, float | None, float | None]:
        """Plan once per clock tick: apply the latest GUI state, update the state, plan, and publish the controls.

        ``gui_state`` is None when the GUI has not changed since the last step.

        Returns:
            The control spline, the rollout traces with their rollout size (if any), the plan time (if planned), and
            the control period ``1 / control_freq`` when it changed (first step, config change or task switch) so the
            clock can retune; None otherwise.
        """
        # inputs arrive wrapped in read-only ``Const`` proxies
        states = const_cast(states)
        if gui_state is not None:
            self._apply_gui_state(const_cast(gui_state))

        plan_time = None
        # Stale states from a previous task (task switch race) have the wrong dimensions and are ignored.
        model = self.controller.model
        if len(states.qpos) == model.nq and len(states.qvel) == model.nv:
            self.controller.update_states(states)
            if not self._paused:
                start = time.perf_counter()
                self.controller.update_action()
                plan_time = time.perf_counter() - start

        traces = self.controller.traces
        traces_out = (
            (traces, int(self.controller.all_traces_rollout_size)) if traces is not None and len(traces) > 0 else None
        )

        # The rate is set by the clock node (controller_clock_tick), which knows nothing about control_freq: it is built
        # with a placeholder period and only learns the real one from this output. control_freq can change at runtime
        # (GUI controller config edit, per-task override on a task switch), so the new period must be sent to the clock
        # whenever it changes, or the controller keeps planning at the old rate. Unchanged periods return None, which
        # publishes nothing.
        period = 1 / self.controller.controller_cfg.control_freq
        period_out = None
        if period != self._published_period:
            period_out = self._published_period = period
        return self.controller.spline_data, traces_out, plan_time, period_out


def _fits(config: dict[str, Any] | None, cls: type) -> TypeGuard[dict[str, Any]]:
    """Whether ``config`` is set and has exactly the fields of the config dataclass ``cls``."""
    return config is not None and set(config) == {f.name for f in dataclasses.fields(cls)}
