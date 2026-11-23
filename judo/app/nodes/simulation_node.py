# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
import warnings
from typing import Callable

from omegaconf import DictConfig

from judo.app.structs import GuiState, MujocoState, RenderPose, SplineData
from judo.simulation import DEFAULT_SIMULATION_BACKEND_REGISTRY
from judo.simulation.base import Simulation
from judo.tasks import get_registered_tasks
from judo.unlocked.const import const_cast


class SimulationNode:
    """The simulation node. Steps the simulation at its model timestep, driven by the latest controls."""

    def __init__(
        self,
        init_task: str = "cylinder_push",
        task_registration_cfg: DictConfig | None = None,
        backend_registry: dict[str, type[Simulation]] | None = None,
    ) -> None:
        """Store construction args. The simulation itself is built in :meth:`setup`, after the process forks.

        Args:
            init_task: Name of the task to initialize.
            task_registration_cfg: Optional config for task registration overrides.
            backend_registry: Optional mapping of backend names → Simulation classes. Checked first before built-in registry.
        """
        self._init_task = init_task
        self._task_registration_cfg = task_registration_cfg
        self._backend_registry = dict(DEFAULT_SIMULATION_BACKEND_REGISTRY)
        self._backend_registry.update(backend_registry or {})
        self.control_spline: Callable | None = None
        # compared with each incoming GuiState
        self._task_name = init_task
        self._applied_revisions: dict[str, int] = {}
        # Last timestep sent to the clock node. ``step`` publishes the timestep to the clock only when it changes
        # (startup, task switch), so the clock updates its tick rate and triggers the simulation at the new frequency;
        # unchanged timesteps are not republished. None forces the first publish.
        self._published_timestep: float | None = None

    def setup(self) -> None:
        """Build the simulation (runs once the node's process has started)."""
        self._init_sim(self._init_task)

    def _resolve_backend(self, backend_name: str) -> type[Simulation]:
        """Resolve a simulation backend class by name from merged registry."""
        backend_cls = self._backend_registry.get(backend_name)
        if backend_cls is None:
            raise KeyError(f"Unknown simulation backend: {backend_name!r}")
        return backend_cls

    def _init_sim(self, task_name: str) -> None:
        """Initialize simulation using the task's registered simulation backend."""
        task_entry = get_registered_tasks().get(task_name)
        if task_entry is None:
            raise ValueError(f"Task {task_name} not found in task registry.")

        sim_backend_cls = self._resolve_backend(task_entry.simulation_backend)
        self.sim = sim_backend_cls(init_task=task_name, task_registration_cfg=self._task_registration_cfg)

    def _apply_gui_state(self, state: GuiState) -> None:
        """Apply what changed in ``state`` since the last applied one (see :class:`GuiState`)."""
        if state.task != self._task_name:
            self._init_sim(state.task)
            self._task_name = state.task
            self.control_spline = None  # clear stale spline
        if "task_reset" in state.moved(self._applied_revisions):
            self.sim.task.reset()
        self._applied_revisions = dict(state.revisions)
        self.sim.paused = state.sim_paused

    def step(
        self,
        tick: int,
        controls: SplineData | None,
        gui_state: GuiState | None = None,
    ) -> tuple[MujocoState, RenderPose, float | None]:
        """Advance the simulation one step per clock tick, applying the latest GUI state and controls.

        ``gui_state`` is None when the GUI has not changed since the last step.

        Returns:
            The new state, the render pose, and the model timestep when it changed (first step or task switch) so
            the clock can retune its period; None otherwise.
        """
        start_time = time.time()
        # inputs arrive wrapped in read-only ``Const`` proxies
        controls = const_cast(controls)
        if gui_state is not None:
            self._apply_gui_state(const_cast(gui_state))
        if controls is not None:
            self.control_spline = controls.spline()

        if self.control_spline is not None:
            command = self.control_spline(self.sim.task.data.time)
            if command.shape[-1] == self.sim.task.nu:
                self.sim.step(command)
            else:
                warnings.warn(
                    f"Control command has wrong number of dimensions! Expected {self.sim.task.nu}, got {command.shape[-1]}",
                    stacklevel=2,
                )

        states, render_pose = self.sim.sim_state, self.sim.render_pose

        # The rate is set by the clock node; only report overruns here.
        dt_des = self.sim.timestep
        dt_elapsed = time.time() - start_time
        if dt_elapsed > dt_des:
            warnings.warn(f"Sim step {dt_elapsed:.3f} longer than desired step {dt_des:.3f}!", stacklevel=2)

        # The rate is set by the clock node (sim_clock_tick), which knows nothing about the model timestep: it is built
        # with a placeholder period and only learns the real one from this output. The timestep can change at runtime
        # (it differs between tasks, e.g. cartpole 0.04 s vs cylinder_push 0.02 s), so the new timestep must be sent to
        # the clock whenever it changes, or the sim keeps stepping at the old rate. Unchanged timesteps return None,
        # which publishes nothing.
        timestep = None
        if dt_des != self._published_timestep:
            timestep = self._published_timestep = dt_des
        return states, render_pose, timestep
