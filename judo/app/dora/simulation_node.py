# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
import warnings
from typing import Callable

from dora_utils.dataclasses import from_arrow, to_arrow
from dora_utils.node import DoraNode, on_event
from omegaconf import DictConfig

from judo.simulation import DEFAULT_SIMULATION_BACKEND_REGISTRY
from judo.simulation.base import Simulation
from judo.structs import SplineData
from judo.tasks import get_registered_tasks


class SimulationNode(DoraNode):
    """The simulation node."""

    def __init__(
        self,
        node_id: str = "simulation",
        init_task: str = "cylinder_push",
        max_workers: int | None = None,
        task_registration_cfg: DictConfig | None = None,
        backend_registry: dict[str, type[Simulation]] | None = None,
    ) -> None:
        """Initialize the simulation node.

        Args:
            node_id: Identifier for this dora node.
            init_task: Name of the task to initialize.
            max_workers: Maximum number of worker threads for dora (None = auto).
            task_registration_cfg: Optional config for task registration overrides.
            backend_registry: Optional mapping of backend names → Simulation classes. Checked first before built-in registry.
        """
        super().__init__(node_id=node_id, max_workers=max_workers)
        self._task_registration_cfg = task_registration_cfg
        self._backend_registry = dict(DEFAULT_SIMULATION_BACKEND_REGISTRY)
        self._backend_registry.update(backend_registry or {})
        self._init_sim(init_task)
        self.control_spline: Callable | None = None
        self.write_states()

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
        sim_kwargs: dict = {"init_task": task_name, "task_registration_cfg": self._task_registration_cfg}
        # The hierarchical backend needs the low-level policy path passed explicitly (the
        # Simulation classes are decoupled from the task registry).
        if task_entry.simulation_backend == "mujoco_hierarchical":
            sim_kwargs["locomotion_policy_path"] = task_entry.locomotion_policy_path
        self.sim = sim_backend_cls(**sim_kwargs)

    @on_event("INPUT", "task")
    def update_task(self, event: dict) -> None:
        """Event handler for processing task updates."""
        new_task = event["value"].to_numpy(zero_copy_only=False)[0]
        self._init_sim(new_task)
        self.control_spline = None  # Clear stale spline

    def spin(self) -> None:
        """Spin logic for the simulation node."""
        try:
            while True:
                start_time = time.time()
                self.parse_messages()

                if self.control_spline is not None:
                    command = self.control_spline(self.sim.task.data.time)
                    if command.shape[-1] == self.sim.task.nu:
                        self.sim.step(command)
                    else:
                        warnings.warn(
                            f"Control command has wrong number of dimensions! Expected {self.sim.task.nu}, got {command.shape[-1]}",
                            stacklevel=2,
                        )

                self.write_states()

                # Force simulation node to run at fixed rate specified by simulation timestep (specified in the model).
                dt_des = self.sim.timestep
                dt_elapsed = time.time() - start_time
                if dt_elapsed < dt_des:
                    time.sleep(dt_des - dt_elapsed)
                else:
                    warnings.warn(
                        f"Sim step {dt_elapsed:.3f} longer than desired step {dt_des:.3f}!",
                        stacklevel=2,
                    )
        except KeyboardInterrupt:
            pass

    def write_states(self) -> None:
        """Reads data from simulation and writes to output topic."""
        arr, metadata = to_arrow(self.sim.sim_state)
        self.node.send_output("states", arr, metadata)
        arr, metadata = to_arrow(self.sim.render_pose)
        self.node.send_output("render_pose", arr, metadata)

    @on_event("INPUT", "sim_pause")
    def set_paused_status(self, event: dict) -> None:
        """Event handler for processing pause status updates."""
        self.sim.pause()

    @on_event("INPUT", "task_reset")
    def reset_task(self, event: dict) -> None:
        """Resets the task."""
        self.sim.task.reset()

    @on_event("INPUT", "controls")
    def update_control(self, event: dict) -> None:
        """Event handler for processing controls received from controller node."""
        spline_data = from_arrow(event["value"], event["metadata"], SplineData)
        self.control_spline = spline_data.spline()
