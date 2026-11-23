# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import dataclasses
import threading
import warnings
from typing import Any

import numpy as np

from judo.app.structs import GuiState, RenderPose
from judo.tasks import TaskRegistration
from judo.unlocked.const import const_cast
from judo.unlocked.node import Node, NodeType
from judo.visualizers.visualizer import Visualizer


class VisualizationNode:
    """The visualization node (Viser web GUI for task/optimizer control).

    Two graph nodes share this instance: ``gui_step`` renders the latest sim/controller data, and ``gui_events`` is
    a CALLBACK node that the Viser GUI handlers drive directly, publishing a :class:`GuiState` snapshot the moment
    any setting changes.
    """

    def __init__(
        self,
        init_task: str = "cylinder_push",
        init_optimizer: str = "cem",
        task_registration_cfg: dict | None = None,
        optimizer_registration_cfg: dict | None = None,
        controller_override_cfg: dict | None = None,
        optimizer_override_cfg: dict | None = None,
        sim_pause_button: bool = True,
        geom_exclude_substring: str = "collision",
        available_tasks: dict[str, TaskRegistration] | None = None,
    ) -> None:
        """Store construction args. The Viser server is started in :meth:`setup`, after the process forks.

        Args:
            init_task: Name of the task to initialize.
            init_optimizer: Name of the optimizer to initialize (e.g., "cem", "ps").
            task_registration_cfg: Optional config for task registration overrides.
            optimizer_registration_cfg: Optional config for optimizer registration overrides.
            controller_override_cfg: Optional config overrides for the controller.
            optimizer_override_cfg: Optional config overrides for the optimizer.
            sim_pause_button: Whether to display a simulation pause button in the GUI.
            geom_exclude_substring: Geometry name substring to exclude from visualization (default "collision" hides collision shapes).
            available_tasks: Optional pre-computed mapping of task names to TaskRegistration entries
                for the task selector. If None, tasks are inferred from the task registry.
        """
        self._visualizer_kwargs: dict[str, Any] = {
            "init_task": init_task,
            "init_optimizer": init_optimizer,
            "task_registration_cfg": task_registration_cfg,
            "optimizer_registration_cfg": optimizer_registration_cfg,
            "controller_override_cfg": controller_override_cfg,
            "optimizer_override_cfg": optimizer_override_cfg,
            "sim_pause_button": sim_pause_button,
            "geom_exclude_substring": geom_exclude_substring,
            "available_tasks": available_tasks,
        }
        # Built here, not in setup: the graph wires this node when it materializes, before the process forks.
        self._events_node = Node("gui_events", self._state_output, node_type=NodeType.CALLBACK)
        # Viser runs handlers on a thread pool, and ``Node.callback`` must not run concurrently on one node.
        self._publish_lock = threading.RLock()
        # GuiState.revisions: bumped per reset click / config edit so consumers know what to (re)apply
        self._revisions = {"task_reset": 0, "controller_config": 0, "optimizer_config": 0, "task_config": 0}
        self._initial_state_sent = False

    def setup(self) -> None:
        """Start the Viser GUI (runs once the node's process has started)."""
        self.visualizer = Visualizer(**self._visualizer_kwargs, on_event=self._on_gui_event)

    def events_node(self) -> Node:
        """The externally driven node publishing the :class:`GuiState` (wired as ``gui_events``)."""
        return self._events_node

    def _state_output(self, state: GuiState) -> GuiState:
        """Target of ``gui_events``: publish ``state`` on its ``gui_state`` port."""
        return state

    def _on_gui_event(self, event: str) -> None:
        """Visualizer ``on_event`` hook: record the change and publish the new GUI state right away."""
        with self._publish_lock:
            if event in self._revisions:
                self._revisions[event] += 1
            if "visualizer" in self.__dict__:  # still being built in ``setup``; ``step`` sends the initial state
                self._publish_state()

    def _publish_state(self) -> None:
        v = self.visualizer
        with self._publish_lock:
            # each getter takes the lock guarding its value so the snapshot is consistent
            with v.controller_config_lock:
                controller_config = dataclasses.asdict(v.controller_config)
            with v.optimizer_config_lock:
                optimizer_config = dataclasses.asdict(v.optimizer_config)
            with v.task_config_lock:
                task_config = dataclasses.asdict(v.task_config)
            sim_pause_button = v.gui_elements.get("sim_pause_button")
            state = GuiState(
                task=v.task_name,
                optimizer=v.optimizer_name,
                sim_paused=sim_pause_button is not None and sim_pause_button.label == "Resume Simulation",
                controller_config=controller_config,
                optimizer_config=optimizer_config,
                task_config=task_config,
                revisions=dict(self._revisions),
            )
            self._events_node.callback(state)

    def step(self, render_pose: RenderPose, traces: tuple[np.ndarray, int] | None, plan_time: float | None) -> None:
        """Update the GUI from the latest simulation/controller data."""
        if not self._initial_state_sent:
            # The Visualizer announces its initial configs while it is built in ``setup``, before ``self.visualizer``
            # is assigned; send the initial state once the graph is running.
            self._initial_state_sent = True
            self._publish_state()
        # inputs arrive wrapped in read-only ``Const`` proxies
        render_pose, traces = const_cast(render_pose), const_cast(traces)
        self._update_render_pose(render_pose)
        if traces is not None:
            traces_arr, all_traces_rollout_size = traces
            with self.visualizer.task_lock:
                self.visualizer.viser_model.set_traces(traces_arr, all_traces_rollout_size)
        # GUI element might not exist during task switch
        if plan_time is not None and "plan_time_display" in self.visualizer.gui_elements:
            self.visualizer.gui_elements["plan_time_display"].value = plan_time * 1000  # ms

    def _update_render_pose(self, render_pose: RenderPose) -> None:
        if self.visualizer.controller_config.spline_order == "cubic" and self.visualizer.optimizer_config.num_nodes < 4:
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            for e in self.visualizer.gui_elements["optimizer_params"]:
                if e.label == "num_nodes":
                    e.value = 4
                    break
            self.visualizer.optimizer_config_updated.set()

        try:
            with self.visualizer.task_lock:
                self.visualizer.data.xpos[:] = render_pose.xpos
                self.visualizer.data.xquat[:] = render_pose.xquat
                self.visualizer.viser_model.set_data(self.visualizer.data)
        except ValueError:
            # we're switching tasks and the new task has a different number of xpos/xquat
            return

    def cleanup(self) -> None:
        """Cleanup the visualization node."""
        self.visualizer.cleanup()
