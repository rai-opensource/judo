# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings

import pyarrow as pa
from dora_utils.dataclasses import from_arrow, to_arrow
from dora_utils.node import DoraNode, on_event
from omegaconf import DictConfig
from viser import GuiFolderHandle, GuiImageHandle, GuiInputHandle, IcosphereHandle, MeshHandle

from judo.structs import RenderPose
from judo.tasks import TaskRegistration
from judo.visualizers.visualizer import Visualizer

ElementType = GuiImageHandle | GuiInputHandle | GuiFolderHandle | MeshHandle | IcosphereHandle


class VisualizationNode(DoraNode):
    """The visualization node."""

    def __init__(
        self,
        node_id: str = "visualization",
        max_workers: int | None = None,
        init_task: str = "cylinder_push",
        init_optimizer: str = "cem",
        task_registration_cfg: DictConfig | None = None,
        optimizer_registration_cfg: DictConfig | None = None,
        controller_override_cfg: DictConfig | None = None,
        optimizer_override_cfg: DictConfig | None = None,
        sim_pause_button: bool = True,
        geom_exclude_substring: str = "collision",
        available_tasks: dict[str, TaskRegistration] | None = None,
    ) -> None:
        """Initialize the visualization node (Viser web GUI for task/optimizer control).

        Args:
            node_id: Identifier for this dora node.
            max_workers: Maximum number of worker threads for dora (None = auto).
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
        super().__init__(node_id=node_id, max_workers=max_workers)
        self.visualizer = Visualizer(
            init_task=init_task,
            init_optimizer=init_optimizer,
            task_registration_cfg=task_registration_cfg,
            optimizer_registration_cfg=optimizer_registration_cfg,
            controller_override_cfg=controller_override_cfg,
            optimizer_override_cfg=optimizer_override_cfg,
            sim_pause_button=sim_pause_button,
            geom_exclude_substring=geom_exclude_substring,
            available_tasks=available_tasks,
        )

    def write_sim_pause(self) -> None:
        """Write the sim pause signal to the GUI."""
        with self.visualizer.sim_pause_lock:
            self.node.send_output("sim_pause", pa.array([1]))  # dummy value
        self.visualizer.sim_pause_updated.clear()

    def write_task(self) -> None:
        """Write the task name to the GUI."""
        with self.visualizer.task_lock:
            self.node.send_output("task", pa.array([self.visualizer.task_name]))
        self.visualizer.task_updated.clear()

    def write_task_reset(self) -> None:
        """Write the task reset signal to the GUI."""
        with self.visualizer.task_lock:
            self.node.send_output("task_reset", pa.array([1]))  # dummy value
        self.visualizer.task_reset_updated.clear()

    def write_optimizer(self) -> None:
        """Write the optimizer name to the GUI."""
        with self.visualizer.optimizer_lock:
            self.node.send_output("optimizer", pa.array([self.visualizer.optimizer_name]))
        self.visualizer.optimizer_updated.clear()

    def write_controller_config(self) -> None:
        """Write the controller config to the GUI."""
        with self.visualizer.controller_config_lock:
            self.node.send_output("controller_config", *to_arrow(self.visualizer.controller_config))
        self.visualizer.controller_config_updated.clear()

    def write_optimizer_config(self) -> None:
        """Write the optimizer config to the GUI."""
        with self.visualizer.optimizer_config_lock:
            self.node.send_output("optimizer_config", *to_arrow(self.visualizer.optimizer_config))
        self.visualizer.optimizer_config_updated.clear()

    def write_task_config(self) -> None:
        """Write the task config to the GUI."""
        with self.visualizer.task_config_lock:
            self.node.send_output("task_config", *to_arrow(self.visualizer.task_config))
        self.visualizer.task_config_updated.clear()

    @on_event("INPUT", "render_pose")
    def update_states(self, event: dict) -> None:
        """Callback to update states on receiving a new state measurement."""
        if self.visualizer.controller_config.spline_order == "cubic" and self.visualizer.optimizer_config.num_nodes < 4:
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            for e in self.visualizer.gui_elements["optimizer_params"]:
                if e.label == "num_nodes":
                    e.value = 4
                    break
            self.visualizer.optimizer_config_updated.set()

        render_pose_msg = from_arrow(event["value"], event["metadata"], RenderPose)
        try:
            with self.visualizer.task_lock:
                self.visualizer.data.xpos[:] = render_pose_msg.xpos
                self.visualizer.data.xquat[:] = render_pose_msg.xquat
                self.visualizer.viser_model.set_data(self.visualizer.data)
        except ValueError:
            # we're switching tasks and the new task has a different number of xpos/xquat
            return

    @on_event("INPUT", "traces")
    def update_traces(self, event: dict) -> None:
        """Callback to update traces on receiving a new trace measurement."""
        traces_flat = event["value"].to_numpy()
        all_traces_rollout_size = int(event["metadata"]["all_traces_rollout_size"])
        shape = event["metadata"]["shape"]
        traces = traces_flat.reshape(*shape)
        with self.visualizer.task_lock:
            self.visualizer.viser_model.set_traces(traces, all_traces_rollout_size)

    @on_event("INPUT", "plan_time")
    def update_plan_time(self, event: dict) -> None:
        """Callback to update plan time on receiving a new plan time measurement."""
        plan_time_s = event["value"].to_numpy(zero_copy_only=False)[0]
        # GUI element might not exist during task switch
        if "plan_time_display" in self.visualizer.gui_elements:
            self.visualizer.gui_elements["plan_time_display"].value = plan_time_s * 1000  # ms

    def spin(self) -> None:
        """Spin logic for the visualization node."""
        try:
            for event in self.node:
                if self.visualizer.sim_pause_updated.is_set():
                    self.write_sim_pause()
                if self.visualizer.task_updated.is_set():
                    self.write_task()
                if self.visualizer.task_reset_updated.is_set():
                    self.write_task_reset()
                if self.visualizer.optimizer_updated.is_set():
                    self.write_optimizer()
                if self.visualizer.controller_config_updated.is_set():
                    self.write_controller_config()
                if self.visualizer.optimizer_config_updated.is_set():
                    self.write_optimizer_config()
                if self.visualizer.task_config_updated.is_set():
                    self.write_task_config()

                self.handle(event)
        except KeyboardInterrupt:
            pass

    def cleanup(self) -> None:
        """Cleanup the visualization node."""
        self.visualizer.cleanup()
        super().cleanup()
