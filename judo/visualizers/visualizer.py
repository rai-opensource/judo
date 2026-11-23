# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import threading
from typing import Callable

import mujoco
import numpy as np
import viser
from omegaconf import DictConfig
from PIL import Image
from viser import GuiFolderHandle, GuiImageHandle, GuiInputHandle, IcosphereHandle, MeshHandle

from judo import PACKAGE_ROOT
from judo.app.utils import register_optimizers_from_cfg, register_tasks_from_cfg
from judo.config import set_config_overrides
from judo.controller import ControllerConfig
from judo.gui import create_gui_elements
from judo.optimizers import get_registered_optimizers
from judo.tasks import TaskRegistration, get_registered_tasks
from judo.visualizers.model import ViserMjModel

ElementType = GuiImageHandle | GuiInputHandle | GuiFolderHandle | MeshHandle | IcosphereHandle


class _NotifyingEvent(threading.Event):
    """A ``threading.Event`` that also calls ``on_set`` every time it is set, to push GUI changes immediately."""

    def __init__(self, on_set: Callable[[], None]) -> None:
        super().__init__()
        self._on_set = on_set

    def set(self) -> None:
        """Set the flag and notify."""
        super().set()
        self._on_set()


class Visualizer:
    """Visualizer for a visualization node to reference.

    This class is a small container which includes the data required for a visualization node to run. This includes all
    of the locks and events required to run. Middleware nodes should keep data within this class and implement methods
    to send, process, and receive data from here.
    """

    def __init__(
        self,
        init_task: str = "cylinder_push",
        init_optimizer: str = "cem",
        task_registration_cfg: DictConfig | None = None,
        optimizer_registration_cfg: DictConfig | None = None,
        controller_override_cfg: DictConfig | None = None,
        optimizer_override_cfg: DictConfig | None = None,
        sim_pause_button: bool = True,
        geom_exclude_substring: str = "collision",
        available_tasks: dict[str, TaskRegistration] | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the visualization node.

        Args:
            init_task: Name of the task to initialize.
            init_optimizer: Name of the optimizer to initialize.
            task_registration_cfg: Optional config for custom task registration.
            optimizer_registration_cfg: Optional config for custom optimizer registration.
            controller_override_cfg: Optional task-specific controller config overrides.
            optimizer_override_cfg: Optional task-specific optimizer config overrides.
            sim_pause_button: Whether to display a simulation pause button.
            geom_exclude_substring: Geometry name substring to exclude from visualization.
            available_tasks: Optional mapping of task names to registrations for the task selector.
            on_event: Optional hook called with the event name ("task", "task_reset", "sim_pause", "optimizer",
                "controller_config", "optimizer_config" or "task_config") each time the GUI sets that change flag,
                so the change can be published immediately instead of polling the flags.
        """
        self._on_event = on_event
        # handling custom task and optimizer registration
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)
        if optimizer_registration_cfg is not None:
            register_optimizers_from_cfg(optimizer_registration_cfg)

        # starting the server
        self.server = viser.ViserServer()
        self.available_tasks = available_tasks or get_registered_tasks()
        self.available_optimizers = get_registered_optimizers()
        self.geom_exclude_substring = geom_exclude_substring

        self.task_name = ""
        self.optimizer_name = ""

        self.task_lock = threading.Lock()
        self.task_updated = self._make_event("task")

        self.sim_pause_button = sim_pause_button
        if self.sim_pause_button:
            self.sim_pause_lock = threading.Lock()
            self.sim_pause_updated = self._make_event("sim_pause")

        # handling task-specific config overrides
        if controller_override_cfg is not None:
            self.register_controller_config_overrides(controller_override_cfg)
        if optimizer_override_cfg is not None:
            self.register_optimizer_config_overrides(optimizer_override_cfg)

        self.set_task(init_task, init_optimizer)

    def _make_event(self, name: str) -> threading.Event:
        """Create the change flag for ``name``; setting it also calls the ``on_event`` hook, if any."""
        on_event = self._on_event
        if on_event is None:
            return threading.Event()
        return _NotifyingEvent(lambda: on_event(name))

    def register_controller_config_overrides(self, controller_override_cfg: DictConfig) -> None:
        """Register task-specific controller config overrides.

        We do this in the Visualization node because this is the "master" node in the graph.
        """
        # as opposed to the optimizer, there is only one Controller class, so for each task, we simply need to specify
        # the controller parameter overrides
        for task_name in controller_override_cfg.keys():
            field_override_values = controller_override_cfg.get(task_name, {})
            set_config_overrides(str(task_name), ControllerConfig, field_override_values)

    def register_optimizer_config_overrides(self, optimizer_override_cfg: DictConfig) -> None:
        """Register task-specific optimizer config overrides.

        We do this in the Visualization node because this is the "master" node in the graph.
        """
        for task_name in optimizer_override_cfg.keys():  # task name in registry
            for optimizer_name in optimizer_override_cfg[task_name].keys():  # optimizer name in registry
                _, cfg_cls = get_registered_optimizers().get(optimizer_name, (None, None))
                if cfg_cls is None:
                    raise ValueError(f"Optimizer {optimizer_name} not found in the registry!")
                field_override_values = optimizer_override_cfg[task_name].get(optimizer_name, {})
                set_config_overrides(str(task_name), cfg_cls, field_override_values)

    def set_task(self, task_name: str, optimizer_name: str) -> None:
        """Helper to initialize task from task name."""
        self.task_name = task_name
        self.optimizer_name = optimizer_name

        task_entry = self.available_tasks.get(task_name)
        if task_entry is None:
            raise ValueError(f"Task {task_name} not found in the task registry.")

        self.task = task_entry.task_type()
        self.task_config = self.task.config
        self.data = mujoco.MjData(self.task.model)
        self.viser_model = ViserMjModel(
            self.server,
            self.task.spec,
            geom_exclude_substring=self.geom_exclude_substring,
        )

        optimizer_entry = self.available_optimizers.get(optimizer_name)
        if optimizer_entry is None:
            raise ValueError(f"Optimizer {optimizer_name} not found in optimizer registry.")
        _, optimizer_config_cls = optimizer_entry

        self.controller_config_lock = threading.Lock()
        self.controller_config_updated = self._make_event("controller_config")
        self.controller_config = ControllerConfig()
        self.controller_config.set_override(task_name)

        self.optimizer_lock = threading.Lock()
        self.optimizer_updated = self._make_event("optimizer")

        self.optimizer_config_lock = threading.Lock()
        self.optimizer_config_updated = self._make_event("optimizer_config")
        self.optimizer_config = optimizer_config_cls()
        self.optimizer_config.set_override(task_name)

        self.task_config_lock = threading.Lock()
        self.task_config_updated = self._make_event("task_config")

        self.task_reset_updated = self._make_event("task_reset")

        self.setup_gui()

        # send the configs to the other nodes
        self.controller_config_updated.set()
        self.task_config_updated.set()
        self.optimizer_config_updated.set()

    def setup_gui(self) -> None:
        """Set up the GUI for the visualization node."""
        self.gui_elements = {}

        # add the Judo logo
        logo_path = PACKAGE_ROOT / "app" / "asset" / "viser-logo-light.png"
        logo = self.server.gui.add_image(np.array(Image.open(logo_path)))
        self.gui_elements["logo"] = logo

        # create the dropdown to select the task
        task_dropdown = self.server.gui.add_dropdown(
            "task",
            list(self.available_tasks.keys()),
            initial_value=self.task_name,
        )
        self.gui_elements["task"] = task_dropdown

        # create the dropdown to select the optimizer
        optimizer_dropdown = self.server.gui.add_dropdown(
            "optimizer",
            list(self.available_optimizers.keys()),
            initial_value=self.optimizer_name,
        )
        self.gui_elements["optimizer"] = optimizer_dropdown

        # create a task reset button
        reset_button = self.server.gui.add_button("Reset Task")
        self.gui_elements["reset_button"] = reset_button

        @reset_button.on_click
        def _(_: viser.GuiEvent) -> None:
            """Callback for when the reset button is clicked."""
            self.task_reset_updated.set()

        # create a config reset button
        config_reset_button = self.server.gui.add_button("Reset Configs")
        self.gui_elements["config_reset_button"] = config_reset_button

        @config_reset_button.on_click
        def _(_: viser.GuiEvent) -> None:
            """Callback for when the reset button is clicked."""
            # reset configs to (task) defaults
            self.controller_config.set_override(self.task_name)
            self.optimizer_config.set_override(self.task_name)
            self.task_config = self.task_config.__class__()

            # reset gui elements
            for handle in self.gui_elements["controller_params"]:
                self.remove_handles(handle)
            for handle in self.gui_elements["optimizer_params"]:
                self.remove_handles(handle)
            for handle in self.gui_elements["task_params"]:
                self.remove_handles(handle)
            with self.gui_elements["controller_tab"]:
                self.gui_elements["controller_params"] = create_gui_elements(
                    self.server,
                    self.controller_config,
                    self.controller_config_updated,
                    self.controller_config_lock,
                )
            with self.gui_elements["optimizer_tab"]:
                self.gui_elements["optimizer_params"] = create_gui_elements(
                    self.server,
                    self.optimizer_config,
                    self.optimizer_config_updated,
                    self.optimizer_config_lock,
                )
            with self.gui_elements["task_tab"]:
                self.gui_elements["task_params"] = create_gui_elements(
                    self.server,
                    self.task_config,
                    self.task_config_updated,
                    self.task_config_lock,
                )

            # send the configs to the other nodes
            self.controller_config_updated.set()
            self.optimizer_config_updated.set()
            self.task_config_updated.set()

        # create a sim pause button if requested
        if self.sim_pause_button:
            sim_pause_button = self.server.gui.add_button("Pause Simulation")
            self.gui_elements["sim_pause_button"] = sim_pause_button

            @sim_pause_button.on_click
            def _(_: viser.GuiEvent) -> None:
                """Callback for when the pause button is clicked."""
                if sim_pause_button.label == "Pause Simulation":
                    sim_pause_button.label = "Resume Simulation"
                    self.sim_pause_updated.set()
                else:
                    sim_pause_button.label = "Pause Simulation"
                    self.sim_pause_updated.set()

        # create a display for plan time
        self.gui_elements["plan_time_display"] = self.server.gui.add_number(
            "plan time (ms)",
            initial_value=0.0,
            step=0.01,
            disabled=True,
        )

        # main tab containing parameter GUI elements
        tab_group = self.server.gui.add_tab_group()
        self.gui_elements["tab_group"] = tab_group

        controller_tab = tab_group.add_tab("Controller", viser.Icon.DEVICE_GAMEPAD)
        task_tab = tab_group.add_tab("Task", viser.Icon.CHECKLIST)
        optimizer_tab = tab_group.add_tab("Optimizer", viser.Icon.BINARY_TREE_2)

        # create controller GUI elements
        self.gui_elements["controller_tab"] = controller_tab
        with controller_tab:
            self.gui_elements["controller_params"] = create_gui_elements(
                self.server,
                self.controller_config,
                self.controller_config_updated,
                self.controller_config_lock,
            )

        # create the task GUI elements
        self.gui_elements["task_tab"] = task_tab
        with task_tab:
            self.gui_elements["task_params"] = create_gui_elements(
                self.server,
                self.task_config,
                self.task_config_updated,
                self.task_config_lock,
            )

        # create the optimizer GUI elements
        self.gui_elements["optimizer_tab"] = optimizer_tab
        with optimizer_tab:
            # create the rest of the optimizer GUI elements
            self.gui_elements["optimizer_params"] = create_gui_elements(
                self.server,
                self.optimizer_config,
                self.optimizer_config_updated,
                self.optimizer_config_lock,
            )

        # create task and optimizer callbacks after tabs are created
        @optimizer_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            """Callback for when the optimizer dropdown is updated."""
            # first, send the name of the new optimizer (set the name before the flag: ``on_event`` reads it at once)
            self.optimizer_name = optimizer_dropdown.value
            self.optimizer_updated.set()

            # update config
            optimizer_entry = self.available_optimizers.get(self.optimizer_name)
            assert optimizer_entry is not None
            optimizer_config_cls = optimizer_entry[1]
            self.optimizer_config = optimizer_config_cls()
            self.optimizer_config.set_override(self.task_name)
            self.optimizer_config_updated.set()

            # replace optimizer param gui elements
            for v in self.gui_elements["optimizer_params"]:
                v.remove()
            with optimizer_tab:
                self.gui_elements["optimizer_params"] = create_gui_elements(
                    self.server,
                    self.optimizer_config,
                    self.optimizer_config_updated,
                    self.optimizer_config_lock,
                )

            # because the optimizer will be updated, we need to sent the task parameters back to it so it doesn't start
            # with defaults
            self.task_config_updated.set()

        @task_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            """Callback for when the task dropdown is updated."""
            # first, send the name of the new task
            self.task_name = task_dropdown.value
            self.task_updated.set()

            # replace gui elements
            self._remove_gui_elements()

            # set up the entire visualizer from scratch
            with self.task_lock:
                self.set_task(self.task_name, self.optimizer_name)

    def remove_handles(self, handles: list[ElementType] | ElementType) -> None:
        """Remove GUI handles from the visualization node."""
        if isinstance(handles, ElementType):
            if not handles._impl.removed:
                handles.remove()
        else:
            assert isinstance(handles, list), "handles must be a list or a single handle."
            for handle in handles:
                if isinstance(handle, list):
                    self.remove_handles(handle)
                elif not handle._impl.removed:
                    handle.remove()

    def _remove_gui_elements(self) -> None:
        """Remove GUI elements from the visualization node."""
        for v in self.gui_elements.values():
            if isinstance(v, list):
                for handle in v:
                    self.remove_handles(handle)
            else:
                v.remove()
        self.viser_model.remove()

    def cleanup(self) -> None:
        """Cleanup the visualization node."""
        self._remove_gui_elements()
        self.server.flush()
        self.server.stop()
