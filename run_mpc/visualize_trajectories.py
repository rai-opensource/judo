# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import json
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import tyro
from mujoco import MjData, mj_forward
from viser import GuiEvent, ViserServer

from judo.tasks import get_registered_tasks
from judo.visualizers.model import ViserMjModel


def decode_config(obj: dict) -> dict:
    """Custom JSON decoder to handle NumPy arrays and Path strings."""
    for key, value in obj.items():
        if isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            obj[key] = np.array(value)
        if isinstance(value, str) and ("/" in value or "\\" in value):
            obj[key] = Path(value)
    return obj


def visualize_trajectory_hdf5_dataset(dataset_path: Path) -> None:
    """Load and visualize trajectories from an HDF5 dataset."""
    with h5py.File(str(dataset_path), "r") as f:
        # Read task name from HDF5 attrs (preferred), with fallback to config file
        if "task" in f.attrs:
            task_name = f.attrs["task"]
        else:
            config_path = f.attrs["config_path"]
            with open(str(config_path), "r") as config_f:
                configs = json.load(config_f, object_hook=decode_config)
            task_name = configs["task"]

        qpos_dataset: h5py.Dataset = f["qpos"]  # type: ignore[assignment]
        trajectory_length_dataset: h5py.Dataset = f["trajectory_length"]  # type: ignore[assignment]
        goal_positions = np.array(f["goal_pos"]) if "goal_pos" in f else None

        visualize_trajectory_batch(str(task_name), qpos_dataset, trajectory_length_dataset, goal_positions)


def visualize_trajectory_batch(
    task: str,
    qpos_batch: np.ndarray | h5py.Dataset,
    trajectory_lengths: Optional[np.ndarray | h5py.Dataset] = None,
    goal_positions: Optional[np.ndarray] = None,
) -> None:
    """Visualize a batch of trajectories in viser."""
    server = ViserServer()

    registered_tasks = get_registered_tasks()
    task_entry = registered_tasks.get(task)
    assert task_entry is not None, f"Task {task} is not registered!"
    task_cls, _ = task_entry
    task_instance = task_cls()
    # Increase constraint buffers for contact-heavy scenes (e.g. Spot + tire).
    task_instance.spec.nconmax = 512
    task_instance.spec.njmax = 2048
    task_instance.model = task_instance.spec.compile()
    task_instance.data = MjData(task_instance.model)
    viser_mjmodel = ViserMjModel(server, task_instance.spec)

    # Render goal position marker if available
    goal_handle = None
    if goal_positions is not None:
        goal = goal_positions[0]
        goal_handle = server.scene.add_icosphere(
            "/goal_marker",
            radius=0.1,
            color=(0, 255, 0),
            position=(goal[0], goal[1], 0.0),
        )

    # Create GUI elements for controlling visualizer.
    running = False
    pause_button = server.gui.add_button("Start playback")

    def cycle_pause_button() -> None:
        nonlocal running
        if running:
            running = False
            pause_button.label = "Start playback"
        else:
            running = True
            pause_button.label = "Pause playback"

    @pause_button.on_click
    def cycle_pause_button_callback(_: GuiEvent) -> None:
        """Handle GUI callback."""
        cycle_pause_button()

    trajectory_slider = server.gui.add_slider(
        "Trajectory index", min=0, max=len(qpos_batch) - 1, step=1, initial_value=0
    )
    if trajectory_lengths:
        curr_trajectory_length = trajectory_lengths[trajectory_slider.value] - 1
    else:
        curr_trajectory_length = len(qpos_batch[trajectory_slider.value]) - 1
    timestep_slider = server.gui.add_slider(
        "Trajectory timestep", min=0, max=curr_trajectory_length, step=1, initial_value=0
    )

    # Initialize visualization with the first state
    task_instance.data.qpos[:] = qpos_batch[0, 0]
    mj_forward(task_instance.model, task_instance.data)
    viser_mjmodel.set_data(task_instance.data)

    @trajectory_slider.on_update
    def trajectory_slider_on_update_callback(_: GuiEvent) -> None:
        """Handle GUI callback."""
        # Reset timestep to zero for new trajectory.
        timestep_slider.value = 0
        # Change timestep bounds.
        if trajectory_lengths is not None:
            nonlocal curr_trajectory_length
            curr_trajectory_length = trajectory_lengths[trajectory_slider.value] - 1
            timestep_slider.max = curr_trajectory_length
        # Update goal marker position for this trajectory
        if goal_handle is not None and goal_positions is not None:
            goal = goal_positions[trajectory_slider.value]
            goal_handle.position = (goal[0], goal[1], 0.0)

    @timestep_slider.on_update
    def timestep_slider_on_update_callback(_: GuiEvent) -> None:
        """Handle GUI callback."""
        # Update the viser_mjmodel's data object
        qpos_value = qpos_batch[trajectory_slider.value, timestep_slider.value]
        task_instance.data.qpos[:] = qpos_value
        mj_forward(task_instance.model, task_instance.data)
        viser_mjmodel.set_data(task_instance.data)

    try:
        while True:
            if running:
                timestep_slider.value = min(curr_trajectory_length, timestep_slider.value + 1)
                if timestep_slider.value == curr_trajectory_length:
                    cycle_pause_button()

            time.sleep(task_instance.model.opt.timestep)
    except KeyboardInterrupt:
        print("Closing Judo...")


if __name__ == "__main__":
    tyro.cli(visualize_trajectory_hdf5_dataset)
