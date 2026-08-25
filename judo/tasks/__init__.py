# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Dict, Type

from judo.tasks.base import Task, TaskConfig


@dataclass(frozen=True)
class TaskRegistration:
    """Complete registration metadata for a task."""

    task_type: Type[Task]
    task_config_type: Type[TaskConfig]
    rollout_backend: str = "mujoco"
    simulation_backend: str = "mujoco"
    locomotion_policy_path: str | None = None


# The registry is intentionally empty by default. Concrete tasks (and their backends /
# locomotion policies) are registered at the application level so that this library can be
# consumed by third-party packages without pulling in judo's built-in task set. Judo's own
# default tasks are registered in ``judo.app`` (see ``judo.app.register_default_tasks``).
_registered_tasks: Dict[str, TaskRegistration] = {}


def get_registered_tasks() -> Dict[str, TaskRegistration]:
    """Returns a dictionary of registered tasks."""
    return _registered_tasks


def get_task_registration(task_name: str) -> TaskRegistration:
    """Return full registration metadata for a task."""
    task_entry = _registered_tasks.get(task_name)
    if task_entry is None:
        raise ValueError(f"Task {task_name} not found in task registry.")
    return task_entry


def register_task(
    name: str,
    task_type: Type[Task],
    task_config_type: Type[TaskConfig],
    rollout_backend: str = "mujoco",
    simulation_backend: str = "mujoco",
    locomotion_policy_path: str | None = None,
) -> None:
    """Registers a new task and its default controller/simulation backends."""
    _registered_tasks[name] = TaskRegistration(
        task_type=task_type,
        task_config_type=task_config_type,
        rollout_backend=rollout_backend,
        simulation_backend=simulation_backend,
        locomotion_policy_path=locomotion_policy_path,
    )


__all__ = [
    "get_registered_tasks",
    "get_task_registration",
    "register_task",
    "TaskRegistration",
    "Task",
    "TaskConfig",
]
