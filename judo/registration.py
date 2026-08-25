# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import importlib

from omegaconf import DictConfig

from judo.optimizers import register_optimizer
from judo.tasks import register_task


def get_class_from_string(class_path: str) -> type:
    """Get a class from a string path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def register_tasks_from_cfg(task_registration_cfg: DictConfig) -> None:
    """Register custom tasks.

    Args:
        task_registration_cfg: Mapping keyed by task name. Each value must contain:
            `task`: import path to the Task class
            `config`: import path to the TaskConfig class

            Optional keys:
            `rollout_backend`: rollout backend registry key for controllers
            `simulation_backend`: simulation backend registry key for the simulation node
            `locomotion_policy_path`: path to a low-level policy used by hierarchical tasks

            Example schema:
            {
                "cylinder_push": {
                    "task": "judo.tasks.cylinder_push.CylinderPush",
                    "config": "judo.tasks.cylinder_push.CylinderPushConfig",
                    "rollout_backend": "mujoco",
                    "simulation_backend": "mujoco",
                }
            }
    """
    for task_name in task_registration_cfg.keys():
        task_dict = task_registration_cfg.get(task_name, {})
        allowed_keys = {"task", "config", "rollout_backend", "simulation_backend", "locomotion_policy_path"}
        assert set(task_dict.keys()).issubset(allowed_keys) and {"task", "config"}.issubset(task_dict.keys()), (
            "Task registration must include 'task' and 'config', and may optionally include "
            "'rollout_backend', 'simulation_backend', and 'locomotion_policy_path'."
        )
        assert isinstance(task_dict["task"], str), "Task must be a string path to the task class."
        assert isinstance(task_dict["config"], str), "Task config must be a string path to the config class."
        task_cls = get_class_from_string(task_dict["task"])
        task_config_cls = get_class_from_string(task_dict["config"])
        rollout_backend = task_dict.get("rollout_backend", "mujoco")
        simulation_backend = task_dict.get("simulation_backend", "mujoco")
        locomotion_policy_path = task_dict.get("locomotion_policy_path", None)
        assert isinstance(rollout_backend, str), "rollout_backend must be a string."
        assert isinstance(simulation_backend, str), "simulation_backend must be a string."
        assert locomotion_policy_path is None or isinstance(locomotion_policy_path, str), (
            "locomotion_policy_path must be a string if provided."
        )
        register_task(
            str(task_name),
            task_cls,
            task_config_cls,
            rollout_backend=rollout_backend,
            simulation_backend=simulation_backend,
            locomotion_policy_path=locomotion_policy_path,
        )


def register_optimizers_from_cfg(optimizer_registration_cfg: DictConfig) -> None:
    """Register custom optimizers.

    Args:
        optimizer_registration_cfg: Mapping keyed by optimizer name. Each value must contain:
            `optimizer`: import path to the Optimizer class
            `config`: import path to the OptimizerConfig class

            Example schema:
            {
                "cem": {
                    "optimizer": "judo.optimizers.cem.CrossEntropyMethod",
                    "config": "judo.optimizers.cem.CrossEntropyMethodConfig",
                }
            }
    """
    for optimizer_name in optimizer_registration_cfg.keys():
        optimizer_dict = optimizer_registration_cfg.get(optimizer_name, {})
        assert set(optimizer_dict.keys()) == {"optimizer", "config"}, (
            "Optimizer registration must be a dict with keys 'optimizer' and 'config'."
        )
        assert isinstance(optimizer_dict["optimizer"], str), "Optimizer must be a string path to the optimizer class."
        assert isinstance(optimizer_dict["config"], str), "Optimizer config must be a string path to the config class."
        optimizer_cls = get_class_from_string(optimizer_dict["optimizer"])
        optimizer_config_cls = get_class_from_string(optimizer_dict["config"])
        register_optimizer(str(optimizer_name), optimizer_cls, optimizer_config_cls)
