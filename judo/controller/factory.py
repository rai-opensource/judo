# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Registry-aware controller factory.

The :class:`~judo.controller.controller.Controller` class itself is intentionally decoupled from
the task/optimizer registries (it receives its backend and locomotion policy explicitly). This
module provides :func:`make_controller`, a convenience factory that *reads* the registries to
resolve those values and then constructs a controller. It depends only on the registry
*mechanism* (``judo.tasks`` / ``judo.optimizers``), not on any particular set of registered tasks,
so it can be used equally by judo's own app and by third-party applications that register their
own tasks.
"""

from typing import Any

from omegaconf import DictConfig

from judo.controller.controller import Controller, ControllerConfig
from judo.optimizers import get_registered_optimizers
from judo.registration import register_optimizers_from_cfg, register_tasks_from_cfg
from judo.tasks import get_registered_tasks, get_task_registration


def make_controller(
    init_task: str,
    init_optimizer: str,
    task_registration_cfg: DictConfig | None = None,
    optimizer_registration_cfg: DictConfig | None = None,
    controller_cls: type[Controller] | None = None,
    **controller_kwargs: Any,
) -> Controller:
    """Make a controller.

    This factory reads the task/optimizer registries to resolve backends and the locomotion policy
    path, then passes those to the (registry-agnostic) :class:`~judo.controller.Controller`
    constructor explicitly.

    Args:
        init_task: The task name to use.
        init_optimizer: The optimizer name to use.
        task_registration_cfg: Optional task registration overrides keyed by task name.
            Each entry must contain `task` and `config` import paths, and may also define
            `rollout_backend`, `simulation_backend`, and `locomotion_policy_path`.
            See register_tasks_from_cfg for the exact supported schema.
        optimizer_registration_cfg: Optional optimizer registration overrides keyed by
            optimizer name. Each entry must contain `optimizer` and `config` import paths.
            See register_optimizers_from_cfg for the exact supported schema.
        controller_cls: Optional controller class to instantiate instead of Controller.
        **controller_kwargs: Additional keyword arguments forwarded to the controller
            constructor.

    Returns:
        The created Controller instance.
    """
    if task_registration_cfg is not None:
        register_tasks_from_cfg(task_registration_cfg)
    if optimizer_registration_cfg is not None:
        register_optimizers_from_cfg(optimizer_registration_cfg)

    available_optimizers = get_registered_optimizers()
    available_tasks = get_registered_tasks()

    task_entry = available_tasks.get(init_task)
    optimizer_entry = available_optimizers.get(init_optimizer)

    assert task_entry is not None, f"Task {init_task} not found in task registry."
    assert optimizer_entry is not None, f"Optimizer {init_optimizer} not found in optimizer registry."
    task_registration = get_task_registration(init_task)

    # instantiate the task/optimizer/controller
    task = task_entry.task_type()

    optimizer_cls, optimizer_config_cls = optimizer_entry
    optimizer_cfg = optimizer_config_cls()
    optimizer_cfg.set_override(init_task)
    optimizer = optimizer_cls(optimizer_cfg, task.nu)

    controller_cfg = ControllerConfig()
    controller_cfg.set_override(init_task)

    # Resolve the locomotion policy path from the registry and pass it explicitly to the
    # controller so the Controller class itself stays decoupled from the task registry.
    rollout_backend_kwargs = dict(controller_kwargs.pop("rollout_backend_kwargs", None) or {})
    if task_registration.locomotion_policy_path is not None:
        rollout_backend_kwargs.setdefault("policy_path", task_registration.locomotion_policy_path)

    cls = controller_cls or Controller
    return cls(
        controller_config=controller_cfg,
        task=task,
        optimizer=optimizer,
        rollout_backend=task_registration.rollout_backend,
        rollout_backend_kwargs=rollout_backend_kwargs,
        **controller_kwargs,
    )
