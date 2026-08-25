# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Judo application defaults.

This package is judo's *batteries-included* task set. It is intentionally kept out of the
individual libraries (tasks, optimizers, controllers, simulation) so those libraries can be reused
by third-party packages that want to register their own tasks. Its sole job is to declare judo's
default tasks and how each one is wired (rollout backend, simulation backend, and low-level
locomotion policy).

Importing ``judo.app`` registers judo's built-in tasks via :func:`register_default_tasks`.
Third-party applications that only want the *mechanism* (not judo's default tasks) should avoid
importing this module and instead call :func:`judo.tasks.register_task` themselves. Applications
that want judo's defaults *and* their own can import this module (or call
:func:`register_default_tasks`) and then register/override additional tasks by name.
"""

from judo.tasks import register_task
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig
from judo.tasks.spot import (
    SpotBase,
    SpotBaseConfig,
    SpotBoxPush,
    SpotBoxPushConfig,
    SpotNavigate,
    SpotNavigateConfig,
    SpotTireRoll,
    SpotTireRollConfig,
    SpotTireUpright,
    SpotTireUprightConfig,
)
from judo.tasks.spot.spot_constants import SPOT_LOCOMOTION_POLICY_PATH


def register_default_tasks() -> None:
    """Register judo's built-in tasks with their default backends and locomotion policies.

    This is the application-level source of truth for which tasks are available and how each one
    is wired (rollout backend, simulation backend, and low-level locomotion policy). Third-party
    applications can call :func:`judo.tasks.register_task` (or
    :func:`judo.registration.register_tasks_from_cfg`) to register their own tasks instead of, or
    in addition to, these defaults. Re-registering a name overrides the previous entry, so callers
    may register these defaults and then override individual tasks by name.
    """
    register_task(CylinderPush.name, CylinderPush, CylinderPushConfig)
    register_task(Cartpole.name, Cartpole, CartpoleConfig)
    register_task(FR3Pick.name, FR3Pick, FR3PickConfig)
    register_task(LeapCube.name, LeapCube, LeapCubeConfig)
    register_task(LeapCubeDown.name, LeapCubeDown, LeapCubeDownConfig)
    register_task(CaltechLeapCube.name, CaltechLeapCube, CaltechLeapCubeConfig)

    spot_policy_path = str(SPOT_LOCOMOTION_POLICY_PATH)
    for spot_task, spot_config in (
        (SpotBase, SpotBaseConfig),
        (SpotBoxPush, SpotBoxPushConfig),
        (SpotNavigate, SpotNavigateConfig),
        (SpotTireRoll, SpotTireRollConfig),
        (SpotTireUpright, SpotTireUprightConfig),
    ):
        register_task(
            spot_task.name,
            spot_task,
            spot_config,
            rollout_backend="mujoco_hierarchical",
            simulation_backend="mujoco_hierarchical",
            locomotion_policy_path=spot_policy_path,
        )


# Register judo's built-in tasks when the application defaults are imported.
register_default_tasks()


__all__ = [
    "register_default_tasks",
]
