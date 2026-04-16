# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo.controller.controller import (
    BatchedControllers,
    Controller,
    ControllerConfig,
    make_controller,
)
from judo.controller.overrides import (
    set_default_caltech_leap_cube_overrides,
    set_default_cartpole_overrides,
    set_default_cylinder_push_overrides,
    set_default_fr3_pick_overrides,
    set_default_leap_cube_down_overrides,
    set_default_leap_cube_overrides,
    set_default_spot_overrides,
)

set_default_caltech_leap_cube_overrides()
set_default_cartpole_overrides()
set_default_cylinder_push_overrides()
set_default_fr3_pick_overrides()
set_default_leap_cube_overrides()
set_default_leap_cube_down_overrides()
set_default_spot_overrides()

__all__ = [
    "BatchedControllers",
    "Controller",
    "ControllerConfig",
    "make_controller",
]
