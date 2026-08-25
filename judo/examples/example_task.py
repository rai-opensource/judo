# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass

from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig


@dataclass
class MyCylinderPushConfig(CylinderPushConfig):
    """Dummy task config."""

    my_custom_param: int = 42


class MyCylinderPush(CylinderPush):
    """Dummy custom task."""
