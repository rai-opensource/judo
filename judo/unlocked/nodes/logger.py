# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
from typing import Any

from judo.unlocked.node import Node


class Logger:
    """Logger node"""

    def __init__(self, name: str, level: int, msg: str, logger: logging.Logger | None = None) -> None:
        self._level = level
        self._msg = msg
        self._logger = logger or logging.getLogger(name)

    def __call__(self, *args: Any) -> None:
        """Log."""
        self._logger.log(self._level, self._msg, *args)


def make_logger_node(name: str, level: int, msg: str, logger: logging.Logger | None = None) -> Node:
    return Node(name, Logger(name, level, msg, logger))
