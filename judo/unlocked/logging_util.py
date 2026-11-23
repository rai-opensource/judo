# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Minimal logging setup for unlocked runtimes and examples."""

import logging
import sys

DEFAULT_FORMAT = "[%(levelname)s][%(asctime)s][%(processName)s][%(module)s]: %(message)s"
DEFAULT_THREAD_FORMAT = "[%(levelname)s][%(asctime)s][%(processName)s][%(threadName)s][%(module)s]: %(message)s"


def setup_logger(log_level: str | int = logging.INFO, include_thread: bool = False) -> None:
    """Sets the root logger. Run once per node/runtime."""
    logging.basicConfig(
        level=log_level,
        format=DEFAULT_THREAD_FORMAT if include_thread else DEFAULT_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
