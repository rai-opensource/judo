# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.unlocked.node import Node, NodeStop


class Once:
    """Once generator, a callable that returns ``True`` once and then raises ``NodeStop`` on subsequent calls."""

    def __init__(self) -> None:
        self._exec = False

    def __call__(self) -> bool:
        """Triggers once."""
        if self._exec:
            raise NodeStop("Done")
        else:
            self._exec = True
        return self._exec


def make_once_node(name: str) -> Node:
    return Node(name, Once())
