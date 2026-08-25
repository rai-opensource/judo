# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Type stubs for the compiled ``policy_rollout_pybind`` extension module.

These stubs let type checkers resolve the C++ pybind11 module without requiring
the ``.so`` to be built (e.g. in lint-only CI jobs). Regenerate with::

    pixi run stubgen-extension
"""

from __future__ import annotations

from . import policy_rollout

__all__: list[str] = ["policy_rollout"]
