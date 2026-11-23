# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

import numpy as np
from scipy.interpolate import interp1d


class EventType(Enum):
    """Enum for event types."""

    START_SIMULATION = auto()
    PAUSE_SIMULATION = auto()
    START_CONTROLLER = auto()
    PAUSE_CONTROLLER = auto()
    CHANGE_TASK = auto()
    CHANGE_CONTROLLER = auto()


@dataclass
class JudoEvent:
    """Struct for judo events."""

    event: EventType
    value: str | None = None


@dataclass
class GuiState:
    """Snapshot of every GUI setting, published whole whenever any of them changes.

    A newer snapshot supersedes all older ones, so consumers read it on an ``optional`` edge and may safely miss
    intermediate snapshots. ``task``, ``optimizer`` and ``sim_paused`` are applied when they differ from the
    consumer's current value. Resets and configs are applied when their entry in ``revisions`` (bumped on every reset
    click / config edit) moves: a reset has no value to compare, and configs hold numpy arrays.

    Configs are dicts (``dataclasses.asdict``): their dataclasses are generated at runtime and cannot be pickled by
    reference. None means "not set" (the benchmarker only drives the task and optimizer).
    """

    task: str
    optimizer: str
    sim_paused: bool = False
    controller_config: dict[str, Any] | None = None
    optimizer_config: dict[str, Any] | None = None
    task_config: dict[str, Any] | None = None
    revisions: dict[str, int] = field(default_factory=dict)

    def moved(self, applied: dict[str, int]) -> set[str]:
        """Names in ``revisions`` that moved since the ``applied`` revisions."""
        return {name for name, rev in self.revisions.items() if rev != applied.get(name, 0)}


@dataclass
class MujocoState:
    """Struct for writing simulation states between different threads."""

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    sim_metadata: dict[str, Any]


KindType = Literal[
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "linear",
    "quadratic",
    "cubic",
    "previous",
    "next",
]


@dataclass
class SplineData:
    """Struct for (possibly batched) spline data."""

    t: np.ndarray
    """array of times for knot points, shape (T,)"""
    x: np.ndarray
    """(possibly batched) array of values to interpolate, shape (..., T, m)."""
    kind: KindType = "zero"
    """Spline type to use for interpolation. Same as parameter for scipy.interpolate.interp1d."""
    extrapolate: bool = True
    """Flag for whether to allow extrapolation queries. Default true (for re-initialization)."""

    def spline(self) -> interp1d:
        """Helper function for creating spline objects."""
        # fill values for "before" and "after" spline extrapolation.
        fill_value = (self.x[..., 0, :], self.x[..., -1, :])

        # TODO(pculbert): refactor to more modern spline utils (per scipy). https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#tutorial-interpolate-1dsection
        return interp1d(
            self.t,
            self.x,
            kind=self.kind,
            axis=-2,
            copy=False,
            fill_value=fill_value,  # type: ignore
            bounds_error=not self.extrapolate,
        )


@dataclass
class RenderPose:
    """Struct for visualization poses used by the renderer."""

    xpos: np.ndarray
    xquat: np.ndarray
