# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from __future__ import annotations

import collections.abc
import typing

import numpy
import numpy.typing

__all__: list[str] = ["System", "create_systems_vector", "set_state", "threaded_rollout"]

class System:
    def __init__(self, model_filepath: str, policy_filepath: str) -> None: ...
    def get_control(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]: ...
    def get_state(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]: ...
    def load_policy(self, arg0: str, arg1: typing.Any) -> None: ...
    def policy_inference(self) -> None: ...
    def reset(self, arg0: bool) -> None: ...
    def rollout(
        self,
        state: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"],
        command: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"],
        physics_substeps: typing.SupportsInt | typing.SupportsIndex = 2,
        reset_last_output: bool = True,
        cutoff_time: typing.SupportsFloat | typing.SupportsIndex = ...,
    ) -> tuple[
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
    ]: ...
    def set_observation(
        self, command: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]
    ) -> None: ...
    @property
    def observation(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]: ...
    @observation.setter
    def observation(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]
    ) -> None: ...
    @property
    def policy_output(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]: ...
    @policy_output.setter
    def policy_output(
        self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]
    ) -> None: ...

def create_systems_vector(
    model: typing.Any, policy_filepath: str, num_systems: typing.SupportsInt | typing.SupportsIndex
) -> list[System]: ...
def set_state(
    system: System, state: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]
) -> None:
    """Sets the state of a system object"""

def threaded_rollout(
    systems: collections.abc.Sequence[System],
    states: numpy.typing.ArrayLike,
    command: numpy.typing.ArrayLike,
    last_policy_output: numpy.typing.ArrayLike,
    num_threads: typing.SupportsInt | typing.SupportsIndex,
    physics_substeps: typing.SupportsInt | typing.SupportsIndex,
    cutoff_time: typing.SupportsFloat | typing.SupportsIndex = ...,
) -> tuple[
    list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]],
    list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]],
    list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]],
]:
    """Threaded policy rollout with shared pointers to System objects."""
