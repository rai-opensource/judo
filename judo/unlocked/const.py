# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from copy import copy, deepcopy
from functools import partial
from types import MethodType
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def is_builtin_class_instance(obj: Any) -> bool:
    return obj.__class__.__module__ == "builtins" and not isinstance(obj, (list, tuple, dict, set))


class Const(Generic[T]):
    """Turn any python object into a runtime constant object.

    This wrapper allows access to all member values, and non mutating member functions.
    Modifying member values or calling a modifying member funtion will raise RuntimeError.
    """

    def __init__(self, obj: T):
        self.__dict__["__obj"] = obj

    def __getattr__(self, name: str, /) -> Any:
        result = self.__dict__["__obj"].__getattribute__(name)
        if isinstance(result, MethodType):
            result = partial(result.__func__, self)
            return result
        return const(result)

    def __setattr__(self, name: str, value: Any, /) -> None:
        raise RuntimeError("Setting attribute to read-only object")

    def __eq__(self, obj: object) -> bool:
        return self.__dict__["__obj"] == const_cast(obj)

    def __ne__(self, obj: object) -> bool:
        return self.__dict__["__obj"] != const_cast(obj)

    def __ge__(self, obj: object) -> bool:
        return self.__dict__["__obj"] >= const_cast(obj)

    def __gt__(self, obj: object) -> bool:
        return self.__dict__["__obj"] > const_cast(obj)

    def __le__(self, obj: object) -> bool:
        return self.__dict__["__obj"] <= const_cast(obj)

    def __lt__(self, obj: object) -> bool:
        return self.__dict__["__obj"] < const_cast(obj)

    def __len__(self) -> int:
        return len(self.__dict__["__obj"])

    def __getitem__(self, index: Any) -> Any:
        return const(self.__dict__["__obj"][index])

    def __contains__(self, el: Any) -> bool:
        return el in self.__dict__["__obj"]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Const call method"""
        return self.__dict__["__obj"].__class__.__call__(self, *args, **kwargs)

    def __iter__(self) -> Any:
        for val in self.__dict__["__obj"]:
            yield const(val)

    def __str__(self) -> Any:
        return "const " + str(self.__dict__["__obj"])

    def __copy__(self) -> Any:
        return copy(self.__dict__["__obj"])

    def __deepcopy__(self, _: Any) -> Any:
        return deepcopy(self.__dict__["__obj"])


def const_cast(obj: Any) -> Any:
    """A convenience function to remove constant modifyer."""
    if isinstance(obj, Const):
        return obj.__dict__["__obj"]
    return obj


def const(obj: Any) -> Any:
    """A convenience function to make a constant object."""
    if is_builtin_class_instance(obj):
        return obj
    if isinstance(obj, Const):
        return obj
    return Const(obj)
