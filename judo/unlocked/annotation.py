# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from collections.abc import Iterator as IteratorI
from inspect import Parameter
from typing import Any, Callable, Mapping, TypeVar
from typing import Iterator as IteratorT

_F = TypeVar("_F", bound=Callable[..., Any])

#: Attribute name under which :func:`node_outputs` stashes a target's output
#: port names. Read by ``GraphBuilder`` when ``outputs=`` is omitted.
NODE_OUTPUTS_ATTR = "__node_outputs__"


def node_outputs(*names: str) -> Callable[[_F], _F]:
    """Annotate a node target with the names of the elements it returns.

    A node function that returns an N-tuple has no way to name those elements
    from its signature (``-> tuple[A, B]`` carries types, not labels). Decorate
    it so the declarative :class:`GraphBuilder` can infer ``outputs=`` instead of
    requiring it at every call site::

        @node_outputs("command", "arm_target_pos", "base_target_vel")
        def __call__(self, _tick, spot_state) -> tuple[Cmd, np.ndarray, np.ndarray]:
            ...
            return cmd, arm, base

    The return value is unchanged (a plain ``tuple``), so there is no runtime
    cost: the names live only in the ``__node_outputs__`` attribute.
    """

    def deco(fn: _F) -> _F:
        setattr(fn, NODE_OUTPUTS_ATTR, tuple(names))
        return fn

    return deco


def is_tuple_t(annotation: Any) -> bool:
    """Check if annotation is of tuple type."""
    return annotation is tuple or (hasattr(annotation, "__origin__") and annotation.__origin__ is tuple)


def is_iterator_t(annotation: Any) -> bool:
    """Check if annotation is of iterable type."""
    return (
        annotation is IteratorT
        or annotation is IteratorI
        or (hasattr(annotation, "__origin__") and annotation.__origin__ is IteratorI)
    )


def positional(parameters: Mapping, index: int) -> Parameter:
    name = list(parameters)[index]
    param = parameters[name]
    if param.kind >= Parameter.KEYWORD_ONLY:
        raise IndexError(f"Parameter {name} is not positional")
    return param
