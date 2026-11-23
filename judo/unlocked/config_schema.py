# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Describe a config's editable fields as a JSON UI schema (pure stdlib).

Shared by the ``module`` node classes -- which build rich, domain-aware schemas (e.g. the SMPC
optimizer / controller override maps, whose fields come from judo) -- and the export server's
``/instance_schema`` endpoint, which serves them to the React-Flow per-instance editor so kwargs are
edited as typed form controls instead of raw JSON text.

The ``kind`` taxonomy matches the builder-args schema the editor already renders
(``bool | int | float | str | enum | json``) plus ``group`` for a nested config map that carries its
own ``fields``. Kept in ``core`` with only-stdlib imports so any layer can use it without pulling
heavy deps.

A field entry is::

    {
      "name": str,
      "kind": "bool" | "int" | "float" | "str" | "enum" | "json" | "group",
      "default": <json>,           # value used when the field is left unset
      "value": <json>,             # current value (from the instance's kwargs)
      "choices": [<json>, ...],    # present for "enum"
      "fields": [<field>, ...],    # present for "group" (the nested sub-schema)
      "affects_schema": true,      # optional: editing it should re-request the schema
      "doc": str,                  # optional one-line help
    }
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from typing import Any, Mapping

KIND_BOOL = "bool"
KIND_INT = "int"
KIND_FLOAT = "float"
KIND_STR = "str"
KIND_ENUM = "enum"
KIND_JSON = "json"
KIND_GROUP = "group"

_UNION_ORIGINS = (typing.Union, getattr(types, "UnionType", ()))


def kind_of(annotation: Any) -> tuple[str, list[Any] | None]:
    """Map a type annotation to a coarse UI ``kind`` and optional ``enum`` choices.

    ``Literal[...]`` and ``enum.Enum`` subclasses become ``enum`` (with choices); ``X | None`` is
    classified by its non-``None`` branch; anything unrecognized falls back to ``json``.
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return KIND_ENUM, list(typing.get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return KIND_ENUM, [e.name for e in annotation]
    if annotation is bool:
        return KIND_BOOL, None
    if annotation is int:
        return KIND_INT, None
    if annotation is float:
        return KIND_FLOAT, None
    if annotation is str:
        return KIND_STR, None
    if origin in _UNION_ORIGINS:
        for arg in typing.get_args(annotation):
            if arg is type(None):
                continue
            kind, choices = kind_of(arg)
            if kind != KIND_JSON:
                return kind, choices
    return KIND_JSON, None


def jsonable(value: Any) -> Any:
    """Coerce a value into a JSON-serializable form (scalars/list/dict/enum/dataclass)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        inner = value.value
        return inner if isinstance(inner, (bool, int, float, str)) else value.name
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return None


def scalar_field(
    name: str,
    annotation: Any,
    default: Any,
    value: Any,
    *,
    choices: list[Any] | None = None,
    affects_schema: bool = False,
    app_level: bool = False,
    doc: str | None = None,
) -> dict[str, Any]:
    """Build one scalar/enum/json field entry, inferring ``kind`` from ``annotation``."""
    kind, ann_choices = kind_of(annotation)
    if choices is not None:
        kind = KIND_ENUM
    else:
        choices = ann_choices
    entry: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "default": jsonable(default),
        "value": jsonable(value),
    }
    if choices is not None:
        entry["choices"] = [jsonable(c) for c in choices]
    if affects_schema:
        entry["affects_schema"] = True
    if app_level:
        entry["app_level"] = True
    if doc:
        entry["doc"] = doc.strip().split("\n", 1)[0]
    return entry


def group_field(name: str, fields: list[dict[str, Any]], value: Any, *, doc: str | None = None) -> dict[str, Any]:
    """Build a nested ``group`` field carrying its own sub-``fields`` and current map ``value``."""
    entry: dict[str, Any] = {
        "name": name,
        "kind": KIND_GROUP,
        "fields": fields,
        "value": jsonable(value) or {},
    }
    if doc:
        entry["doc"] = doc.strip().split("\n", 1)[0]
    return entry


def field_default(field: dataclasses.Field) -> Any:
    """Resolve a dataclass field's default (via ``default`` or ``default_factory``), else ``None``."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return field.default_factory()
        except Exception:  # (best-effort default for the UI)
            return None
    return None


def _lookup(source: Any, name: str, fallback: Any) -> Any:
    """Read ``name`` from a Mapping or an object (attribute), falling back if absent."""
    if source is None:
        return fallback
    if isinstance(source, Mapping):
        return source.get(name, fallback)
    return getattr(source, name, fallback)


def schema_for_dataclass(
    dc_type: type,
    values: Mapping[str, Any] | None = None,
    *,
    defaults: Any = None,
    choices: Mapping[str, list[Any]] | None = None,
    affects_schema: set[str] | None = None,
    app_level: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generic: one scalar/json field per dataclass field, overlaying current ``values``.

    ``defaults`` optionally overrides the per-field default (a Mapping or an object with matching
    attributes) -- e.g. a judo config instance after ``set_override(task)`` so the form shows the
    *effective* baseline that runs when a field is left unset. ``choices`` / ``affects_schema`` layer
    per-field enum choices and the "re-request the schema on change" flag. ``app_level`` marks fields
    owned by the app-level Run-graph window (rendered read-only per-instance).
    """
    values = values or {}
    choices = choices or {}
    affects_schema = affects_schema or set()
    app_level = app_level or set()
    try:
        hints = typing.get_type_hints(dc_type)
    except Exception:  # (fall back to raw annotations)
        hints = {}
    out: list[dict[str, Any]] = []
    for field in dataclasses.fields(dc_type):
        annotation = hints.get(field.name, field.type)
        default = _lookup(defaults, field.name, field_default(field))
        value = values.get(field.name, default)
        out.append(
            scalar_field(
                field.name,
                annotation,
                default,
                value,
                choices=choices.get(field.name),
                affects_schema=field.name in affects_schema,
                app_level=field.name in app_level,
            )
        )
    return out
