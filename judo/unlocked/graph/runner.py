# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
r"""Run a declarative :class:`GraphBuilder` directly (no app wrapper).

Usage:

    python -m judo.unlocked.graph.runner --builder=judo.unlocked.graph.example.loopback_graph:build

Loads the named ``module:callable`` (which must return a :class:`GraphBuilder`),
materializes it into live :class:`Node` objects, groups them by their declared
``process=`` tag into :class:`Threaded` schedulers, and blocks until SIGINT.

The builder callable may take keyword arguments; supply them with repeatable
``--builder-kwarg KEY=VALUE`` flags. This is how the different app graphs are
selected on the CLI, mirroring the options the React Flow editor exposes for
:func:`judo.app.graph.build`::

    python -m judo.unlocked.graph.runner \\
        --builder=judo.app.graph:build \\
        --builder-kwarg task=cartpole \\
        --builder-kwarg optimizer=mppi

``VALUE`` is parsed as JSON when possible (so ``true`` / ``false`` / numbers /
quoted strings work) and otherwise treated as a bare string (so ``backend=mujoco``
does the obvious thing).

Alternatively pass ``--graph-json=PATH`` to load a serialised graph (e.g. one
that was edited in the React Flow editor and saved). Either flag, not both.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from typing import Any

from judo.unlocked.logging_util import setup_logger
from judo.unlocked.graph.builder import GraphBuilder
from judo.unlocked.shm_session import cleanup_abandoned_sessions, shared_memory_session

_LOG = logging.getLogger(__name__)


def _cleanup_leftover_shm(g: GraphBuilder) -> None:
    """Reclaim abandoned runtime sessions, independent of this graph's node IDs."""
    cleanup_abandoned_sessions()


def _dry_run_report(g: GraphBuilder) -> int:
    """Print declarative grouping/order without importing targets or constructing runtime objects."""
    g._instance_order()  # noqa: SLF001 -- validate before any runtime construction
    spec = g.to_dict()
    # Reuse authoring-time port validation for serialized graphs, without materializing.
    validator = GraphBuilder.from_dict({**spec, "edges": []})
    for edge in spec["edges"]:
        validator.connect(edge["from"], edge["to"], policy=edge["policy"])
    groups: dict[str, list[dict[str, Any]]] = {}
    for node in spec["nodes"]:
        groups.setdefault(node["process"] or "main", []).append(node)
    print(f"graph {spec['name']!r}: {len(groups)} process(es)")
    for tier_idx, names in enumerate(g._shutdown_tiers()):  # noqa: SLF001
        grace = max((g._grace_for(n) for n in names), default=3.0)  # noqa: SLF001
        print(f"  shutdown tier {tier_idx} (grace {grace:.0f}s): {', '.join(names)}")
    for name, members in groups.items():
        member_names = [n["id"] for n in sorted(members, key=lambda n: n["close_order"])]
        print(f"  process {name!r} open/close order: {member_names}")
    return 0


def run_builder(g: GraphBuilder, *, shm_cleanup: bool = True, dry_run: bool = False) -> int:
    """Materialize + run an already-built :class:`GraphBuilder`.

    Shared by :func:`run_serialized_graph` and the ``//app:app`` entry point so both take the
    identical shm-cleanup / dry-run / run path. An unclean shutdown returns nonzero even
    if reporting hooks succeed; hooks still run so they can report failure diagnostics.
    """
    _LOG.info("running graph %r with %d nodes", g._name, len(g._nodes))  # noqa: SLF001
    if dry_run:
        return _dry_run_report(g)
    if shm_cleanup:
        _cleanup_leftover_shm(g)
    start = time.time()
    with shared_memory_session():
        g.run()
    hook_code = _run_post_run_hooks(g, start)
    return hook_code or (0 if bool(getattr(g, "shutdown_clean", False)) else 1)


def _run_post_run_hooks(g: GraphBuilder, start: float) -> int:
    """Invoke any ``metadata['post_run']`` hooks after the graph stops; return the worst exit code.

    Each hook is ``{"target": "module:qualname", "kwargs": {...}}`` (registered via
    :meth:`GraphBuilder.add_post_run_hook`). It is resolved by name here -- so the core layer stays
    decoupled from app-level graders -- and called as ``fn(since=start, **kwargs)``. A hook returning
    a truthy int sets the process exit code; a hook that raises is logged and ignored (a reporting
    hook must never mask the run itself).

    The reserved ``include_shutdown_status`` kwarg is consumed here. When true, the hook
    receives ``shutdown_clean`` from the graph (default False), overriding any supplied value.
    """
    hooks = (g._metadata or {}).get("post_run", [])  # noqa: SLF001
    exit_code = 0
    for hook in hooks:
        target = hook.get("target")
        if not target:
            continue
        try:
            kwargs = dict(hook.get("kwargs", {}))
            if kwargs.pop("include_shutdown_status", False):
                kwargs["shutdown_clean"] = bool(getattr(g, "shutdown_clean", False))
            mod_name, attr = target.split(":", 1)
            fn = getattr(importlib.import_module(mod_name), attr)
            code = fn(since=start, **kwargs)
            if isinstance(code, int) and code:
                exit_code = code
        except Exception:  # (a post-run reporting hook must not mask the run)
            _LOG.exception("post_run hook %r failed", target)
    return exit_code


def _load_from_module(spec: str, kwargs: dict[str, Any] | None = None) -> GraphBuilder:
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    g = fn(**(kwargs or {}))
    if not isinstance(g, GraphBuilder):
        raise TypeError(f"{spec} returned {type(g).__name__}, expected GraphBuilder")
    return g


def _parse_builder_kwarg(item: str) -> tuple[str, Any]:
    """Parse a ``KEY=VALUE`` builder kwarg; VALUE is JSON if possible, else a bare string."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"--builder-kwarg must be KEY=VALUE, got {item!r}")
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def _load_from_json(path: str) -> GraphBuilder:
    with open(path, encoding="utf-8") as f:
        return GraphBuilder.from_dict(json.load(f))


def _load_from_yaml(path: str) -> GraphBuilder:
    with open(path, encoding="utf-8") as f:
        return GraphBuilder.from_yaml(f.read())


def _kill_existing() -> int:
    """SIGKILL any other graph-``run`` processes, then return.

    Counterpart to the lifecycle hardening in :class:`Threaded` /
    :meth:`GraphBuilder.run`: if a previous invocation was force-killed and
    left scheduler children dangling (still bound to viser/foxglove ports,
    holding ``/dev/shm`` segments), call::

        python -m judo.unlocked.graph.runner --kill

    Skips the current pid (and its bazel-launcher parent) so we don't suicide.
    """
    import signal as _signal

    my_pid = os.getpid()
    parent_pid = os.getppid()
    killed = 0
    if not os.path.isdir("/proc"):
        _LOG.warning("/proc not available; cannot scan for graph runner processes")
        return 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in (my_pid, parent_pid):
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "graph/runner" not in cmdline and "graph.runner" not in cmdline:
            continue
        try:
            os.kill(pid, _signal.SIGKILL)
            killed += 1
            _LOG.info("killed pid=%d (%s)", pid, cmdline.strip()[:80])
        except OSError as exc:
            _LOG.warning("could not kill pid=%d: %s", pid, exc)
    _LOG.info("--kill: %d process(es) terminated", killed)
    # A worker may still be exiting; its inherited lease protects queues until it is gone.
    cleanup_abandoned_sessions()
    return 0


def run_serialized_graph(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--builder", help='"module:callable" returning a GraphBuilder')
    src.add_argument("--graph-json", help="Path to a serialized graph JSON file")
    src.add_argument("--graph-yaml", help="Path to a serialized graph YAML file")
    src.add_argument(
        "--kill",
        action="store_true",
        help="SIGKILL any other graph runner processes (and sweep /dev/shm) and exit.",
    )
    p.add_argument("--log-level", default="INFO", help="DEBUG / INFO / WARN / ERROR")
    p.add_argument(
        "--builder-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keyword argument forwarded to the --builder callable (repeatable). VALUE is parsed "
        "as JSON when possible, else a bare string. E.g. --builder-kwarg backend=mujoco "
        "--builder-kwarg use_mocap=true",
    )
    p.add_argument(
        "--no-shm-cleanup",
        action="store_true",
        help="Skip /dev/shm cleanup of leftover SharedMemoryQueue segments.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the graph and print the shutdown tiers + per-process open/close order, then exit "
        "without running.",
    )
    args = p.parse_args(argv)

    setup_logger(log_level=getattr(logging, args.log_level.upper()), include_thread=True)

    if args.kill:
        return _kill_existing()
    if not args.builder and not args.graph_json and not args.graph_yaml:
        p.error("one of --builder, --graph-json, --graph-yaml, or --kill is required")
    if args.builder_kwarg and not args.builder:
        p.error("--builder-kwarg requires --builder")

    builder_kwargs = dict(_parse_builder_kwarg(item) for item in args.builder_kwarg)
    if args.builder:
        g = _load_from_module(args.builder, builder_kwargs)
    elif args.graph_json:
        g = _load_from_json(args.graph_json)
    else:
        g = _load_from_yaml(args.graph_yaml)
    return run_builder(g, shm_cleanup=not args.no_shm_cleanup, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(run_serialized_graph())
