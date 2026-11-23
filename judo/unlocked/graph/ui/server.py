# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Tiny stdlib HTTP server for the React Flow graph editor.

Run via:
    python -m judo.unlocked.graph.ui.server --builder judo.app.graph:build

Then open http://localhost:8770 in a browser.

The server:
  - serves static assets (index.html, app.js) from the bundled `ui/` directory,
  - on GET /graph.json, returns the in-memory graph (initialized from a Python builder),
  - on POST /graph.json, replaces the in-memory graph with the posted JSON
    and prints it (so users can copy/save it).

No third-party deps. React Flow is fetched from a CDN by the browser.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import inspect
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from judo.unlocked import config_schema as _cfgschema
from judo.unlocked.graph.builder import GraphBuilder

_LOG = logging.getLogger("unlocked.graph.ui.server")

_STATIC_DIR = Path(__file__).resolve().parent
_MIME = {".html": "text/html", ".js": "application/javascript", ".css": "text/css", ".json": "application/json"}


def _workspace_root() -> Path:
    """Best-effort workspace root for resolving ``module:qualname`` to source files.

    Walks up from this file until a ``pyproject.toml`` is found.
    """
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


_WORKSPACE = _workspace_root()


def _find_code_cli() -> str | None:
    """Locate the VS Code CLI (``code``) usable from this (possibly Remote-SSH) server process.

    In a Remote-SSH session the server runs on the remote host; VS Code injects a per-window IPC
    socket via ``VSCODE_IPC_HOOK_CLI`` and ships a remote CLI wrapper. Invoking that wrapper opens
    files in the *remote* window the user is actually looking at -- unlike a ``vscode://file`` URI,
    which the browser hands to the user's *local* VS Code.

    Prefers the remote-cli wrapper of the *current* window's server (VS Code exposes that server's
    path via its git-askpass helper env vars) so it matches the IPC socket exactly; else the newest
    bundled remote-cli wrapper; else whatever ``code`` is on PATH.
    """
    for var in ("VSCODE_GIT_ASKPASS_MAIN", "VSCODE_GIT_ASKPASS_NODE"):
        m = re.search(r"(.*/\.vscode-server/cli/servers/[^/]+/server)/", os.environ.get(var, ""))
        if m:
            cli = Path(m.group(1)) / "bin" / "remote-cli" / "code"
            if cli.exists():
                return str(cli)
    candidates = sorted(
        Path.home().glob(".vscode-server/cli/servers/*/server/bin/remote-cli/code"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which("code") or shutil.which("code-insiders")


def _open_in_editor(file: str, line: int) -> tuple[bool, str]:
    """Open ``file:line`` in the user's VS Code window via the remote-aware CLI.

    Confines the target to the workspace (the file paths come from an untrusted browser POST), then
    shells out to ``code --goto <file>:<line>``. The CLI uses ``VSCODE_IPC_HOOK_CLI`` to reach the
    connected window, so in a Remote-SSH session the file opens on the remote where it lives.
    Returns ``(ok, message)``; ``ok=False`` lets the client fall back to a ``vscode://file`` URI.
    """
    try:
        target = Path(file).resolve()
    except (OSError, ValueError) as e:
        _LOG.warning("open-in-editor: bad path %r: %s", file, e)
        return False, f"bad path: {e}"
    try:
        target.relative_to(_WORKSPACE)
    except ValueError:
        _LOG.warning("open-in-editor: %s is outside workspace %s", target, _WORKSPACE)
        return False, "path outside workspace"
    if not target.is_file():
        _LOG.warning("open-in-editor: file not found: %s", target)
        return False, "file not found"
    code = _find_code_cli()
    if not code:
        _LOG.warning("open-in-editor: no 'code' CLI found (not a VS Code session?)")
        return False, "code CLI not found"
    if not os.environ.get("VSCODE_IPC_HOOK_CLI"):
        _LOG.warning(
            "open-in-editor: VSCODE_IPC_HOOK_CLI is not set -- the server was not launched from a "
            "VS Code integrated terminal, so 'code' cannot reach your window. Restart the server "
            "from a VS Code terminal."
        )
    try:
        subprocess.run(
            [code, "--goto", f"{target}:{max(1, int(line))}"],
            env=os.environ.copy(),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        detail = getattr(e, "stderr", "") or ""
        _LOG.warning("open-in-editor: '%s' failed: %s %s", code, e, detail)
        return False, f"code CLI failed: {e}"
    _LOG.info("open-in-editor: opened %s:%s via %s", target, line, code)
    return True, str(target)


# Per-edge expected-frequency config (stable edge id ``src.outport->dst.inport`` -> expected Hz),
# persisted so it survives runs/restarts. The editor stays generic by co-locating the file with the
# ``--builder`` module rather than hardcoding an app path: ``judo.app.graph:build`` ->
# ``judo/app/edge_expected_hz.json``; the examples get their own beside their source.
def _edge_config_path() -> Path:
    """``edge_expected_hz.json`` next to the current builder's module (workspace root if unresolved)."""
    mod_name = _BUILDER_SPEC.split(":", 1)[0]
    src = _module_source_path(mod_name) if mod_name else None
    return (src.parent if src is not None else _WORKSPACE) / "edge_expected_hz.json"


def _load_edge_config() -> dict[str, float]:
    path = _edge_config_path()
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out: dict[str, float] = {}
                for k, v in data.items():
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if fv > 0:
                        out[str(k)] = fv
                return out
    except Exception:  # - a malformed config must not break the editor
        _LOG.exception("reading edge config %s failed", path)
    return {}


def _save_edge_config(cfg: dict[str, Any]) -> tuple[bool, str]:
    clean: dict[str, float] = {}
    for k, v in (cfg or {}).items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            clean[str(k)] = fv
    try:
        path = _edge_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(dict(sorted(clean.items())), indent=2) + "\n"
        path.write_text(body, encoding="utf-8")
        return True, str(path)
    except Exception as e:
        _LOG.exception("writing edge config %s failed", _edge_config_path())
        return False, str(e)


# The port this server is bound to (set in ``main``); used to build the child's
# tap/control URL so it never depends on how the browser reached us.
_SERVE_PORT = 8770

# Additional roots scanned for ``module/qualname`` source files. Populated from
# ``--search-root`` / ``UNLOCKED_SEARCH_ROOTS`` and (lazily) from bazel external
# pip ``site-packages`` directories.
_EXTRA_ROOTS: list[Path] = []
_AUTO_ROOTS_SCANNED = False


def _user_extra_roots() -> list[Path]:
    raw = os.environ.get("UNLOCKED_SEARCH_ROOTS", "")
    out: list[Path] = []
    for chunk in raw.split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            p = Path(chunk).expanduser()
            if p.is_dir():
                out.append(p)
    return out


def _auto_discover_roots() -> list[Path]:
    """Best-effort: pick up bazel external pip ``site-packages`` directories.

    These come from ``rules_python``'s pip repos and contain wheels like
    ``bosdyn-api`` whose ``_pb2`` modules are referenced from annotations.
    Cached after the first call.
    """
    global _AUTO_ROOTS_SCANNED
    if _AUTO_ROOTS_SCANNED:
        return []
    _AUTO_ROOTS_SCANNED = True
    import glob

    patterns = [
        # ``bazel-<workspace>`` is a symlink into the output base. ``external``
        # holds all third-party repos including pip ones.
        str(_WORKSPACE / f"bazel-{_WORKSPACE.name}" / "external" / "*" / "site-packages"),
        # Direct user-cache fallback. Multiple bazel hashes may exist; take all.
        str(Path.home() / ".cache" / "bazel" / "_bazel_*" / "*" / "external" / "*" / "site-packages"),
    ]
    seen: set[str] = set()
    out: list[Path] = []
    for pat in patterns:
        for p in glob.glob(pat):
            if p in seen:
                continue
            seen.add(p)
            out.append(Path(p))
    return out


def _module_source_path(mod_name: str) -> Path | None:
    """Locate a module's source file without executing it.

    Tries: workspace, configured ``_EXTRA_ROOTS``, ``importlib.util.find_spec``,
    then (lazily) autodiscovered bazel external pip ``site-packages``.
    """
    rel = Path(*mod_name.split("."))
    roots = [_WORKSPACE, *_EXTRA_ROOTS]
    for root in roots:
        for c in (root / f"{rel}.py", root / rel / "__init__.py"):
            if c.is_file():
                return c

    try:
        import importlib.util  # local import to keep startup cheap

        spec = importlib.util.find_spec(mod_name)
    except (ImportError, ValueError, ModuleNotFoundError):
        spec = None
    if spec is not None:
        origin = spec.origin
        if origin and origin != "built-in":
            p = Path(origin)
            if p.suffix == ".py" and p.is_file():
                return p
            if p.suffix == ".pyc":
                alt = p.with_suffix(".py")
                if alt.is_file():
                    return alt
            if p.is_file():
                return p
        else:
            for loc in getattr(spec, "submodule_search_locations", None) or []:
                init = Path(loc) / "__init__.py"
                if init.is_file():
                    return init

    # Lazy autodiscovery as a last resort.
    for root in _auto_discover_roots():
        for c in (root / f"{rel}.py", root / rel / "__init__.py"):
            if c.is_file():
                _EXTRA_ROOTS.append(root)  # promote so future lookups are fast.
                return c
    return None


def _resolve_target(target: str) -> tuple[Path, int] | None:
    """Resolve ``module:qualname`` to ``(absolute_path, 1-based-line)`` without importing the module.

    Returns ``None`` if the file can't be located.
    """
    if ":" not in target:
        return None
    mod_name, qualname = target.split(":", 1)
    src_path = _module_source_path(mod_name)
    if src_path is None:
        return None
    if not qualname:
        return src_path, 1
    line = _find_qualname_line(src_path.read_text(encoding="utf-8", errors="replace"), qualname)
    return src_path, line


def _find_qualname_line(text: str, qualname: str) -> int:
    """Walk ``qualname`` parts (e.g. ``Foo.bar``) looking for ``class`` / ``def`` definitions.

    Walks in source order and returns the 1-based line of the deepest match, or 1 if no part matched.
    """
    lines = text.splitlines()
    idx = 0
    found_line = 1
    for part in qualname.split("."):
        pat = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(part)}\b")
        for i in range(idx, len(lines)):
            if pat.search(lines[i]):
                found_line = i + 1
                idx = i + 1
                break
        else:
            break
    return found_line


def _find_class_node(text: str, qualname: str) -> ast.ClassDef | None:
    """Locate the ``ast.ClassDef`` matching ``qualname`` (dotted, e.g. ``Outer.Inner``)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    parts = qualname.split(".")
    scope: list[ast.AST] = [tree]
    found: ast.ClassDef | None = None
    for part in parts:
        nxt: ast.ClassDef | None = None
        for node in scope:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef) and child.name == part:
                    nxt = child
                    break
            if nxt is not None:
                break
        if nxt is None:
            return found  # partial match: caller may still want the outer class
        found = nxt
        scope = [nxt]
    return found


def _build_name_map(text: str, this_module: str) -> dict[str, str]:
    """Build a ``local_name -> "module:qualname"`` lookup from a module's source.

    Captures top-level ``from x import Y [as Z]`` and same-module class / function
    definitions so an annotation like ``Foxglove`` or ``Foo.Bar`` can be resolved
    back to a clickable target.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    name_map: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module:
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                name_map[local] = f"{stmt.module}:{alias.name}"
        elif isinstance(stmt, ast.Import):
            # ``import foo.bar [as b]``: map the local name to the module itself.
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".")[0]
                # Resolver expects a ``module:qualname`` form; use empty qualname.
                name_map.setdefault(local, f"{alias.name}:")
        elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name_map.setdefault(stmt.name, f"{this_module}:{stmt.name}")
    return name_map


def _annotation_refs(annotation: str | None, name_map: dict[str, str]) -> list[dict[str, str]]:
    """Extract resolvable ``module:qualname`` references from an annotation expression."""
    if not annotation:
        return []
    try:
        tree = ast.parse(annotation, mode="eval")
    except SyntaxError:
        return []
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    # Walk Name and Attribute (dotted) references in source order.
    for sub in ast.walk(tree):
        name: str | None = None
        target: str | None = None
        if isinstance(sub, ast.Name):
            name = sub.id
            target = name_map.get(name)
        elif isinstance(sub, ast.Attribute):
            # Walk leftmost ``Name`` and rebuild the dotted path.
            parts: list[str] = []
            cur: ast.AST = sub
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                parts.reverse()
                name = ".".join(parts)
                base = name_map.get(parts[0])
                if base:
                    # If the base was a ``from x import Y``: target = "x:Y.<rest>".
                    # If it was a bare ``import foo.bar``: target = "foo.bar:rest".
                    mod, qual = base.split(":", 1)
                    rest = parts[1:]
                    if qual:
                        new_qual = ".".join([qual, *rest])
                    else:
                        new_qual = ".".join(rest)
                    target = f"{mod}:{new_qual}"
        if name and target and name not in seen:
            seen.add(name)
            refs.append({"name": name, "target": target})
    return refs


def _infer_annotation(value: ast.AST | None) -> str | None:
    """Best-effort type inference for ``self.x = <value>`` when no annotation is present."""
    if value is None:
        return None
    if isinstance(value, ast.Call):
        return ast.unparse(value.func)
    if isinstance(value, ast.Constant):
        v = value.value
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "str"
        if isinstance(v, bytes):
            return "bytes"
        if v is None:
            return "None"
    if isinstance(value, ast.List):
        return "list"
    if isinstance(value, ast.Dict):
        return "dict"
    if isinstance(value, ast.Set):
        return "set"
    if isinstance(value, ast.Tuple):
        return "tuple"
    if isinstance(value, ast.ListComp):
        return "list"
    if isinstance(value, ast.DictComp):
        return "dict"
    if isinstance(value, ast.SetComp):
        return "set"
    if isinstance(value, ast.JoinedStr):
        return "str"
    return None


def _param_annotations(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Map parameter name -> annotation source for a function (including kw-only)."""
    out: dict[str, str] = {}
    args = func.args
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    for a in all_args:
        if a.annotation is not None:
            out[a.arg] = ast.unparse(a.annotation)
    if args.vararg and args.vararg.annotation is not None:
        out[args.vararg.arg] = ast.unparse(args.vararg.annotation)
    if args.kwarg and args.kwarg.annotation is not None:
        out[args.kwarg.arg] = ast.unparse(args.kwarg.annotation)
    return out


_DUNDERS_TO_SKIP = {"__init__"}


def _introspect_class(target: str) -> dict[str, Any] | None:
    """Return shared-state info for a class ``module:qualname``.

    Includes ``self.X`` attrs assigned in ``__init__`` plus public method names. ``None`` if not found.
    """
    resolved = _resolve_target(target)
    if resolved is None:
        return None
    src_path, _ = resolved
    text = src_path.read_text(encoding="utf-8", errors="replace")
    qualname = target.split(":", 1)[1]
    cls = _find_class_node(text, qualname)
    if cls is None:
        return None

    module_name = target.split(":", 1)[0]
    name_map = _build_name_map(text, module_name)
    attrs: list[dict[str, Any]] = []
    seen: set[str] = set()
    methods: list[dict[str, Any]] = []
    bases_raw = [ast.unparse(b) for b in cls.bases]
    bases = [{"text": b, "refs": _annotation_refs(b, name_map)} for b in bases_raw]

    def _push_attr(name: str, annotation: str | None, line: int, source: str) -> None:
        attrs.append(
            {
                "name": name,
                "annotation": annotation,
                "annotation_refs": _annotation_refs(annotation, name_map),
                "line": line,
                "source": source,
            }
        )

    # Class-level annotations / assignments (e.g. dataclass fields).
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            if name in seen:
                continue
            seen.add(name)
            attr_ann = ast.unparse(stmt.annotation) if stmt.annotation else _infer_annotation(stmt.value)
            _push_attr(name, attr_ann, stmt.lineno, "class")
        elif isinstance(stmt, ast.Assign):
            inferred = _infer_annotation(stmt.value)
            for assign_tgt in stmt.targets:
                if isinstance(assign_tgt, ast.Name) and assign_tgt.id not in seen:
                    seen.add(assign_tgt.id)
                    _push_attr(assign_tgt.id, inferred, stmt.lineno, "class")

    # Walk methods: collect ``self.X = ...`` (and annotated) assignments, plus
    # the method list (excluding ``__init__`` which is just construction noise).
    for stmt in cls.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if stmt.name not in _DUNDERS_TO_SKIP:
            methods.append({"name": stmt.name, "line": stmt.lineno})
        params = _param_annotations(stmt)
        for sub in ast.walk(stmt):
            tgts: list[ast.AST] = []
            ann: str | None = None
            value: ast.expr | None = None
            sub_lineno = 0
            if isinstance(sub, ast.Assign):
                tgts = list(sub.targets)
                value = sub.value
                sub_lineno = sub.lineno
            elif isinstance(sub, ast.AnnAssign):
                tgts = [sub.target]
                ann = ast.unparse(sub.annotation) if sub.annotation else None
                value = sub.value
                sub_lineno = sub.lineno
            for tgt in tgts:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                    and tgt.attr not in seen
                ):
                    seen.add(tgt.attr)
                    a = ann
                    if a is None:
                        # 1) ``self.x = some_param`` -> use param annotation.
                        if isinstance(value, ast.Name) and value.id in params:
                            a = params[value.id]
                        # 2) ``self.x = SomeClass(...)`` / literal -> infer.
                        if a is None:
                            a = _infer_annotation(value)
                    _push_attr(tgt.attr, a, sub_lineno, stmt.name)

    return {
        "target": target,
        "file": str(src_path),
        "line": cls.lineno,
        "class": cls.name,
        "bases": bases,
        "doc": ast.get_docstring(cls),
        "attrs": attrs,
        "methods": methods,
    }


def _find_function_node(text: str, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Locate the ``FunctionDef`` / ``AsyncFunctionDef`` for a dotted qualname."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    parts = qualname.split(".")
    scope: list[ast.AST] = [tree]
    found: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        nxt: ast.AST | None = None
        for node in scope:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == part:
                    nxt = child
                    break
                if not last and isinstance(child, ast.ClassDef) and child.name == part:
                    nxt = child
                    break
            if nxt is not None:
                break
        if nxt is None:
            return None
        if last:
            if isinstance(nxt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found = nxt
            return found
        scope = [nxt]
    return found


def _tuple_elements(annotation: ast.AST | None) -> list[ast.AST] | None:
    """If ``annotation`` is a ``tuple[A, B, C]`` / ``Tuple[A, B, C]``, return its elements."""
    if not isinstance(annotation, ast.Subscript):
        return None
    base = annotation.value
    if isinstance(base, ast.Name):
        if base.id not in {"tuple", "Tuple"}:
            return None
    elif isinstance(base, ast.Attribute):
        if base.attr != "Tuple":
            return None
    else:
        return None
    sl = annotation.slice
    if isinstance(sl, ast.Tuple):
        return list(sl.elts)
    return [sl]


def _resolve_port_type(target: str, side: str, index: int) -> dict[str, Any] | None:
    """Return annotation info for a node port.

    ``target`` is ``module:qualname`` of the node's callable. ``side`` is
    ``"in"`` (parameter at ``index``, skipping ``self``) or ``"out"``
    (i-th element of the return tuple annotation, or the whole return if
    not a tuple). Returns ``{annotation, refs, name?}`` or ``None``.
    """
    if ":" not in target:
        return None
    mod_name, qualname = target.split(":", 1)
    resolved = _resolve_target(target)
    if resolved is None:
        return None
    src_path, _line = resolved
    text = src_path.read_text(encoding="utf-8", errors="replace")
    func = _find_function_node(text, qualname)
    if func is None:
        return None
    name_map = _build_name_map(text, mod_name)

    if side == "in":
        args = func.args
        ordered: list[ast.arg] = list(args.posonlyargs) + list(args.args)
        # Drop leading ``self`` / ``cls`` for methods.
        if ordered and ordered[0].arg in {"self", "cls"}:
            ordered = ordered[1:]
        ordered += list(args.kwonlyargs)
        if 0 <= index < len(ordered):
            arg = ordered[index]
            ann = ast.unparse(arg.annotation) if arg.annotation else None
            return {
                "annotation": ann,
                "refs": _annotation_refs(ann, name_map),
                "name": arg.arg,
                "kind": "param",
            }
        # Beyond positional/kw-only: must be a vararg.
        if args.vararg is not None:
            arg = args.vararg
            ann = ast.unparse(arg.annotation) if arg.annotation else None
            return {
                "annotation": ann,
                "refs": _annotation_refs(ann, name_map),
                "name": f"*{arg.arg}",
                "kind": "vararg",
            }
        return None

    if side == "out":
        ret = func.returns
        if ret is None:
            return None
        elts = _tuple_elements(ret)
        # Variadic homogeneous tuple: ``tuple[X, ...]``. Every output port has
        # type ``X``; the literal Ellipsis is not a real element.
        if elts is not None and len(elts) == 2 and isinstance(elts[1], ast.Constant) and elts[1].value is Ellipsis:
            ann = ast.unparse(elts[0])
        elif elts is not None and 0 <= index < len(elts):
            ann = ast.unparse(elts[index])
        elif elts is None:
            ann = ast.unparse(ret)
        else:
            return None
        return {
            "annotation": ann,
            "refs": _annotation_refs(ann, name_map),
            "kind": "return",
        }

    return None


def _load_builder(spec: str, kwargs: dict[str, Any] | None = None) -> GraphBuilder:
    """Load a builder by ``module:callable`` reference; the callable must return a GraphBuilder.

    ``kwargs`` are forwarded to the builder. Module caches are cleared so the
    builder picks up edits made in the source file between calls.
    """
    mod_name, attr = spec.split(":")
    # Reload so a user editing the builder source sees fresh kwargs/wiring on
    # the next /builder/build, not whatever was imported at server start.
    if mod_name in sys.modules:
        mod = importlib.reload(sys.modules[mod_name])
    else:
        mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    g = fn(**(kwargs or {}))
    if not isinstance(g, GraphBuilder):
        raise TypeError(f"{spec} returned {type(g).__name__}, expected GraphBuilder")
    return g


def _builder_params(spec: str) -> list[dict[str, Any]]:
    """Introspect the builder's signature into a JSON-serializable param list.

    Each entry is ``{name, kind, default, annotation, choices?, required, nullable}``;
    ``kind`` is a coarse UI hint (``bool|int|float|str|enum|json``),
    ``choices`` is populated when the annotation is a ``Literal[...]``, and
    ``nullable`` marks ``X | None`` annotations (a nullable bool gets an unset/Auto state).
    Skips ``*args`` / ``**kwargs`` since the UI can't render them generically.
    """
    import types
    import typing

    mod_name, attr = spec.split(":")
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=False)
    except Exception:  # (best-effort; fall back to raw annotations)
        hints = {}

    # Optional per-param choices override from the builder module/function.
    # The builder can attach a ``param_choices`` callable or dict returning
    # ``{param_name: [choice, ...]}`` to populate dropdowns dynamically
    # without requiring a Literal annotation (useful when choices come from
    # a heavy import we want to keep lazy).
    extra_choices: dict[str, list[Any]] = {}
    pc = getattr(fn, "param_choices", None) or getattr(mod, "param_choices", None)
    try:
        if callable(pc):
            pc = pc()
        if isinstance(pc, dict):
            extra_choices = {str(k): list(v) for k, v in pc.items()}
    except Exception:  # (best-effort; ignore broken hooks)
        extra_choices = {}

    def _ser_default(v: Any) -> Any:
        # JSON-friendly: pass through scalars/list/dict; stringify the rest.
        if v is inspect.Parameter.empty:
            return None
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)):
            return [_ser_default(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _ser_default(x) for k, x in v.items()}
        return repr(v)

    _union_origins = (typing.Union, types.UnionType)

    out: list[dict[str, Any]] = []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(name, p.annotation)
        kind = "json"
        choices: list[Any] | None = None
        origin = typing.get_origin(ann)
        if origin is typing.Literal:
            kind = "enum"
            choices = list(typing.get_args(ann))
        elif ann is bool:
            kind = "bool"
        elif ann is int:
            kind = "int"
        elif ann is float:
            kind = "float"
        elif ann is str:
            kind = "str"
        elif origin in _union_origins:
            # `X | None` / `Optional[X]`: classify by the non-None branch.
            for a in typing.get_args(ann):
                if a is type(None):
                    continue
                if a is bool:
                    kind = "bool"
                    break
                if a is int:
                    kind = "int"
                    break
                if a is float:
                    kind = "float"
                    break
                if a is str:
                    kind = "str"
                    break
        if name in extra_choices:
            choices = extra_choices[name]
            kind = "enum"
        try:
            ann_str = ann if isinstance(ann, str) else (getattr(ann, "__name__", None) or str(ann))
        except Exception:
            ann_str = ""
        entry: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "default": _ser_default(p.default),
            "annotation": ann_str,
            "required": p.default is inspect.Parameter.empty,
            "nullable": origin in _union_origins and type(None) in typing.get_args(ann),
        }
        if choices is not None:
            entry["choices"] = [_ser_default(c) for c in choices]
        out.append(entry)
    return out


def _resolve_object(target: str) -> Any:
    """Import + attribute-walk ``module:qualname`` to the live object (class / function / method).

    Unlike :func:`_resolve_target` (which stays source-only for go-to-def), this returns the actual
    runtime object so we can introspect dataclass defaults, judo registries and signatures. Safe
    here because the server already imports the builder module (and thus every node class).
    """
    if ":" not in target:
        return None
    mod_name, qualname = target.split(":", 1)
    try:
        mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    except Exception:  # (missing / broken module -> no schema)
        return None
    obj: Any = mod
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _signature_schema(fn: Any) -> dict[str, dict[str, Any]]:
    """Introspect a callable's params into ``{name: {kind, default, choices?}}`` (skips *args/**kwargs)."""
    import typing

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    hint_src = fn.__init__ if inspect.isclass(fn) else fn
    try:
        hints = typing.get_type_hints(hint_src)
    except Exception:  # (best-effort; fall back to raw annotations)
        hints = {}
    out: dict[str, dict[str, Any]] = {}
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in ("self", "cls"):
            continue
        ann = hints.get(name, p.annotation)
        if ann is inspect.Parameter.empty:
            kind, choices = _cfgschema.KIND_JSON, None
        else:
            kind, choices = _cfgschema.kind_of(ann)
        entry: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "default": None if p.default is inspect.Parameter.empty else _cfgschema.jsonable(p.default),
        }
        if choices is not None:
            entry["choices"] = [_cfgschema.jsonable(c) for c in choices]
        out[name] = entry
    return out


def _infer_field(name: str, value: Any) -> dict[str, Any]:
    """Best-effort field descriptor for a kwarg with no type info, from its current value's JSON type."""
    if isinstance(value, bool):
        kind = _cfgschema.KIND_BOOL
    elif isinstance(value, int):
        kind = _cfgschema.KIND_INT
    elif isinstance(value, float):
        kind = _cfgschema.KIND_FLOAT
    elif isinstance(value, str):
        kind = _cfgschema.KIND_STR
    else:
        kind = _cfgschema.KIND_JSON
    return {"name": name, "kind": kind, "default": _cfgschema.jsonable(value)}


def _resolve_instance_fields(factory: str, current: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort typed schema for an instance's ctor kwargs, for the per-instance UI editor.

    Resolution order:
      1. the node class's own ``config_schema(current)`` -- rich + domain-aware (e.g. SMPC's nested
         judo controller / optimizer groups + optimizer dropdown);
      2. a generic dataclass schema from a declared ``CONFIG_TYPE``;
      3. the factory / class signature, enriched with the builder's own param schema (so a
         ``make_*(**config)`` factory still yields typed dropdowns for the shared config fields),
         with value-inference for anything left.
    """
    current = dict(current or {})
    obj = _resolve_object(factory)
    owner: Any = None
    if inspect.isclass(obj):
        owner = obj
    elif inspect.ismethod(obj) and inspect.isclass(getattr(obj, "__self__", None)):
        owner = obj.__self__

    # 1) Rich, domain-aware schema provided by the node class.
    if owner is not None and callable(getattr(owner, "config_schema", None)):
        try:
            return list(owner.config_schema(current))
        except Exception:  # (fall back to the generic paths below)
            _LOG.exception("config_schema(%s) failed; falling back to generic introspection", factory)

    # 2) Generic dataclass schema from a declared CONFIG_TYPE.
    cfg_type = getattr(owner, "CONFIG_TYPE", None) if owner is not None else None
    if dataclasses.is_dataclass(cfg_type) and isinstance(cfg_type, type):
        return _cfgschema.schema_for_dataclass(cfg_type, current)

    # 3) Signature of the factory / class, enriched with the builder's param schema.
    sig_params = _signature_schema(obj) if obj is not None else {}
    builder_params: dict[str, dict[str, Any]] = {}
    if _BUILDER_SPEC:
        try:
            builder_params = {p["name"]: p for p in _builder_params(_BUILDER_SPEC)}
        except Exception:  # (builder introspection is optional here)
            builder_params = {}
    names: list[str] = list(sig_params)
    for key in current:
        if key not in names:
            names.append(key)
    out: list[dict[str, Any]] = []
    for name in names:
        base = sig_params.get(name) or builder_params.get(name)
        if base is not None:
            entry: dict[str, Any] = {
                "name": name,
                "kind": base.get("kind", _cfgschema.KIND_JSON),
                "default": base.get("default"),
            }
            if base.get("choices") is not None:
                entry["choices"] = base["choices"]
        else:
            entry = _infer_field(name, current.get(name))
        entry["value"] = _cfgschema.jsonable(current.get(name, entry.get("default")))
        out.append(entry)
    return out


def _flag_app_level(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark instance fields owned by the app-level Run-graph window (the builder params) read-only.

    An instance kwarg that mirrors a builder parameter (``task`` / ``robot`` / ``connection`` /
    ``use_mocap`` / ...) is propagated from the single app-level config on every rebuild, so it must
    not be edited per-instance (that would let two instances disagree on e.g. the task). The UI
    renders ``app_level`` fields disabled. A flag a node already set on its own schema (e.g. SMPC's
    aliases ``task_name`` / ``spot_id`` / ``requires_mocap``) is preserved.
    """
    app_names: set[str] = set()
    if _BUILDER_SPEC:
        try:
            app_names = {p["name"] for p in _builder_params(_BUILDER_SPEC)}
        except Exception:  # (builder introspection optional here)
            app_names = set()
    for f in fields:
        if not f.get("app_level") and f.get("name") in app_names:
            f["app_level"] = True
    return fields


def _instance_schema(factory: str, current: dict[str, Any]) -> list[dict[str, Any]]:
    """Typed per-instance schema (see :func:`_resolve_instance_fields`) with app-level fields flagged."""
    # References are builder-owned wiring, not editable object/config fields.
    refs = {k: v for k, v in current.items() if isinstance(v, dict) and "_instance_ref_" in v}
    fields = _resolve_instance_fields(factory, {k: v for k, v in current.items() if k not in refs})
    fields = [f for f in fields if f["name"] not in refs]
    fields.extend(
        {"name": k, "kind": _cfgschema.KIND_JSON, "value": v, "default": v, "app_level": True} for k, v in refs.items()
    )
    return _flag_app_level(fields)


def _builder_meta(spec: str) -> dict[str, Any]:
    """Optional free-form UI metadata from a builder ``param_meta()`` hook (e.g. the vision-task set)."""
    mod_name, attr = spec.split(":")
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    fn = getattr(mod, attr, None)
    pm = getattr(fn, "param_meta", None) or getattr(mod, "param_meta", None)
    try:
        if callable(pm):
            pm = pm()
        return dict(pm) if isinstance(pm, dict) else {}
    except Exception:  # (best-effort; ignore broken hooks)
        return {}


def _has_target(obj: Any) -> bool:
    """True for build-owned Hydra sub-configs or instance references (recursively)."""
    if isinstance(obj, dict):
        return "_target_" in obj or "_instance_ref_" in obj or any(_has_target(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_target(v) for v in obj)
    return False


def _merge_instance_edits(new_spec: dict[str, Any], old_json: str, built_json: str) -> None:
    """Carry per-instance *tunable* edits across a builder rebuild (mutates ``new_spec`` in place).

    A rebuild regenerates every instance from the app-level config so app-level fields (task / robot
    / ...) re-propagate. Without this it would also wipe genuine per-instance tuning (e.g. SMPC
    optimizer / controller overrides, the recorder ``log_path``). For each instance present in both
    the old and new spec, restore the old value of every kwarg the user edited (it differs from
    ``built_json``, the previous build's own output) that is NOT ``app_level`` in the instance's
    schema -- so tuning survives while app-level and build-derived values (e.g. a mocap id derived
    from ``robot``) take the fresh build.
    """
    try:
        old = json.loads(old_json) if old_json else {}
        built = json.loads(built_json) if built_json else {}
    except json.JSONDecodeError:
        return
    old_insts = {i.get("id"): i for i in old.get("instances", [])}
    built_insts = {i.get("id"): i for i in built.get("instances", [])}
    for inst in new_spec.get("instances", []):
        old_inst = old_insts.get(inst.get("id"))
        if not old_inst or old_inst.get("factory") != inst.get("factory"):
            continue
        new_kwargs = inst.get("kwargs") or {}
        old_kwargs = old_inst.get("kwargs") or {}
        built_kwargs = (built_insts.get(inst.get("id")) or {}).get("kwargs") or {}
        if not new_kwargs or not old_kwargs:
            continue
        try:
            app_level = {f["name"] for f in _instance_schema(inst["factory"], new_kwargs) if f.get("app_level")}
        except Exception:  # (schema optional; keep the fresh build for this instance)
            continue
        for k, v in old_kwargs.items():
            if k not in new_kwargs or k in app_level or (k in built_kwargs and built_kwargs[k] == v):
                continue
            # A Hydra ``_target_`` sub-config (e.g. ``spot_control``'s ``config`` / ``robot_provider``)
            # is regenerated by ``build()`` from the app-level config -- it carries app fields like
            # task / robot -- so it must take the FRESH build value, never a stale restore. Plain-map
            # tunables (e.g. SMPC controller / optimizer overrides) have no ``_target_`` and still merge.
            if _has_target(v) or _has_target(new_kwargs.get(k)):
                continue
            new_kwargs[k] = v
        inst["kwargs"] = new_kwargs


# Builder state shared across requests. Set by ``main`` and rewritten on
# ``POST /builder/build`` so the UI can rebuild the graph from kwargs.
_BUILDER_SPEC: str = ""
_BUILDER_KWARGS: dict[str, Any] = {}

# --- persistent editor state ------------------------------------------------
# Survives server restarts: the last builder config (so a subsequent run starts
# where the user left off) and the skill policy last chosen for each vision task
# (so re-selecting that task refills its skill). Stored under the user's state
# dir -- never in the repo (would dirty git) or the bazel runfiles (read-only /
# ephemeral). Best-effort throughout: a missing / unreadable file just means "no
# memory yet", and a failed write is logged but never blocks a build.
_PERSIST: dict[str, Any] = {"kwargs": {}, "skill_policies": {}}


def _state_path() -> Path:
    """Path to the persisted editor-state JSON (honors ``$XDG_STATE_HOME``)."""
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "state"
    return root / "starfish" / "unlocked_editor.json"


def _builder_key() -> str:
    """Namespace key for persisted state: the active builder spec.

    Distinct builders describe distinct graphs with distinct parameters, so each gets its OWN cache
    entry -- switching builders never feeds one graph's kwargs to another.
    """
    return _BUILDER_SPEC or "default"


def _load_all_persisted() -> dict[str, Any]:
    """Read the whole persisted-state file ({} on any error)."""
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_persisted_state() -> None:
    """Load THIS builder's persisted config + per-task skill policies and seed ``_BUILDER_KWARGS``.

    Called once at startup (after ``_BUILDER_SPEC`` is set). State is namespaced per builder (keyed
    by ``module:callable``), so distinct graphs never share a cache. A legacy flat file (pre
    namespacing) is adopted for the current builder only if it still builds; the next save migrates
    it. Drops any persisted kwarg that is no longer a builder parameter, and finally any whole set
    that fails to build, so a stale key/value can't crash ``build(**kwargs)`` on the first rebuild.
    """
    global _BUILDER_KWARGS
    data = _load_all_persisted()
    if not data:
        return
    builders = data.get("builders")
    if isinstance(builders, dict):
        entry = builders.get(_builder_key())
        entry = entry if isinstance(entry, dict) else {}
    else:
        # Legacy flat state (one shared cache, pre per-builder): adopt it for THIS builder; the next
        # save migrates it into the namespaced format.
        entry = data
    _PERSIST["kwargs"] = dict(entry.get("kwargs") or {})
    _PERSIST["skill_policies"] = dict(entry.get("skill_policies") or {})
    if _PERSIST["kwargs"] and _BUILDER_SPEC:
        try:
            valid = {p["name"] for p in _builder_params(_BUILDER_SPEC)}
            _PERSIST["kwargs"] = {k: v for k, v in _PERSIST["kwargs"].items() if k in valid}
        except Exception:  # (best-effort filter; keep what we have)
            pass
        # Name-filtering can't catch a VALUE this builder rejects (e.g. a legacy flat file carrying a
        # single-robot task fed to the multi-robot builder, which requires a 2-robot task). Verify
        # the kwargs actually build; if not, drop them (fall back to this builder's defaults) with a
        # concise note rather than letting ``main`` surface a startup traceback.
        if _PERSIST["kwargs"]:
            try:
                _load_builder(_BUILDER_SPEC, _PERSIST["kwargs"])
            except Exception as exc:
                _LOG.warning(
                    "discarding persisted editor kwargs %s: incompatible with builder %s (%s)",
                    _PERSIST["kwargs"],
                    _BUILDER_SPEC,
                    exc,
                )
                _PERSIST["kwargs"] = {}
    if _PERSIST["kwargs"]:
        _BUILDER_KWARGS = dict(_PERSIST["kwargs"])
    _LOG.info("restored editor state (%s) from %s", _builder_key(), _state_path())


def _save_persisted_state() -> None:
    """Atomically write THIS builder's ``_PERSIST`` entry into the namespaced state file (best-effort).

    Reads the whole file first and rewrites only the current builder's entry, so other builders'
    saved configs are preserved.
    """
    path = _state_path()
    try:
        data = _load_all_persisted()
        builders = data.get("builders")
        builders = builders if isinstance(builders, dict) else {}
        builders[_builder_key()] = {
            "kwargs": _PERSIST.get("kwargs", {}),
            "skill_policies": _PERSIST.get("skill_policies", {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"builders": builders}, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        _LOG.exception("failed to persist editor state to %s", path)


def _remember_config(kwargs: dict[str, Any]) -> None:
    """Persist the current config and, per task, its chosen skill policy.

    A non-empty ``skill_policy`` only occurs for a vision task -- ``build()``
    forces it empty for state-based (SMPC) tasks -- so we can key the skill by
    task without consulting the vision-task set here (which keeps this robust
    even if the task registry isn't importable).
    """
    _PERSIST["kwargs"] = dict(kwargs)
    task = kwargs.get("task")
    skill = kwargs.get("skill_policy")
    if task and skill:
        _PERSIST["skill_policies"][str(task)] = skill
    _save_persisted_state()


class _Runner:
    """Manages a single child app process and a rolling log buffer.

    Logs are appended with monotonically increasing ids so clients can poll
    ``GET /run/status?offset=N`` and receive only new lines.
    """

    # Console log retention. ``None`` keeps every line (unbounded) so the full
    # run log stays scrollable / downloadable no matter how long the run is.
    MAX_LINES: int | None = None
    # How long to wait for the child to finish its visible safe-shutdown
    # animation (mujoco sit-down) before falling through to SIGINT. Tuned
    # for ~1.5 s sit-down + a half-second of slack.
    _SAFE_SHUTDOWN_GRACE_S = 3.0

    def __init__(self) -> None:
        # Reentrant so a helper called inside a locked region can still take
        # the lock (e.g. ``_append`` from inside ``_drain``'s exit handler).
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lines: deque[tuple[int, str]] = deque(maxlen=self.MAX_LINES)
        self._next_id = 0
        self._exit_code: int | None = None
        self._cmd: list[str] = []
        self._started_at: float | None = None

    # ---- internal helpers ----------------------------------------------------
    def _append(self, line: str) -> None:
        with self._lock:
            self._lines.append((self._next_id, line.rstrip("\n")))
            self._next_id += 1

    def _drain(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self._append(line)
        proc.wait()
        with self._lock:
            # Only record this exit on the currently-tracked process so an old
            # drain finishing after a new start does not clobber the new run.
            if self._proc is proc:
                self._exit_code = proc.returncode
        # ``_append`` re-acquires ``self._lock`` (non-reentrant); call it
        # outside the locked region above or the runner deadlocks and every
        # subsequent ``status()`` / ``start()`` blocks forever.
        self._append(f"[runner] process {proc.pid} exited with code {proc.returncode}")

    # ---- public API ----------------------------------------------------------
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self, cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[bool, str]:
        with self._lock:
            prev = self._proc
        # If a previous child is still tearing down (Stop was just pressed),
        # block briefly for it to exit before we sweep /dev/shm — otherwise
        # the old scheduler recreates its SharedMemoryQueue segments faster
        # than we can unlink them, and the new run_graph hits FileExistsError.
        if prev is not None and prev.poll() is None:
            # Verify the pid still exists at the OS level. The bazel/pixel
            # launcher can detach so ``proc.poll()`` is stuck at None even
            # though the process is gone — without this guard ``wait()``
            # below would block indefinitely.
            try:
                os.kill(prev.pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            if alive:
                try:
                    os.killpg(os.getpgid(prev.pid), signal.SIGINT)
                except (ProcessLookupError, OSError):
                    pass
                deadline = time.time() + 6.0
                for sig in (None, signal.SIGTERM, signal.SIGKILL):
                    if sig is not None:
                        try:
                            os.killpg(os.getpgid(prev.pid), sig)
                        except (ProcessLookupError, OSError):
                            break
                    while time.time() < deadline:
                        try:
                            os.kill(prev.pid, 0)
                        except ProcessLookupError:
                            break
                        time.sleep(0.1)
                    else:
                        continue
                    break
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    os.kill(self._proc.pid, 0)
                except ProcessLookupError:
                    self._proc = None
                else:
                    return False, "already running"
            # Keep previous run's lines for post-mortem; add a visible separator
            # instead of clearing the ring buffer.
            if self._lines:
                self._lines.append((self._next_id, ""))
                self._next_id += 1
                self._lines.append((self._next_id, f"==== new run @ {time.strftime('%H:%M:%S')} ===="))
                self._next_id += 1
            self._exit_code = None
            self._cmd = list(cmd)
            self._started_at = time.time()
        self._append(f"[runner] cwd: {cwd}")
        self._append(f"[runner] $ {' '.join(shlex.quote(c) for c in cmd)}")
        # Reap any orphan app processes from earlier runs that the previous
        # Stop did not collect (e.g. crashed schedulers). Without this the next
        # start fails on already-bound TCP ports (foxglove 8765, viser 8080).
        try:
            killed = 0
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry.name}/cmdline", "rb") as f:
                        cmdline = f.read()
                except OSError:
                    continue
                if b"_app_stage2_bootstrap" in cmdline or b"graph/runner" in cmdline:
                    try:
                        os.kill(int(entry.name), signal.SIGKILL)
                        killed += 1
                    except OSError:
                        pass
            if killed:
                self._append(f"[runner] reaped {killed} orphan app process(es)")
                time.sleep(0.5)
        except OSError:
            pass
        # Best-effort cleanup of stale /dev/shm segments left by a prior crash.
        # Worker queues from this app live in POSIX shm and re-creation fails
        # with ``FileExistsError`` if the previous run did not unlink cleanly.
        try:
            removed = 0
            for shm_entry in os.listdir("/dev/shm"):
                full = os.path.join("/dev/shm", shm_entry)
                try:
                    if os.path.isfile(full) and os.stat(full).st_uid == os.getuid():
                        os.unlink(full)
                        removed += 1
                except OSError:
                    pass
            if removed:
                self._append(f"[runner] cleaned {removed} stale /dev/shm segment(s)")
        except OSError:
            pass
        try:
            proc = subprocess.Popen(  # - user-controlled in a dev tool
                cmd,
                cwd=str(cwd),
                env={**os.environ, **(env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,  # so we can SIGINT the whole group
            )
        except OSError as e:
            self._append(f"[runner] failed to start: {e}")
            return False, str(e)
        with self._lock:
            self._proc = proc
            self._reader = threading.Thread(target=self._drain, args=(proc,), daemon=True)
            self._reader.start()
        return True, f"pid {proc.pid}"

    def stop(self, escalate: bool = False) -> tuple[bool, str]:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            # No live direct child. A previous server may have died without cleanup
            # (SSH drop / SIGKILL), leaving the graph orphaned in its own session --
            # so Stop must still hunt those down instead of silently doing nothing.
            reaped = self._reap_orphans()
            if reaped:
                return True, f"no direct child; killed {reaped} orphan process(es)"
            return False, "not running"
        if escalate:
            # Kill button: SIGKILL immediately, no escalation thread.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError) as e:
                if not isinstance(e, ProcessLookupError):
                    return False, str(e)
            self._append(f"[runner] sent SIGKILL to pgid {proc.pid}")
            return True, "killed"

        # Stop button: SIGINT, then escalate in a background thread so the UI
        # gets a definitive "exited" status even if the child ignores SIGINT
        # (bazel launchers, hung schedulers, blocked C extensions, etc.).
        # First: RESUME the clock. If the graph is paused, its pausable nodes are
        # blocked in the debug gate (``while _PAUSED``) and its master clock is
        # frozen, so the scheduler can never join them and shutdown hangs until
        # the SIGKILL escalation. Unpausing lets every node tick, run its safe
        # shutdown, and exit cleanly.
        _CTRL.set_paused(False)
        # First ask the child to begin its visible safe-shutdown sequence
        # (mujoco sit-down animation). The actual SIGINT is deferred up to
        # ``_SAFE_SHUTDOWN_GRACE_S`` to give the animation time to play
        # while viser is still alive.
        _CTRL.request_safe_shutdown()
        self._append(
            f"[runner] safe-shutdown requested on pgid {proc.pid}; "
            f"will SIGINT once child reports ready or after "
            f"{self._SAFE_SHUTDOWN_GRACE_S:.0f}s"
        )

        def _wait_then_signal(p: subprocess.Popen) -> None:
            deadline = time.time() + self._SAFE_SHUTDOWN_GRACE_S
            while time.time() < deadline:
                if _CTRL.is_shutdown_ready():
                    self._append("[runner] child reported shutdown_ready")
                    break
                if p.poll() is not None:
                    # Direct child gone (launcher exited/detached) -- its descendants
                    # may still be running; reap them rather than returning silently.
                    self._reap_orphans()
                    return
                time.sleep(0.1)
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except (ProcessLookupError, OSError):
                return
            self._append(f"[runner] sent SIGINT to pgid {p.pid}; escalating if needed")
            # SIGINT wait is long: on hardware, SpotControl.stop has to
            # stand → stow arm → sit → power-off → release-lease, which
            # realistically takes 10-15 s. Cut it short and we leave the
            # robot powered on holding a stale lease. SIGTERM/SIGKILL stay
            # short — by that point we're past graceful shutdown anyway.
            for sig, label, wait_s in (
                (signal.SIGINT, "SIGINT", 20.0),  # already sent; wait for it
                (signal.SIGTERM, "SIGTERM", 3.0),
                (signal.SIGKILL, "SIGKILL", 2.0),
            ):
                if sig is not signal.SIGINT:
                    try:
                        os.killpg(os.getpgid(p.pid), sig)
                        self._append(f"[runner] escalated to {label} on pgid {p.pid}")
                    except (ProcessLookupError, OSError):
                        return
                try:
                    p.wait(timeout=wait_s)
                    return
                except subprocess.TimeoutExpired:
                    continue

        threading.Thread(target=_wait_then_signal, args=(proc,), daemon=True).start()
        return True, "safe-shutdown in progress"

    def shutdown(self, timeout: float = 5.0) -> None:
        """Best-effort terminate the child + any orphan run_graph descendants.

        Called from the server's main on KeyboardInterrupt/SIGTERM so closing
        the editor does not leave ``run_graph`` (or its app-stage2 children)
        running and holding ports.
        """
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except ProcessLookupError:
                    break
                except OSError:
                    break
                try:
                    proc.wait(timeout=timeout if sig is signal.SIGINT else 1.0)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self._reap_orphans()

    def _reap_orphans(self) -> int:
        """SIGKILL any orphan run_graph / app-stage2 processes (ours by uid) and return the count.

        Orphans happen when a previous server died without cleanup (SIGKILL, SSH drop):
        the child runs in its own session and survives, holding viser/foxglove ports.
        """
        reaped = 0
        try:
            my_uid = os.getuid()
            my_pid = os.getpid()
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid == my_pid:
                    continue
                try:
                    if os.stat(entry.path).st_uid != my_uid:
                        continue
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmdline = f.read()
                except OSError:
                    continue
                if b"graph/runner" in cmdline or b"_app_stage2_bootstrap" in cmdline:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        reaped += 1
                        self._append(f"[runner] reaped orphan pid {pid}")
                    except OSError:
                        pass
        except OSError:
            pass
        return reaped

    def status(self, offset: int) -> dict[str, Any]:
        disappeared_msg: str | None = None
        with self._lock:
            proc = self._proc
            if proc is not None:
                # ``proc.poll()`` only reaps OUR direct child. The bazel/pixel
                # launcher can re-exec or detach, leaving ``proc.pid`` pointing
                # at a pid that is no longer our child; ``poll()`` then returns
                # None forever even after the process is gone. Cross-check by
                # asking the kernel whether the pid still exists.
                running = proc.poll() is None
                if running:
                    try:
                        os.kill(proc.pid, 0)
                    except ProcessLookupError:
                        running = False
                        if self._exit_code is None:
                            self._exit_code = -1
                            disappeared_msg = f"[runner] process {proc.pid} disappeared (detached or already reaped)"
            else:
                running = False
            pid = proc.pid if proc is not None else None
            exit_code = self._exit_code
            cmd = list(self._cmd)
            started_at = self._started_at
            lines = [(i, t) for (i, t) in self._lines if i >= offset]
            next_offset = self._next_id
        # ``_append`` re-acquires ``self._lock`` (non-reentrant) — never call
        # it inside the locked region above or the status endpoint deadlocks
        # and the UI freezes on "stopping (pid …)".
        if disappeared_msg is not None:
            self._append(disappeared_msg)
        return {
            "running": running,
            "pid": pid,
            "exit_code": exit_code,
            "cmd": cmd,
            "started_at": started_at,
            "lines": [t for (_i, t) in lines],
            "offset": next_offset,
        }


_RUNNER = _Runner()


class _TapStore:
    """In-memory store for live edge values streamed from the child app.

    The UI subscribes to a set of ``queue_name`` -> ``[edge_id, ...]`` mappings.
    The child polls ``GET /tap/queues`` for the active queue list and pushes
    latest values to ``POST /tap/ingest``. The store keeps the most recent
    sample per ``edge_id`` plus a monotonically increasing version so the UI
    can poll ``GET /tap/latest?since=N`` for changes.
    """

    HISTORY = 256  # rolling history per edge for the time-travel slider

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue_to_edges: dict[str, list[str]] = {}
        self._samples: dict[str, dict[str, Any]] = {}  # edge_id -> sample
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self._version = 0

    def set_taps(self, taps: list[dict[str, str]]) -> int:
        mapping: dict[str, list[str]] = {}
        for t in taps:
            qn = str(t.get("queue_name", ""))
            eid = str(t.get("edge_id", ""))
            if not qn or not eid:
                continue
            mapping.setdefault(qn, []).append(eid)
        with self._lock:
            prev_keep = {eid for lst in self._queue_to_edges.values() for eid in lst}
            self._queue_to_edges = mapping
            keep = {eid for lst in mapping.values() for eid in lst}
            # Keep samples/history ONLY for edges that were continuously subscribed (in both the old
            # and new tap sets). An edge that is newly (re)subscribed starts a FRESH measurement --
            # dropping its stale samples so a re-tap re-measures from scratch instead of showing the
            # previous session's frozen last value. Edges no longer subscribed are dropped too.
            survive = keep & prev_keep
            self._samples = {k: v for k, v in self._samples.items() if k in survive}
            self._history = {k: v for k, v in self._history.items() if k in survive}
            self._version += 1
            return self._version

    def queues(self) -> list[str]:
        with self._lock:
            return list(self._queue_to_edges.keys())

    def ingest(self, samples: list[dict[str, Any]]) -> int:
        with self._lock:
            for s in samples:
                qn = str(s.get("queue", ""))
                ts = float(s.get("ts", time.time()))
                value = s.get("value")
                stats = s.get("stats")
                for eid in self._queue_to_edges.get(qn, []):
                    self._version += 1
                    prev = self._samples.get(eid)
                    if value is None and prev is not None:
                        # Stats-only update (plain Run, or a not-freshly-pushed queue in Debug): keep
                        # the last value, just refresh the flow stats + version.
                        sample = {**prev, "ts": ts, "stats": stats, "v": self._version}
                    else:
                        sample = {"edge_id": eid, "ts": ts, "value": value, "stats": stats, "v": self._version}
                    self._samples[eid] = sample
                    hist = self._history.setdefault(eid, deque(maxlen=self.HISTORY))
                    hist.append(sample)
            return self._version

    def latest(self, since: int) -> dict[str, Any]:
        with self._lock:
            updates = [s for s in self._samples.values() if s["v"] > since]
            return {"version": self._version, "samples": updates}

    def history(self, edge_id: str) -> list[dict[str, Any]]:
        with self._lock:
            hist = self._history.get(edge_id)
            return list(hist) if hist else []

    def clear(self) -> None:
        with self._lock:
            self._queue_to_edges.clear()
            self._samples.clear()
            self._history.clear()
            self._version += 1


_TAPS = _TapStore()


class _Control:
    """Pause/step control plane consumed by the child app via polling.

    State:
      * ``paused``: while ``True``, every node blocks before entering ``exec``.
      * ``global_epoch``: monotonically increasing counter; bumping it lets
        each live node execute exactly once ("step all").
      * ``node_epochs[name]``: per-node counter; bumping the entry for a
        single node lets just that node execute one more time.
    """

    # A Step whose target advances virtual time by more than this uses the
    # RESUME-then-auto-pause path (control keeps updating so the robot stays
    # balanced); shorter steps use a precise frozen jump. ~150 ms: longer than
    # the 100 ms /control poll (so auto-pause overshoot is a small fraction of
    # the window) yet short enough that a held command can't topple a stand.
    _STEP_RESUME_THRESHOLD_NS = 150_000_000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused = False
        self._global_epoch = 0
        self._node_epochs: dict[str, int] = {}
        self._reset_epoch = 0
        self._replay_ts: float | None = None
        self._version = 0
        # Master-clock control plane (consumed by
        # :class:`judo.unlocked.nodes.master_clock.MasterClock` via
        # debug_tap). ``clock_paused`` is what gates the master tick; the
        # legacy ``_paused`` flag still drives the per-node debug_tap gate
        # so graphs not yet migrated to the master clock keep their old
        # Play/Pause semantics. Pressing Play/Pause toggles both.
        self._clock_paused = False
        self._clock_rate = 1.0
        # Monotonically-increasing total nanoseconds of virtual time the
        # editor has requested via Step. Each child's debug_tap diffs this
        # against its own local high-water-mark and adds the delta to its
        # local pending-step buffer, so a multi-click burst accumulates
        # correctly even if some polls are missed.
        self._clock_step_total_ns = 0
        # ``name -> period_ns`` advertised by each SubordinateTimer at
        # construction time (pushed via /control/clock/periods). The Step
        # button computes the advance amount as ``min(period)`` over the
        # configured step group.
        self._subordinate_periods_ns: dict[str, int] = {}
        # Optional explicit step group. ``None`` = default (every known
        # subordinate timer). Set via /control/clock/step_group.
        self._step_group: list[str] | None = None
        # ---- server-authoritative virtual clock ----------------------------
        # The single source of time every process derives from (via
        # ``debug_tap.now_ns``), so edge-tap sample timestamps and
        # Foxglove/MCAP message log-times share ONE clock and stay in sync.
        # Virtual time = wall time minus paused gaps plus Step jumps: it looks
        # like a normal Unix timestamp but FREEZES on Pause and JUMPS on Step.
        # ``_virtual_accum_ns`` is virtual time banked up to the last state
        # change; while running add ``(wall_now - _wall_anchor_ns) * rate``.
        _now = time.time_ns()
        self._virtual_accum_ns: int = _now
        self._wall_anchor_ns: int = _now
        # High-level controller (MPC) on/off, judo-style. Independent from
        # ``_paused``: when off the low-level command stream keeps running
        # (sending a safe DEFAULT_POLICY_CMD via the locomotion policy) so
        # the robot stays balanced. The user toggles this with Play/Pause.
        self._controller_running = False
        # Per-pid health snapshots. The graph runs in N forked subprocesses
        # and each subprocess's debug_tap maintains its own _NODE_REGISTRY,
        # so we MUST key by pid and aggregate; otherwise the last POST
        # wins and the UI flips ready/not-ready as each process reports.
        self._health_by_pid: dict[int, dict[str, Any]] = {}
        self._health: dict[str, Any] = {
            "ready": False,
            "started": 0,
            "total": 0,
            "pending": [],
            "errors": [],
            "unmanaged_threads": [],
        }
        # Safe-shutdown handshake. ``shutdown_requested`` is set when
        # the user presses Stop in the editor; the child (notably
        # :class:`MujocoSimulation`) mirrors it and begins a visible
        # sit-down animation. ``shutdown_ready`` aggregates the child's
        # readiness so :meth:`_Runner.stop` can defer SIGINT until the
        # animation completes.
        self._shutdown_requested: bool = False
        self._shutdown_ready_by_pid: dict[int, bool] = {}
        # Boot-tick handshake: the master clock starts paused (so the
        # user must press Play to begin virtual time), but we want one
        # round of subordinate ticks to fire the moment every pre_open
        # finishes so publishers prime initial state (sitting/standing
        # robot, mocap, viser scene) before the user clicks anything.
        # ``_ready_boot_fired`` flips True after that tick is queued.
        self._ready_boot_fired: bool = False
        # Step-to-timer resume window. For a Step whose target timer is slower
        # than the fastest timer, we RESUME the graph for one window (so faster
        # timers -- the control loop -- keep firing and the robot stays
        # balanced) and auto-pause at the end via this one-shot timer, instead
        # of a frozen virtual-time jump that would hold one stale command.
        self._step_pause_timer: "threading.Timer | None" = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snap_locked()

    def _virtual_now_ns_locked(self) -> int:
        """Current virtual time. Frozen while paused; else wall-paced by rate."""
        if self._clock_paused:
            return self._virtual_accum_ns
        return self._virtual_accum_ns + int((time.time_ns() - self._wall_anchor_ns) * self._clock_rate)

    def set_paused(self, paused: bool) -> dict[str, Any]:
        with self._lock:
            # A manual Play/Pause overrides any pending step-resume auto-pause.
            self._cancel_step_pause_locked()
            changed = False
            if self._paused != paused:
                self._paused = paused
                changed = True
            # Mirror Play/Pause into the master-clock pause flag so graphs
            # that opted into the master clock also start/stop with the
            # same button.
            if self._clock_paused != paused:
                # Virtual-clock transition: bank elapsed virtual time when
                # pausing (freeze it), re-anchor wall time when resuming.
                if paused:
                    self._virtual_accum_ns = self._virtual_now_ns_locked()
                else:
                    self._wall_anchor_ns = time.time_ns()
                self._clock_paused = paused
                changed = True
            if changed:
                self._version += 1
            return self._snap_locked()

    def step(self, node: str | None) -> dict[str, Any]:
        """Bump epoch to let one node (or every node) take one tick.

        Also forces ``paused=True`` so the step is well-defined. When the
        graph uses the master clock, additionally advance
        ``clock_step_total_ns`` to the next firing of the configured step
        group -- ``min(period)`` over the selected timers -- so execution
        resumes only until ONE selected timer triggers, then re-pauses. With
        no explicit selection the group is every timer (step to the next tick
        of anything). Falls back to ``max(period)`` (fire everything once)
        only before any periods are known.
        """
        print(f"[server] /control/step node={node!r}", file=sys.stderr, flush=True)
        with self._lock:
            # Bank virtual time before changing state if we were running.
            if not self._clock_paused:
                self._virtual_accum_ns = self._virtual_now_ns_locked()
            # Any manual/automatic step overrides a pending step-resume window.
            self._cancel_step_pause_locked()
            if node:
                # Single-node step: precise one-tick jump for that node.
                self._paused = True
                self._clock_paused = True
                self._node_epochs[node] = self._node_epochs.get(node, 0) + 1
                period = self._subordinate_periods_ns.get(node)
                if period and period > 0:
                    self._clock_step_total_ns += period
                    self._virtual_accum_ns += period
                self._version += 1
                return self._snap_locked()
            advance = self._min_step_group_period_ns_locked() or self._max_subordinate_period_ns_locked()
            if advance > self._STEP_RESUME_THRESHOLD_NS:
                # Long step: RESUME the whole graph for one window so the faster
                # timers (control) fire at their own rate and the robot stays
                # balanced, then auto-pause. A frozen virtual-time jump here
                # would advance physics by the whole window holding ONE stale
                # control command -> the robot falls.
                self._paused = False
                self._clock_paused = False
                self._wall_anchor_ns = time.time_ns()
                window_s = (advance / max(self._clock_rate, 1.0e-6)) * 1.0e-9
                self._schedule_step_pause_locked(window_s)
            else:
                # Short step (<= threshold): precise virtual-time jump. Holding
                # one control command for so brief a window can't topple a
                # standing robot, and the jump is exact (no ~poll-lag overshoot).
                self._paused = True
                self._clock_paused = True
                self._global_epoch += 1
                if advance > 0:
                    self._clock_step_total_ns += advance
                    self._virtual_accum_ns += advance
            self._version += 1
            return self._snap_locked()

    def _schedule_step_pause_locked(self, delay_s: float) -> None:
        """Arm a one-shot timer to auto-pause after a step-resume window."""
        self._cancel_step_pause_locked()
        t = threading.Timer(max(0.0, delay_s), self._auto_step_pause)
        t.daemon = True
        self._step_pause_timer = t
        t.start()

    def _cancel_step_pause_locked(self) -> None:
        if self._step_pause_timer is not None:
            self._step_pause_timer.cancel()
            self._step_pause_timer = None

    def _auto_step_pause(self) -> None:
        """Fired at the end of a step-resume window: pause the clock."""
        self.set_paused(True)

    def _min_step_group_period_ns_locked(self) -> int:
        """Smallest period among the configured step group; 0 only if NO timers are registered.

        A selected step group whose names don't match any registered timer (e.g. a stale selection
        persisted from another graph) falls back to the fastest timer overall rather than 0. That
        matters: returning 0 makes ``step()`` fall through to ``_max_subordinate_period`` and take a
        LONG resume window, which looks like Step never re-pauses. Falling back to the fastest timer
        keeps it a short, precise step that stays paused.
        """
        if not self._subordinate_periods_ns:
            return 0
        if self._step_group is not None:
            members = [self._subordinate_periods_ns[n] for n in self._step_group if n in self._subordinate_periods_ns]
            if members:
                return min(members)
            # Stale / mismatched selection -> step the fastest timer instead of a long resume.
        return min(self._subordinate_periods_ns.values())

    def _max_subordinate_period_ns_locked(self) -> int:
        """Largest known subordinate period; 0 if none registered yet.

        Used by the boot-prime path: advancing virtual time by this much
        guarantees every subordinate timer crosses its deadline at least
        once, so every downstream node fires once and publishers prime
        initial state into viser.
        """
        if not self._subordinate_periods_ns:
            return 0
        return max(self._subordinate_periods_ns.values())

    def set_clock_periods(self, periods_ns: dict[str, int]) -> dict[str, Any]:
        """Merge subordinate-timer periods reported by a child process."""
        with self._lock:
            for name, period in periods_ns.items():
                try:
                    self._subordinate_periods_ns[str(name)] = int(period)
                except (TypeError, ValueError):
                    continue
            # Periods arriving late can be the trigger for boot prime if
            # health already flipped ready before any subprocess managed
            # to POST its registered periods.
            self._maybe_fire_boot_prime_locked()
            self._version += 1
            return self._snap_locked()

    def set_clock_rate(self, rate: float) -> dict[str, Any]:
        with self._lock:
            # Re-anchor so the rate change doesn't create a discontinuity in
            # the virtual clock.
            self._virtual_accum_ns = self._virtual_now_ns_locked()
            self._wall_anchor_ns = time.time_ns()
            self._clock_rate = max(float(rate), 1.0e-6)
            self._version += 1
            return self._snap_locked()

    def set_step_group(self, group: list[str] | None) -> dict[str, Any]:
        with self._lock:
            self._step_group = list(group) if group is not None else None
            self._version += 1
            return self._snap_locked()

    def reset(self) -> None:
        """Reset the control plane for a fresh subprocess launch.

        Starts **running** (both legacy per-node gate and master clock) so
        pressing Run immediately advances virtual time. Play/Pause then
        acts as Pause/Resume. ``global_epoch`` is bumped
        to 1 so every legacy (non-master) node ticks exactly once and
        publishes its initial state to downstream consumers. The master-
        clock boot tick is issued by :class:`MasterClock` itself on its
        first call (one fixed-resolution tick regardless of pause), so
        every subordinate timer fires once and primes the same initial
        publish path.
        """
        with self._lock:
            self._cancel_step_pause_locked()
            self._paused = False
            self._global_epoch = 1
            self._node_epochs.clear()
            self._reset_epoch = 0
            self._replay_ts = None
            self._controller_running = False
            self._health_by_pid = {}
            self._health = {
                "ready": False,
                "started": 0,
                "total": 0,
                "pending": [],
                "errors": [],
                "unmanaged_threads": [],
            }
            self._shutdown_requested = False
            self._shutdown_ready_by_pid = {}
            # Master-clock mirrors the global run state: start unpaused so
            # Run immediately executes the graph and the UI shows Pause.
            self._clock_paused = False
            self._clock_rate = 1.0
            self._clock_step_total_ns = 0
            self._subordinate_periods_ns.clear()
            self._step_group = None
            # Restart the virtual clock at the current wall time so it looks
            # like a normal Unix timestamp and advances with wall time until
            # the user pauses/steps.
            _now = time.time_ns()
            self._virtual_accum_ns = _now
            self._wall_anchor_ns = _now
            self._ready_boot_fired = False
            self._version += 1

    def request_reset(self) -> dict[str, Any]:
        """Soft-reset: pause and bump ``reset_epoch`` so the child reinits each node's state in place.

        No subprocess restart. Also bumps ``global_epoch`` by one so every node gets exactly one
        tick after the reset completes. That single tick is what lets the
        state/mocap publishers and viser republish the freshly-
        reset world — otherwise viser stays frozen on the pre-reset frame
        and the user has to hit Play to see anything change.

        Also clears ``controller_running`` so the high-level controller
        is parked after a reset (matches judo: Reset = stop controller +
        re-init state). On hardware this is critical: we do NOT freeze the
        graph (which would starve the joint-control stream and trip Spot's
        safety stop) — the low-level loop keeps streaming a safe
        ``DEFAULT_POLICY_CMD`` via SplineFollower while node instances
        reset in place.
        """
        print("[server] /control/reset_request", file=sys.stderr, flush=True)
        with self._lock:
            self._controller_running = False
            self._reset_epoch += 1
            self._global_epoch += 1
            # Issue one round of subordinate ticks so each timer fires once
            # after the reset applies and republishes the fresh state
            # (otherwise viser stays frozen on the pre-reset frame while
            # the master clock is paused). ``max(period)`` guarantees
            # every subordinate -- including the slow viser_timer
            # -- crosses its deadline.
            advance = self._max_subordinate_period_ns_locked()
            if advance > 0:
                self._clock_step_total_ns += advance
            self._version += 1
            return self._snap_locked()

    def set_replay(self, ts: float | None) -> dict[str, Any]:
        """Set the global time-travel timestamp (``None`` = follow live)."""
        with self._lock:
            new_ts = float(ts) if ts is not None else None
            if self._replay_ts != new_ts:
                self._replay_ts = new_ts
                self._version += 1
            return self._snap_locked()

    def set_controller_running(self, running: bool) -> dict[str, Any]:
        """Toggle the high-level controller on/off (judo-style Play/Pause).

        Does NOT touch ``_paused``: the rest of the graph keeps ticking so
        the low-level command stream stays alive. SplineFollower reads this
        flag (via :func:`judo.unlocked.debug_tap.is_controller_running`)
        and switches between MPC-driven commands and ``DEFAULT_POLICY_CMD``.

        The health gate is informational only — the UI surfaces "starting
        (N/M ready)" so the user can see startup progress, but Play is
        never silently refused server-side. Cross-process health
        aggregation has races (each forked subprocess reports its own
        node count independently); rather than gate on a possibly-stale
        aggregate we trust the user's intent.
        """
        print(f"[server] /control/controller running={running}", file=sys.stderr, flush=True)
        with self._lock:
            if self._controller_running != bool(running):
                self._controller_running = bool(running)
                self._version += 1
            return self._snap_locked()

    def _maybe_fire_boot_prime_locked(self) -> None:
        """Queue one round of subordinate ticks if the system just came up.

        Conditions: aggregated health is ready, at least one subordinate
        period has been reported, and we haven't already fired. Uses
        ``max(period)`` so every subordinate (including the slowest,
        e.g. ``viser_timer`` at 10 Hz) crosses its deadline
        once and publishers prime initial state into viser. Called from
        :meth:`set_health` and :meth:`set_clock_periods` so whichever
        message arrives last triggers the prime -- subprocess startup
        ordering is not guaranteed.
        """
        if self._ready_boot_fired:
            return
        if not self._health.get("ready"):
            return
        advance = self._max_subordinate_period_ns_locked()
        if advance <= 0:
            return  # periods not reported yet; retry on next update
        self._clock_step_total_ns += advance
        self._ready_boot_fired = True

    def set_health(self, health: dict[str, Any]) -> dict[str, Any]:
        """Record the latest per-node readiness snapshot from one child pid.

        Each forked subprocess's ``debug_tap._health_loop`` POSTs its own
        local view; we key by ``pid`` and aggregate so the UI sees the
        union across processes.
        """
        with self._lock:
            pid = int(health.get("pid", 0))
            self._health_by_pid[pid] = {
                "ready": bool(health.get("ready", False)),
                "started": int(health.get("started", 0)),
                "total": int(health.get("total", 0)),
                "pending": list(health.get("pending") or [])[:16],
                "errors": list(health.get("errors") or [])[:16],
                "unmanaged_threads": list(health.get("unmanaged_threads") or [])[:32],
            }
            # ``shutdown_ready`` is reported alongside the readiness
            # snapshot. Default True: pids with no MujocoSimulation-like
            # instance never block the safe-shutdown handshake.
            self._shutdown_ready_by_pid[pid] = bool(health.get("shutdown_ready", True))
            self._health = self._aggregate_health_locked()
            self._maybe_fire_boot_prime_locked()
            self._version += 1
            return self._snap_locked()

    def request_safe_shutdown(self) -> dict[str, Any]:
        """Ask the child to begin its safe-shutdown sequence.

        Triggered by :meth:`_Runner.stop` before SIGINT so the user sees
        the simulated robot stow its arm + sit down in viser instead of
        the subprocess just disappearing. Also clears
        ``controller_running`` so the high-level loop stops issuing MPC
        commands during the sit-down ramp.
        """
        with self._lock:
            if not self._shutdown_requested:
                self._shutdown_requested = True
                self._controller_running = False
                self._version += 1
            return self._snap_locked()

    def is_shutdown_ready(self) -> bool:
        """True once every reporting pid has finished its safe-shutdown."""
        with self._lock:
            if not self._shutdown_ready_by_pid:
                return False
            return all(self._shutdown_ready_by_pid.values())

    def _aggregate_health_locked(self) -> dict[str, Any]:
        """Union per-pid health snapshots into a single rollup.

        ``ready`` is True iff every reporting pid is ready AND at least
        one pre_open node exists across all pids. Pending lists and
        error lists are unioned (capped at 16 each).
        """
        if not self._health_by_pid:
            return {"ready": False, "started": 0, "total": 0, "pending": [], "errors": [], "unmanaged_threads": []}
        started = sum(h["started"] for h in self._health_by_pid.values())
        total = sum(h["total"] for h in self._health_by_pid.values())
        pending: list[str] = []
        errors: list[dict[str, str]] = []
        unmanaged: list[dict[str, Any]] = []
        seen_threads: set[tuple[Any, Any]] = set()
        for h in self._health_by_pid.values():
            pending.extend(h.get("pending", []))
            errors.extend(h.get("errors", []))
            for t in h.get("unmanaged_threads", []):
                # Normalize legacy bare-string entries to the {name, process, owner} shape.
                td = t if isinstance(t, dict) else {"name": t, "process": None, "owner": None}
                key = (td.get("process"), td.get("name"))
                if key in seen_threads:
                    continue
                seen_threads.add(key)
                unmanaged.append(td)
        ready = total > 0 and started == total and all(h.get("ready", False) for h in self._health_by_pid.values())
        return {
            "ready": ready,
            "started": started,
            "total": total,
            "pending": pending[:16],
            "errors": errors[:16],
            "unmanaged_threads": sorted(unmanaged, key=lambda d: (d.get("process") or "", d.get("name") or ""))[:64],
        }

    def _snap_locked(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "global_epoch": self._global_epoch,
            "node_epochs": dict(self._node_epochs),
            "reset_epoch": self._reset_epoch,
            "replay_ts": self._replay_ts,
            "controller_running": self._controller_running,
            "health": dict(self._health),
            "shutdown_requested": self._shutdown_requested,
            "shutdown_ready": bool(self._shutdown_ready_by_pid) and all(self._shutdown_ready_by_pid.values()),
            # Master-clock state, consumed by debug_tap mirrors in each
            # child pid. ``clock_step_total_ns`` is monotonically increasing;
            # children diff it against their local high-water mark to derive
            # the per-poll pending advance, so multi-click bursts accumulate
            # correctly without double-applying.
            "clock_paused": self._clock_paused,
            "clock_rate": self._clock_rate,
            "clock_step_total_ns": self._clock_step_total_ns,
            # Server-authoritative virtual clock. Children mirror these and
            # compute ``now_ns`` locally (same-machine wall clock), so every
            # process shares one timeline that freezes on pause / jumps on step.
            "virtual_accum_ns": self._virtual_accum_ns,
            "wall_anchor_ns": self._wall_anchor_ns,
            "virtual_ns": self._virtual_now_ns_locked(),
            "subordinate_periods_ns": dict(self._subordinate_periods_ns),
            "step_group": list(self._step_group) if self._step_group is not None else None,
            "version": self._version,
        }


_CTRL = _Control()

# Whether the current/last graph run is a Debug run (master-clock gating + edge
# taps + pause/step via ``UNLOCKED_TAP_URL``) or a plain lightweight Run (no tap
# install: no taps, no clock gating -- only Stop/Kill apply). Surfaced in
# ``/run/status`` so the editor shows the right controls after a reload.
_RUN_DEBUG: bool = True


def _build_app_command(flags: dict[str, Any]) -> list[str]:
    """Translate UI form values into a runner argv for the ``--builder`` graph."""
    cmd = [sys.executable, "-m", "judo.unlocked.graph.runner", f"--builder={_BUILDER_SPEC}"]
    log_level = flags.get("log_level")
    if log_level and log_level != "INFO":
        cmd += ["--log-level", str(log_level)]
    for key in ("task", "optimizer"):
        if flags.get(key):
            cmd += ["--builder-kwarg", f"{key}={flags[key]}"]
    extra = flags.get("extra_args")
    if isinstance(extra, list):
        cmd += [str(x) for x in extra]
    elif isinstance(extra, str) and extra.strip():
        cmd += shlex.split(extra)
    return cmd


def _build_run_graph_command(graph_json_path: str) -> list[str]:
    """Build a command that runs the graph runner against ``graph_json_path``."""
    return [sys.executable, "-m", "judo.unlocked.graph.runner", f"--graph-json={graph_json_path}"]


def _build_build_graph_command() -> list[str]:
    """No compile step in pure Python; just verify the runner imports."""
    return [sys.executable, "-c", "import judo.unlocked.graph.runner"]


class _Handler(BaseHTTPRequestHandler):
    # Class-level so all threads share the same in-memory graph.
    graph_json: str = "{}"
    # Last builder output before edit merging; kwargs still equal to it are not user edits.
    built_json: str = "{}"

    def log_message(self, fmt: str, *args: object) -> None:  # silence default stderr logging
        _LOG.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # (stdlib API)
        url = urllib.parse.urlparse(self.path)
        path = url.path
        if path == "/graph.json":
            self._send(HTTPStatus.OK, _Handler.graph_json.encode("utf-8"), "application/json")
            return
        if path == "/edge_config":
            body = json.dumps({"expected_hz": _load_edge_config()}).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/resolve":
            params = urllib.parse.parse_qs(url.query)
            target = (params.get("target") or [""])[0]
            res = _resolve_target(target) if target else None
            if res is None:
                self._send(HTTPStatus.NOT_FOUND, b'{"error": "not found"}', "application/json")
                return
            file_path, line = res
            body = json.dumps({"file": str(file_path), "line": line}).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/instance_state":
            params = urllib.parse.parse_qs(url.query)
            target = (params.get("target") or [""])[0]
            info = _introspect_class(target) if target else None
            if info is None:
                self._send(HTTPStatus.NOT_FOUND, b'{"error": "not found"}', "application/json")
                return
            body = json.dumps(info).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/instance_schema":
            params = urllib.parse.parse_qs(url.query)
            target = (params.get("target") or [""])[0]
            raw_kwargs = (params.get("kwargs") or ["{}"])[0]
            try:
                cur = json.loads(raw_kwargs) if raw_kwargs else {}
                if not isinstance(cur, dict):
                    cur = {}
            except json.JSONDecodeError:
                cur = {}
            if not target:
                self._send(HTTPStatus.NOT_FOUND, b'{"error": "not found"}', "application/json")
                return
            try:
                fields = _instance_schema(target, cur)
            except Exception as e:
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR, json.dumps({"error": str(e)}).encode("utf-8"), "application/json"
                )
                return
            body = json.dumps({"target": target, "fields": fields}).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/port_type":
            params = urllib.parse.parse_qs(url.query)
            target = (params.get("target") or [""])[0]
            side = (params.get("side") or [""])[0]
            try:
                index = int((params.get("index") or ["0"])[0])
            except ValueError:
                index = 0
            info = _resolve_port_type(target, side, index) if target and side in ("in", "out") else None
            if info is None:
                self._send(HTTPStatus.NOT_FOUND, b'{"error": "not found"}', "application/json")
                return
            body = json.dumps(info).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/run/status":
            params = urllib.parse.parse_qs(url.query)
            try:
                offset = int((params.get("offset") or ["0"])[0])
            except ValueError:
                offset = 0
            st = _RUNNER.status(offset)
            st["debug"] = _RUN_DEBUG
            body = json.dumps(st).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/builder/args":
            try:
                params_list = _builder_params(_BUILDER_SPEC) if _BUILDER_SPEC else []
                payload = {
                    "spec": _BUILDER_SPEC,
                    "params": params_list,
                    "kwargs": _BUILDER_KWARGS,
                    "meta": _builder_meta(_BUILDER_SPEC) if _BUILDER_SPEC else {},
                    "skill_policies": _PERSIST.get("skill_policies", {}),
                }
                self._send(HTTPStatus.OK, json.dumps(payload).encode("utf-8"), "application/json")
            except Exception as e:
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR, json.dumps({"error": str(e)}).encode("utf-8"), "application/json"
                )
            return
        if path == "/tap/queues":
            body = json.dumps({"queues": _TAPS.queues()}).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/tap/latest":
            params = urllib.parse.parse_qs(url.query)
            try:
                since = int((params.get("since") or ["0"])[0])
            except ValueError:
                since = 0
            body = json.dumps(_TAPS.latest(since)).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/tap/history":
            params = urllib.parse.parse_qs(url.query)
            edge = (params.get("edge") or [""])[0]
            body = json.dumps({"edge_id": edge, "samples": _TAPS.history(edge)}).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/control":
            body = json.dumps(_CTRL.snapshot()).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/":
            path = "/index.html"
        fs_path = (_STATIC_DIR / path.lstrip("/")).resolve()
        if _STATIC_DIR not in fs_path.parents and fs_path != _STATIC_DIR / "index.html":
            # Stay inside static/.
            if not str(fs_path).startswith(str(_STATIC_DIR)):
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
                return
        if not fs_path.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        ctype = _MIME.get(fs_path.suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, fs_path.read_bytes(), ctype)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        if path == "/builder/build":
            # Rebuild the graph from the builder + user-supplied kwargs and
            # publish the result as the new in-memory graph_json. Returns the
            # spec so the editor can refresh the diagram immediately.
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            if not _BUILDER_SPEC:
                self._send(HTTPStatus.CONFLICT, b'{"error": "no builder configured"}', "application/json")
                return
            kwargs = dict(data.get("kwargs") or {})
            try:
                graph = _load_builder(_BUILDER_SPEC, kwargs)
            except Exception as e:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    json.dumps({"ok": False, "error": str(e)}).encode("utf-8"),
                    "application/json",
                )
                return
            global _BUILDER_KWARGS
            _BUILDER_KWARGS = kwargs
            # Remember this config for subsequent runs (and the skill policy for a
            # vision task) so a server restart / fresh page load resumes here.
            _remember_config(kwargs)
            spec = graph.to_dict()
            built_json = json.dumps(spec)
            # Re-propagate app-level fields to every instance (the fresh build) while keeping the
            # user's per-instance tuning (edited, non app-level kwargs) from the previous spec.
            try:
                _merge_instance_edits(spec, _Handler.graph_json, _Handler.built_json)
            except Exception:  # (never let edit-merge break a rebuild)
                _LOG.exception("merge instance edits failed; using fresh build")
            _Handler.built_json = built_json
            _Handler.graph_json = json.dumps(spec, indent=2)
            self._send(
                HTTPStatus.OK,
                json.dumps({"ok": True, "graph": spec, "kwargs": kwargs}).encode("utf-8"),
                "application/json",
            )
            return
        if path == "/build/graph":
            # Bazel-build the run_graph binary (via pixel) so subsequent Runs
            # take the fast direct-launch path. Streams build output into the
            # same console through ``_RUNNER``; it's a one-shot command that
            # exits, after which Run can be pressed.
            cmd = _build_build_graph_command()
            ok, msg = _RUNNER.start(cmd, _WORKSPACE)
            body = json.dumps({"ok": ok, "msg": msg, "cmd": cmd}).encode("utf-8")
            status = HTTPStatus.OK if ok else HTTPStatus.CONFLICT
            self._send(status, body, "application/json")
            return
        if path == "/run/graph":
            # Run the in-memory graph spec via ``run_graph --graph-json=PATH``.
            # Writes the current ``_Handler.graph_json`` to a tempfile so the
            # child can be relaunched without coupling it to this server's
            # lifetime.
            import tempfile

            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {}
            # Debug run (default): install the debug tap (edge taps + master-clock
            # gating + pause/step). Plain Run: omit ``UNLOCKED_TAP_URL`` so
            # ``debug_tap.install()`` is a no-op -- a truly lightweight run with
            # no tap overhead and no clock gating (only Stop/Kill apply).
            debug = bool(data.get("debug", True))
            global _RUN_DEBUG
            _RUN_DEBUG = debug

            tmp_dir = Path(tempfile.gettempdir()) / "unlocked_editor"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / "current_graph.json"
            tmp_path.write_text(_Handler.graph_json, encoding="utf-8")
            cmd = _build_run_graph_command(str(tmp_path))
            # The child always runs on this machine, so point it at the locally-bound
            # server port -- NEVER at the browser's Host header, which is a tunnel-local
            # address (e.g. a remapped VS Code port-forward) that may not exist here.
            # A wrong URL silently breaks taps ("measuring..." forever) and health.
            host = f"127.0.0.1:{_SERVE_PORT}"
            # Debug: full tap (values + pause/step). Plain Run: a lightweight STATS-ONLY tap so the
            # user can still probe one edge's flow (count/rate/staleness) to see where data is blocked
            # or slow -- without value summarization or clock gating.
            env = (
                {"UNLOCKED_TAP_URL": f"http://{host}"}
                if debug
                else {"UNLOCKED_TAP_URL": f"http://{host}", "UNLOCKED_TAP_STATS_ONLY": "1"}
            )
            _TAPS.clear()
            # Reset clears ``controller_running``: judo-style, the high-level
            # controller starts parked. The graph itself runs normally so
            # the low-level loop streams DEFAULT_POLICY_CMD (zero velocity,
            # standing, stowed arm) — on hardware this keeps the joint-
            # control stream alive and Spot standing; on sim the robot
            # holds its reset pose. Press Play to start the controller.
            _CTRL.reset()
            ok, msg = _RUNNER.start(cmd, _WORKSPACE, env=env)
            body = json.dumps({"ok": ok, "msg": msg, "cmd": cmd, "graph_json": str(tmp_path), "debug": debug}).encode(
                "utf-8"
            )
            status = HTTPStatus.OK if ok else HTTPStatus.CONFLICT
            self._send(status, body, "application/json")
            return
        if path == "/run":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            flags = data.get("flags") or {}
            cmd = _build_app_command(flags)
            host = self.headers.get("Host") or "127.0.0.1:8770"
            env = {
                "UNLOCKED_TAP_URL": f"http://{host}",
            }
            _TAPS.clear()
            _CTRL.reset()
            ok, msg = _RUNNER.start(cmd, _WORKSPACE, env=env)
            body = json.dumps({"ok": ok, "msg": msg, "cmd": cmd}).encode("utf-8")
            status = HTTPStatus.OK if ok else HTTPStatus.CONFLICT
            self._send(status, body, "application/json")
            return
        if path == "/stop":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {}
            ok, msg = _RUNNER.stop(escalate=bool(data.get("kill")))
            body = json.dumps({"ok": ok, "msg": msg}).encode("utf-8")
            status = HTTPStatus.OK if ok else HTTPStatus.CONFLICT
            self._send(status, body, "application/json")
            return
        if path == "/open-in-editor":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            file = str(data.get("file") or "")
            try:
                line = int(data.get("line") or 1)
            except (TypeError, ValueError):
                line = 1
            if not file:
                self._send(HTTPStatus.BAD_REQUEST, b'{"ok": false, "msg": "no file"}', "application/json")
                return
            ok, msg = _open_in_editor(file, line)
            body = json.dumps({"ok": ok, "msg": msg}).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/edge_config":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            ok, msg = _save_edge_config(dict(data.get("expected_hz") or {}))
            body = json.dumps({"ok": ok, "msg": msg, "expected_hz": _load_edge_config()}).encode("utf-8")
            self._send(HTTPStatus.OK if ok else HTTPStatus.INTERNAL_SERVER_ERROR, body, "application/json")
            return
        if path == "/tap/edges":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            version = _TAPS.set_taps(list(data.get("taps") or []))
            body = json.dumps({"ok": True, "version": version}).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/tap/ingest":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            version = _TAPS.ingest(list(data.get("samples") or []))
            body = json.dumps({"ok": True, "version": version}).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if path == "/control/pause":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            snap = _CTRL.set_paused(bool(data.get("paused", True)))
            self._send(HTTPStatus.OK, json.dumps(snap).encode("utf-8"), "application/json")
            return
        if path == "/control/step":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            node = data.get("node")
            snap = _CTRL.step(str(node) if node else None)
            self._send(HTTPStatus.OK, json.dumps(snap).encode("utf-8"), "application/json")
            return
        if path == "/control/clock/periods":
            # Subordinate-timer periods registered by a child pid's
            # debug_tap (one POST per change). Used by the Step button to
            # compute min(period) across the step group.
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            periods = data.get("periods_ns") or {}
            if not isinstance(periods, dict):
                self._send(HTTPStatus.BAD_REQUEST, b"periods_ns must be an object", "text/plain")
                return
            snap = _CTRL.set_clock_periods(periods)
            self._send(HTTPStatus.OK, json.dumps(snap).encode("utf-8"), "application/json")
            return
        if path == "/control/clock/rate":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            try:
                rate = float(data.get("rate", 1.0))
            except (TypeError, ValueError):
                self._send(HTTPStatus.BAD_REQUEST, b"rate must be a number", "text/plain")
                return
            snap = _CTRL.set_clock_rate(rate)
            self._send(HTTPStatus.OK, json.dumps(snap).encode("utf-8"), "application/json")
            return
        if path == "/control/clock/step_group":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            group = data.get("group")
            if group is not None and not isinstance(group, list):
                self._send(HTTPStatus.BAD_REQUEST, b"group must be a list or null", "text/plain")
                return
            snap = _CTRL.set_step_group([str(x) for x in group] if group is not None else None)
            self._send(HTTPStatus.OK, json.dumps(snap).encode("utf-8"), "application/json")
            return
        if path == "/control/reset_request":
            # Soft-reset: pause + bump ``reset_epoch`` so the child reinits
            # each node instance in place (no subprocess restart, no viser
            # tab refresh, no module re-import).
            snap = _CTRL.request_reset()
            self._send(HTTPStatus.OK, json.dumps(snap).encode("utf-8"), "application/json")
            return
        if path == "/control/controller":
            # Toggle the high-level controller (MPC) on/off. The rest of
            # the graph keeps ticking; SplineFollower switches between
            # MPC-driven commands and DEFAULT_POLICY_CMD based on this
            # flag. Safe on both mujoco and hardware.
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            snap = _CTRL.set_controller_running(bool(data.get("running", False)))
            self._send(HTTPStatus.OK, json.dumps(snap).encode("utf-8"), "application/json")
            return
        if path == "/control/health":
            # Per-node readiness POSTed by the child's tap (debug_tap
            # ``_health_loop``). Mirrored into the ``/control`` snapshot
            # so the UI can show "starting (N/M ready)" and refuse Play
            # until every ``pre_open`` (notably the robot startup
            # sequence) has completed.
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            snap = _CTRL.set_health(data)
            self._send(HTTPStatus.OK, json.dumps(snap).encode("utf-8"), "application/json")
            return
        if path == "/control/replay":
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
                return
            ts = data.get("ts")
            snap = _CTRL.set_replay(None if ts is None else float(ts))
            self._send(HTTPStatus.OK, json.dumps(snap).encode("utf-8"), "application/json")
            return
        if path != "/graph.json":
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            self._send(HTTPStatus.BAD_REQUEST, f"invalid json: {e}".encode(), "text/plain")
            return
        # Validate by round-tripping through GraphBuilder.
        try:
            GraphBuilder.from_dict(data)
        except Exception as e:
            self._send(HTTPStatus.BAD_REQUEST, f"invalid graph: {e}".encode(), "text/plain")
            return
        _Handler.graph_json = json.dumps(data, indent=2)
        _LOG.info("graph updated (%d nodes, %d edges)", len(data.get("nodes", [])), len(data.get("edges", [])))
        print("---- updated graph ----")
        print(_Handler.graph_json)
        print("-----------------------")
        self._send(HTTPStatus.OK, b"ok", "text/plain")


class _EditorServer(ThreadingHTTPServer):
    """Editor HTTP server tuned for the tap ingest load.

    Every run_graph worker process runs its own debug-tap drain loop that POSTs ``/tap/ingest``
    concurrently (high-rate queues fire many times a second). The stdlib default listen backlog of 5
    overflows under that fan-in and the OS resets pending connections; a larger backlog absorbs the
    bursts. ``daemon_threads`` lets the process exit without joining in-flight request threads.
    """

    daemon_threads = True
    request_queue_size = 128


def _parse_builder_kwarg(item: str) -> tuple[str, Any]:
    """Parse a ``KEY=VALUE`` builder kwarg; VALUE is JSON when possible, else a bare string.

    Mirrors the runner's flag (``true``/``false``/numbers/quoted strings parse as JSON; a bare word
    like ``multi_spot_table_lift`` stays a string) so the editor and the runner take kwargs the same
    way.
    """
    key, sep, raw = item.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"--builder-kwarg must be KEY=VALUE, got {item!r}")
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")  # (intentionally bind all interfaces)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument(
        "--builder",
        default="judo.app.graph:build",
        help="module:callable returning a GraphBuilder used to seed the initial graph.",
    )
    p.add_argument(
        "--builder-kwarg",
        action="append",
        default=[],
        type=_parse_builder_kwarg,
        metavar="KEY=VALUE",
        help="Keyword argument for the --builder callable (repeatable). VALUE is parsed as JSON when "
        "possible, else a bare string. Overrides this builder's persisted editor state; e.g. "
        "--builder-kwarg task=multi_spot_table_lift.",
    )
    p.add_argument(
        "--search-root",
        action="append",
        default=[],
        metavar="DIR",
        help="Additional directory to scan when resolving ``module:qualname`` "
        "targets (repeatable). Also honors $UNLOCKED_SEARCH_ROOTS.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Seed extra search roots from env + CLI.
    for r in _user_extra_roots():
        if r not in _EXTRA_ROOTS:
            _EXTRA_ROOTS.append(r)
    for raw in args.search_root:
        r = Path(raw).expanduser()
        if r.is_dir() and r not in _EXTRA_ROOTS:
            _EXTRA_ROOTS.append(r)
    if _EXTRA_ROOTS:
        _LOG.info("extra search roots: %s", ", ".join(str(r) for r in _EXTRA_ROOTS))

    global _BUILDER_SPEC
    _BUILDER_SPEC = args.builder

    # Restore the last session's config + per-task skill policies, then seed the
    # initial in-memory graph from that config so a fresh page load shows where
    # the user left off (falls back to builder defaults on any error).
    _load_persisted_state()
    global _BUILDER_KWARGS
    cli_kwargs = dict(args.builder_kwarg)  # explicit CLI kwargs win over persisted state
    if cli_kwargs:
        merged = {**_BUILDER_KWARGS, **cli_kwargs}
        try:
            _Handler.graph_json = _load_builder(args.builder, merged).to_json()
            _BUILDER_KWARGS = merged
            _PERSIST["kwargs"] = dict(merged)
        except Exception as exc:  # (a bad --builder-kwarg must not block startup)
            _LOG.error("builder %s rejected --builder-kwarg %s (%s); ignoring it", args.builder, cli_kwargs, exc)
            _Handler.graph_json = _load_builder(args.builder, _BUILDER_KWARGS or None).to_json()
    else:
        try:
            _Handler.graph_json = _load_builder(args.builder, _BUILDER_KWARGS or None).to_json()
        except Exception:  # (a bad persisted config must not block startup)
            _LOG.exception("restoring persisted config failed; using builder defaults")
            _Handler.graph_json = _load_builder(args.builder).to_json()
    _Handler.built_json = _Handler.graph_json

    global _SERVE_PORT
    _SERVE_PORT = int(args.port)
    httpd = _EditorServer((args.host, args.port), _Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"unLocked graph editor: {url}", flush=True)
    print(f"  initial graph from: {args.builder}", flush=True)

    def _on_term(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_term)
    # An SSH/terminal drop delivers SIGHUP; without this the server dies with NO
    # cleanup and the running graph is orphaned (next server's Stop finds nothing).
    signal.signal(signal.SIGHUP, _on_term)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _RUNNER.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
