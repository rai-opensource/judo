# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""In-process debug tap.

When the environment variable ``UNLOCKED_TAP_URL`` is set (typically by the
React Flow editor's ``/run`` endpoint), this module installs a lightweight tap
on :class:`SharedMemoryQueue.push` that captures the latest value pushed to
each subscribed queue and forwards summarized snapshots to the editor server
over HTTP.

The flow is intentionally pull-based and rate-limited:

* A background thread polls ``GET {UNLOCKED_TAP_URL}/queues`` periodically and
  refreshes the set of queue names to capture.
* Each push to a captured queue replaces the per-queue *latest* slot; no
  history is kept on the producer side.
* A second thread drains all dirty slots every ~100ms, serializes each value
  to a renderer-friendly JSON payload, and POSTs the batch to
  ``{UNLOCKED_TAP_URL}/ingest``.

This decouples the producer from the network and bounds overhead per push to
a single dict assignment under a small lock.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import struct
import sys
import threading
import time
import urllib.request
import zlib
from functools import wraps
from typing import Any

_LOG = logging.getLogger("unlocked.debug_tap")

# These globals are set by ``install`` and consulted by the patched ``push``.
_INSTALLED_PID: int = 0  # pid the worker threads belong to
_PATCHED: bool = False  # SharedMemoryQueue.push has been monkey-patched
_LOCK = threading.Lock()
_TAPPED: set[str] = set()  # local runtime queue names to capture
_NAME_MAP: dict[str, str] = {}  # local runtime -> subscribed (UI) name
_KNOWN_QUEUES: set[str] = set()  # all SharedMemoryQueue names created in this process
_LATEST: dict[str, tuple[float, Any]] = {}  # subscribed_name -> (ts, value)
_DIRTY: set[str] = set()
# Per-subscribed-queue flow stats (message count / rate / staleness). Populated on every push to a
# tapped queue; drained ~10 Hz so the editor can show "is data flowing at the expected rate" even in
# a plain Run (``_STATS_ONLY``), where values are NOT summarized -- only these cheap counters.
_STATS: dict[str, dict[str, float]] = {}  # subscribed_name -> {count, ema_period_ns, last_ns}
# While ANY edge is tapped, track a lightweight rate hint (last push time + EMA period) for EVERY
# flowing queue. Lets a newly-(re)tapped edge report an ESTIMATED rate immediately -- pre-seeded on
# tap activation from this hint -- instead of sitting at "measuring..." for a full producer period
# waiting for the queue's next push. Gated behind an active tap so normal runs pay nothing. Values
# are replaced (not mutated) on each push so a concurrent reader sees a consistent snapshot. Keyed
# by runtime queue name.
_PUSH_HINT: dict[str, dict[str, float]] = {}  # runtime_name -> {last_ns, ema_period_ns}
_STATS_ONLY: bool = False  # plain-Run mode: compute flow stats only, never summarize/send values
_BASE_URL: str = ""
_RUN_TOKEN: str = ""

# ---- pause / step control (Phase B + C) ----
_CTRL_COND = threading.Condition()
_PAUSED: bool = False
_GLOBAL_EPOCH: int = 0
_NODE_EPOCHS: dict[str, int] = {}  # name -> latest server-side step counter
_LAST_GLOBAL: dict[str, int] = {}  # name -> last global epoch this node consumed
_LAST_NODE: dict[str, int] = {}  # name -> last per-node epoch this node consumed
_NODE_PATCHED: bool = False

# ---- soft-reset (server bumps reset_epoch; child reinits node instances) ----
_RESET_EPOCH: int = 0  # latest reset_epoch seen from /control
_LAST_RESET_APPLIED: int = 0  # last reset_epoch we actually applied locally
_NODE_REGISTRY: list[Any] = []  # all Node instances created in this process

# ---- high-level controller on/off (judo-style Play/Pause) ----
# Mirrored from the server's ``/control`` snapshot. Nodes that drive the
# real or simulated robot (notably ``SplineFollower``) read this flag to
# decide whether to dispatch MPC-derived commands or a safe default
# (``DEFAULT_POLICY_CMD``: zero velocity, standing, stowed arm).
_CONTROLLER_RUNNING = threading.Event()

# ---- safe-shutdown request (mirrors server ``shutdown_requested``) ----
# When set, every registered node instance that exposes
# ``request_safe_shutdown()`` is invoked once (idempotently). Currently
# used by :class:`deploy.nodes.sim.mujoco.spot.mujoco_sim.MujocoSimulation` to play a visible
# sit-down animation before the runner sends SIGINT.
_SHUTDOWN_REQUESTED = threading.Event()

# ---- startup health gate (judo's wait_for_healthy_system equivalent) ----
# Tracks per-node readiness so the UI can refuse Play until every
# ``pre_open`` (notably the robot startup sequence inside
# ``SpotControl.start``) has completed. ``Node.open`` is monkey-patched
# to flip ``node.started`` / ``node.init_error``; a background thread
# pings the server with the rolling counts.
_HEALTH_LAST_PUSHED: dict[str, Any] = {}

# ---- master-clock control plane (mirrored from /control) ----
# Read by :class:`judo.unlocked.nodes.master_clock.MasterClock` once per
# tick. ``_CLOCK_STEP_PENDING_NS`` is bumped by the server's Step button
# and drained atomically by the master clock (consume-and-clear).
_CLOCK_LOCK = threading.Lock()
_CLOCK_PAUSED: bool = False
_CLOCK_RATE: float = 1.0
_CLOCK_STEP_PENDING_NS: int = 0
# High-water mark of server-side ``clock_step_total_ns`` we've already
# applied; new server totals minus this give the per-poll delta to add
# to the local pending buffer.
_CLOCK_LAST_TOTAL_NS: int = 0
# Server-authoritative virtual clock, mirrored from /control. ``now_ns``
# derives from these so every process (taps, foxglove, mcap) shares ONE
# timeline that freezes on pause / jumps on step. Same-machine wall clock,
# so ``time.time_ns() - _WALL_ANCHOR_NS`` is valid across processes.
_VIRTUAL_ACCUM_NS: int = 0
_WALL_ANCHOR_NS: int = 0
# Periods (in ns) advertised by every SubordinateTimer at construction time.
# Pushed to the server lazily so the Step button can compute min(period)
# across the configured step group.
_SUBORDINATE_PERIODS_NS: dict[str, int] = {}
_SUBORDINATE_PERIODS_DIRTY: bool = False

# ---- managed-thread tracking (for the "unmanaged threads" debug banner) ----
# A thread is "managed" (i.e. it freezes on Pause) once it passes through the
# scheduler gate (:func:`_gate`) or a pause-aware pacer (``Rate`` /
# ``TimeTracker`` call :func:`mark_managed_thread`). Any live thread that is
# NOT managed -- e.g. a user-spawned ``while True: sleep`` loop that bypasses
# the pacers -- won't stop on Pause, so we surface it to the editor as a
# warning banner: Debug can freeze the diagram but not those threads.
_MANAGED_IDS: set[int] = set()
_THREAD_FIRST_SEEN: dict[int, float] = {}
_UNMANAGED_GRACE_S: float = 3.0  # let a thread reach its gate/pacer before flagging
_PAUSE_POLL_S: float = 0.01  # idle poll interval while a loop is frozen (see freezes_when_paused)


def mark_managed_thread() -> None:
    """Mark the current thread pause-aware (freezes on the debug clock).

    Called by the scheduler gate and by pause-aware pacers (``Rate`` /
    ``TimeTracker``). A no-op unless the tap is installed (a Debug session).
    """
    if _BASE_URL:
        _MANAGED_IDS.add(threading.get_ident())


def _workspace_source(fn: str) -> str:
    """Map a runtime (bazel runfiles) filename back to its editable workspace source path.

    ``bazel run`` exports ``BUILD_WORKSPACE_DIRECTORY``; thread targets are imported from the
    ``*.runfiles/<repo>/`` tree, so strip everything up to and including the repo segment and
    re-root under the workspace so the editor can open the real (editable) file in VS Code.
    Falls back to the raw path when it can't be resolved.
    """
    ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if ws and ".runfiles" in fn:
        after = fn.split(".runfiles", 1)[1].lstrip("/")  # "<repo>/pkg/mod.py"
        rel = after.split("/", 1)[1] if "/" in after else after  # drop the repo segment
        cand = os.path.join(ws, rel)
        if os.path.exists(cand):
            return cand
    return fn


def _code_of(fn: Any) -> Any:
    """Code object of a callable, unwrapping bound methods (``__func__``). ``None`` if not code-backed."""
    return getattr(fn, "__code__", None) or getattr(getattr(fn, "__func__", None), "__code__", None)


def _node_of_thread(t: threading.Thread) -> Any:
    """The scheduler ``Node`` a thread is the worker for, else ``None``.

    Node worker threads run the scheduler's generic ``_node`` wrapper with the ``Node`` as the sole
    positional arg (``Thread(target=_node, args=(node,))``). Reading the Node straight off the thread
    is fork/timing-proof -- unlike matching names against a per-process registry, which is empty in
    workers whose nodes were created before the tap patched ``Node.__init__``.
    """
    args: tuple[Any, ...] = getattr(t, "_args", None) or ()
    if len(args) == 1 and hasattr(args[0], "_target") and hasattr(args[0], "_name"):
        return args[0]
    return None


def _thread_callable(t: threading.Thread) -> Any:
    """The user-meaningful callable a thread runs.

    For a node worker thread, the interesting code is the Node's target, not the scheduler wrapper,
    so unwrap to ``node._target``. Otherwise the thread's own ``_target`` (plain functions / lambdas
    / bound methods) or, for ``Thread`` subclasses, its ``run`` override.
    """
    node = _node_of_thread(t)
    if node is not None:
        return node._target  # noqa: SLF001
    target = getattr(t, "_target", None)
    if target is not None:
        return target
    return getattr(type(t), "run", None)


def _thread_source(t: threading.Thread) -> tuple[str | None, int | None]:
    """Best-effort ``(file, line)`` of a thread's target callable (or its ``run`` override).

    Resolves the code object of the thread's user-meaningful callable (see ``_thread_callable`` --
    node worker threads are unwrapped to the node's target) -- handling plain functions, lambdas and
    bound methods (``__func__``). Returns the workspace-mapped source path + the callable's first
    line so the editor can jump to it.
    """
    code = _code_of(_thread_callable(t))
    if code is None:
        return None, None
    try:
        # ``co_filename`` is typically a bazel-runfiles symlink into the source tree; realpath
        # resolves it to the editable workspace file. If runfiles are *copies* (no symlink), fall
        # back to re-rooting under ``BUILD_WORKSPACE_DIRECTORY``.
        fn = os.path.realpath(code.co_filename)
        if ".runfiles" in fn:
            fn = _workspace_source(fn)
        return fn, int(code.co_firstlineno)
    except Exception:  # pragma: no cover - defensive
        return None, None


def _scan_unmanaged_threads() -> list[dict[str, Any]]:
    """Non-node background threads (threads not in the node diagram), tagged for the editor.

    Each entry is ``{"name", "process", "owner", "file", "line", "managed"}``: ``process`` is this
    worker's scheduler name (``multiprocessing.current_process().name``) so the editor can nest the
    thread in its process box, ``owner`` is the ``module:Class`` of the bound-method target the thread
    runs when resolvable (else ``None``), ``file``/``line`` locate the thread's target callable in the
    workspace so the editor can open it, and ``managed`` is True when the thread is pause-aware (it
    freezes on Pause -- e.g. a bosdyn command-stream thread that calls ``mark_managed_thread`` / a
    ``freezes_when_paused`` pacer). Both managed and unmanaged non-node threads are reported so the
    editor can show every background thread; ``managed`` lets it distinguish "freezes on Pause" from
    "won't freeze on Pause" (a potential rogue).

    Excludes the main thread, debug_tap's own ``unlocked-tap-*`` daemons (expected infrastructure),
    and scheduler **node** worker threads (already shown as nodes) -- detected structurally (the
    ``Node`` is the thread's sole arg), which is fork/timing-proof. A short grace period avoids
    flagging a brand-new thread before it settles at startup.
    """
    import multiprocessing

    proc = multiprocessing.current_process().name
    now = time.time()
    out: list[dict[str, Any]] = []
    for t in threading.enumerate():
        ident = t.ident
        if ident is None or not t.is_alive():
            continue
        name = t.name or ""
        if name == "MainThread" or name.startswith("unlocked-tap-"):
            continue
        # A scheduler node's worker thread is the node itself (already in the diagram); skip it even
        # while idle -- before its first exec marks it managed -- so pills show only non-node threads.
        if _node_of_thread(t) is not None:
            continue
        first = _THREAD_FIRST_SEEN.setdefault(ident, now)
        if now - first < _UNMANAGED_GRACE_S:
            continue
        managed = ident in _MANAGED_IDS  # pause-aware: freezes on Pause (e.g. the command stream)
        self_obj = getattr(_thread_callable(t), "__self__", None)
        owner = f"{type(self_obj).__module__}:{type(self_obj).__qualname__}" if self_obj is not None else None
        file, line = _thread_source(t)
        out.append(
            {
                "name": name or f"thread-{ident}",
                "process": proc,
                "owner": owner,
                "file": file,
                "line": line,
                "managed": managed,
            }
        )
    return sorted(out, key=lambda d: (d["process"], d["name"]))


def freezes_when_paused(pace_method: Any) -> Any:
    """Decorate a loop-pacing method so the loop obeys the debug clock.

    Wraps a pacer's per-iteration pacing call (e.g. ``Rate.sleep`` /
    ``TimeTracker.sleep_remaining``) to:

      1. mark the calling thread pause-aware (so it isn't flagged as an
         unmanaged/rogue thread), and
      2. while the editor's virtual clock is paused, BLOCK so the loop advances
         no work -- this is what freezes background threads (e.g. the mujoco
         physics loop) on Pause, generically, without per-loop code.

    On resume it just calls the wrapped pacer: any fixed-rate pacer already
    resyncs its deadline from *now* when it finds itself behind schedule (a slow
    iteration or, here, the paused gap), so no burst catch-up occurs and no
    per-pacer hook is needed. Entirely a no-op outside a Debug session
    (``is_clock_paused`` is False whenever the tap isn't installed -- a plain
    Run, or a real robot), so it never affects production pacing.
    """

    @wraps(pace_method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        mark_managed_thread()
        while is_clock_paused():
            time.sleep(_PAUSE_POLL_S)
        return pace_method(self, *args, **kwargs)

    return wrapper


def is_clock_paused() -> bool:
    """True iff the master clock should not advance virtual time."""
    return _CLOCK_PAUSED


def get_clock_rate() -> float:
    """Wall-clock multiplier; 1.0 = real time."""
    return _CLOCK_RATE


def now_ns() -> int:
    """Current virtual time in nanoseconds -- the single shared clock.

    Derived from the server-authoritative virtual clock mirrored via
    ``/control``: it advances with wall time while running, FREEZES while
    paused, and JUMPS on Step. Every process computes it locally from the
    same ``(_VIRTUAL_ACCUM_NS, _WALL_ANCHOR_NS, paused, rate)`` so edge-tap
    samples and Foxglove/MCAP log-times share one in-sync timeline.

    Falls back to wall time (``time.time_ns()``) when the tap was never
    installed (e.g. a lightweight Run or running outside the editor) or
    before the first ``/control`` poll has anchored the clock.
    """
    if not _BASE_URL or _WALL_ANCHOR_NS == 0:
        return time.time_ns()
    if _CLOCK_PAUSED:
        return _VIRTUAL_ACCUM_NS
    return _VIRTUAL_ACCUM_NS + int((time.time_ns() - _WALL_ANCHOR_NS) * _CLOCK_RATE)


def now_s() -> float:
    """Current virtual time in seconds (see :func:`now_ns`)."""
    return now_ns() * 1.0e-9


def consume_clock_step_pending_ns() -> int:
    """Atomically read-and-clear any pending step-advance from the server."""
    global _CLOCK_STEP_PENDING_NS
    with _CLOCK_LOCK:
        pending = _CLOCK_STEP_PENDING_NS
        _CLOCK_STEP_PENDING_NS = 0
        return pending


def notify_clock_step_consumed() -> None:
    """Hook for telemetry. No-op for now; callers may extend later."""


def register_subordinate_period(name: str, period_ns: int) -> None:
    """Record a subordinate timer's period for the editor's step group.

    Periods are pushed to the server on the next control-loop poll.
    Safe to call from any process; periods from all processes accumulate.
    """
    global _SUBORDINATE_PERIODS_DIRTY
    with _CLOCK_LOCK:
        if _SUBORDINATE_PERIODS_NS.get(name) == period_ns:
            return
        _SUBORDINATE_PERIODS_NS[name] = period_ns
        _SUBORDINATE_PERIODS_DIRTY = True


def is_controller_running() -> bool:
    """True while the user wants the high-level controller active.

    Safe to call from any node target; returns ``False`` if the tap was
    never installed (e.g. when running outside the editor).
    """
    return _CONTROLLER_RUNNING.is_set()


def is_shutdown_requested() -> bool:
    """True once the user has pressed Stop in the editor.

    Set by the control loop after mirroring ``shutdown_requested`` from
    the server snapshot; consumed by node instances that want to begin a
    visible safe-shutdown sequence before the subprocess receives SIGINT.
    """
    return _SHUTDOWN_REQUESTED.is_set()


# --- per-second tick counter ---------------------------------------------
# Lightweight, per-process diagnostic helper. Each named tag prints one
# `[tag] N/s` stderr line per ~1s window. Disabled by setting
# UNLOCKED_TICK_DIAG=0. Safe to call from any thread.
_TICK_DIAG_ENABLED: bool = os.environ.get("UNLOCKED_TICK_DIAG", "1") != "0"
_TICK_DIAG_STATE: dict[str, tuple[int, int]] = {}
_TICK_DIAG_LOCK = threading.Lock()


def tick_log(tag: str) -> None:
    """Per-second tick-rate logger. Call once per loop iteration."""
    if not _TICK_DIAG_ENABLED:
        return
    now_ns = time.monotonic_ns()
    with _TICK_DIAG_LOCK:
        count, window_start_ns = _TICK_DIAG_STATE.get(tag, (0, now_ns))
        count += 1
        elapsed_ns = now_ns - window_start_ns
        if elapsed_ns >= 1_000_000_000:
            rate = count * 1.0e9 / elapsed_ns
            print(
                f"[tick] {tag} pid={os.getpid()} {rate:.1f}/s",
                file=sys.stderr,
                flush=True,
            )
            _TICK_DIAG_STATE[tag] = (0, now_ns)
        else:
            _TICK_DIAG_STATE[tag] = (count, window_start_ns)


_AMP_DIAG_STATE: dict[str, tuple[float, int]] = {}
_AMP_DIAG_LOCK = threading.Lock()


def amp_log(tag: str, value: float) -> None:
    """Per-second max-magnitude logger. Tracks ``max(value)`` over ~1s windows.

    Useful for measuring whether something that fires every iteration is
    actually changing (e.g. ``max|qpos - qpos_prev|`` to detect a frozen
    sim) without flooding logs.
    """
    if not _TICK_DIAG_ENABLED:
        return
    now_ns = time.monotonic_ns()
    with _AMP_DIAG_LOCK:
        prev_max, window_start_ns = _AMP_DIAG_STATE.get(tag, (0.0, now_ns))
        new_max = max(prev_max, float(value))
        elapsed_ns = now_ns - window_start_ns
        if elapsed_ns >= 1_000_000_000:
            print(
                f"[amp] {tag} pid={os.getpid()} max={new_max:.6g}",
                file=sys.stderr,
                flush=True,
            )
            _AMP_DIAG_STATE[tag] = (0.0, now_ns)
        else:
            _AMP_DIAG_STATE[tag] = (new_max, window_start_ns)


def install() -> None:
    """Activate the tap; idempotent per-process and a no-op if env not set.

    Safe to call multiple times. Each forked worker process must call this
    again because background threads do not survive ``fork()``.
    """
    global _INSTALLED_PID, _PATCHED, _BASE_URL, _RUN_TOKEN, _NODE_PATCHED, _STATS_ONLY
    url = os.environ.get("UNLOCKED_TAP_URL", "").rstrip("/")
    if not url:
        return
    pid = os.getpid()
    if _INSTALLED_PID == pid:
        return
    _BASE_URL = url
    _RUN_TOKEN = os.environ.get("UNLOCKED_TAP_TOKEN", "")
    # Plain-Run stats-only tap: measure flow (count/rate/staleness) for the one selected edge without
    # summarizing values or gating the clock -- a lightweight "is data flowing at the expected rate"
    # probe. Debug runs leave this False and get the full value tap + pause/step.
    _STATS_ONLY = bool(os.environ.get("UNLOCKED_TAP_STATS_ONLY"))
    # Reset shared state inherited from the parent process.
    # (We keep ``_KNOWN_QUEUES`` — those names were created in the parent and
    # are still valid in the forked worker.)
    with _LOCK:
        _TAPPED.clear()
        _NAME_MAP.clear()
        _LATEST.clear()
        _DIRTY.clear()
        _STATS.clear()
        _PUSH_HINT.clear()
    # Re-init the control condition: locks inherited across fork may be in an
    # undefined state. Each pid runs its own gate.
    global _CTRL_COND, _PAUSED, _GLOBAL_EPOCH, _RESET_EPOCH, _LAST_RESET_APPLIED
    global _CLOCK_LOCK, _CLOCK_PAUSED, _CLOCK_RATE, _CLOCK_STEP_PENDING_NS
    global _CLOCK_LAST_TOTAL_NS, _SUBORDINATE_PERIODS_DIRTY
    _CTRL_COND = threading.Condition()
    _PAUSED = False
    _GLOBAL_EPOCH = 0
    _NODE_EPOCHS.clear()
    _LAST_GLOBAL.clear()
    _LAST_NODE.clear()
    _RESET_EPOCH = 0
    _LAST_RESET_APPLIED = 0
    _NODE_REGISTRY.clear()
    _CONTROLLER_RUNNING.clear()
    _SHUTDOWN_REQUESTED.clear()
    # Master-clock mirror: reset to "running, real-time, no pending step"
    # and force re-publish of any subordinate-timer periods registered in
    # this pid so they reach the server after a fork.
    _CLOCK_LOCK = threading.Lock()
    _CLOCK_PAUSED = False
    _CLOCK_RATE = 1.0
    _CLOCK_STEP_PENDING_NS = 0
    _CLOCK_LAST_TOTAL_NS = 0
    _SUBORDINATE_PERIODS_DIRTY = bool(_SUBORDINATE_PERIODS_NS)
    # Late import; this module is loaded from judo.unlocked.__init__ where
    # SharedMemoryQueue is already in sys.modules.
    from judo.unlocked import shmque as _shmque

    if not _PATCHED:
        original_push = _shmque.SharedMemoryQueue.push
        original_init = _shmque.SharedMemoryQueue.__init__
        _seen_pushes: set[str] = set()

        def _tracking_init(self: Any, name: str, *a: Any, **kw: Any) -> Any:
            res = original_init(self, name, *a, **kw)
            try:
                _KNOWN_QUEUES.add(name)
            except Exception:
                pass
            return res

        def _tapped_push(self: Any, value: Any) -> Any:
            # Match on the queue's *logical* name (``{node}_output_{i}``) set by OutputStage, not the
            # per-run hashed shared-memory name (``{hash}_{i}``) which the editor can't derive. Register
            # it in ``_KNOWN_QUEUES`` on first push so ``_refresh_loop`` can match subscriptions to it.
            name = getattr(self, "_logical_name", None) or getattr(self, "_name", None)
            if name:
                if name not in _KNOWN_QUEUES:
                    _KNOWN_QUEUES.add(name)
                # Only do any per-push tap work while at least one edge is actively tapped. In a normal
                # run (nothing tapped) this is a single cheap ``if`` -- ZERO clock reads / dict writes /
                # locking on the hot path. While a tap IS active we maintain a rate hint (last push +
                # EMA period) for every flowing queue so switching/re-tapping an already-flowing edge
                # can pre-seed an estimated rate immediately (see ``_refresh_loop``).
                if _TAPPED:
                    mono = time.perf_counter_ns()
                    hint = _PUSH_HINT.get(name)
                    if hint is None:
                        _PUSH_HINT[name] = {"last_ns": float(mono), "ema_period_ns": 0.0}
                    else:
                        period = mono - hint["last_ns"]
                        ema = hint["ema_period_ns"]
                        if period > 0:
                            ema = float(period) if ema <= 0 else 0.8 * ema + 0.2 * period
                        # Replace (not mutate) so a concurrent reader sees a consistent snapshot.
                        _PUSH_HINT[name] = {"last_ns": float(mono), "ema_period_ns": ema}
                    if name in _TAPPED:
                        subscribed = _NAME_MAP.get(name, name)
                        with _LOCK:
                            # Flow stats: count + EMA inter-arrival period (=> rate). Cheap enough for
                            # the push hot path (a couple of arithmetic ops under the lock).
                            st = _STATS.get(subscribed)
                            if st is None:
                                # Seed the period from the rate hint (previous pushes) so ``hz`` is
                                # available on the FIRST tapped push, not the second.
                                h = _PUSH_HINT.get(name) or {}
                                _STATS[subscribed] = {
                                    "count": 1.0,
                                    "ema_period_ns": float(h.get("ema_period_ns", 0.0)),
                                    "last_ns": float(mono),
                                }
                            else:
                                period = mono - st["last_ns"]
                                st["last_ns"] = float(mono)
                                if period > 0:
                                    st["ema_period_ns"] = (
                                        float(period)
                                        if st["ema_period_ns"] <= 0
                                        else 0.8 * st["ema_period_ns"] + 0.2 * period
                                    )
                                st["count"] += 1.0
                            # Values are only captured/summarized for the FULL (Debug) tap. A plain
                            # Run's stats-only tap never touches the payload -- keeping it lightweight.
                            if not _STATS_ONLY:
                                _LATEST[subscribed] = (now_s(), value)
                            _DIRTY.add(subscribed)
                        if name not in _seen_pushes:
                            _seen_pushes.add(name)
                            import sys as _sys

                            print(
                                f"[debug_tap] pid={os.getpid()} first push to tapped queue: {name} "
                                f"(subscribed as {subscribed}) stats_only={_STATS_ONLY} type={type(value).__name__}",
                                file=_sys.stderr,
                                flush=True,
                            )
            return original_push(self, value)

        _shmque.SharedMemoryQueue.__init__ = _tracking_init  # type: ignore[method-assign]
        _shmque.SharedMemoryQueue.push = _tapped_push  # type: ignore[method-assign, assignment]
        _PATCHED = True

    # Patch Node.exec once per interpreter so pause/step gates every tick.
    # Also patch Node.__init__ to register every Node so the control loop
    # can dispatch soft-resets to each instance's ``reset()`` method.
    if not _NODE_PATCHED:
        try:
            from judo.unlocked.node import Node as _Node

            if not getattr(_Node.exec, "_unlocked_tap_patched", False):
                _orig_exec = _Node.exec

                def _gated_exec(self: Any) -> Any:
                    if not _STATS_ONLY:
                        _gate(self.name)
                    return _orig_exec(self)

                _gated_exec._unlocked_tap_patched = True  # type: ignore[attr-defined]  # noqa: SLF001
                _Node.exec = _gated_exec  # type: ignore[method-assign]
            if not getattr(_Node.__init__, "_unlocked_tap_patched", False):
                _orig_init = _Node.__init__

                def _registering_init(self: Any, *a: Any, **kw: Any) -> Any:
                    res = _orig_init(self, *a, **kw)
                    try:
                        _NODE_REGISTRY.append(self)
                        # Default the readiness flags so a node that has no
                        # ``pre_open`` is considered ready immediately.
                        if not hasattr(self, "_has_pre_open"):
                            self._has_pre_open = getattr(self, "_pre_open", None) is not None
                        if not hasattr(self, "_started"):
                            self._started = not self._has_pre_open
                        if not hasattr(self, "_init_error_str"):
                            self._init_error_str = ""
                        # Mark periodic-timer nodes as pausable. Detected
                        # generically so any graph using a timer (directly
                        # or via ``make_periodic_timer_node``) gets paused
                        # without per-graph configuration. The target may
                        # be either the bare ``PeriodicTimer`` instance
                        # (callable) or a bound method on it.
                        target = getattr(self, "_target", None)
                        candidates = [target, getattr(target, "__self__", None)]
                        # Detect PeriodicTimer by class name/module (duck-typed)
                        # rather than importing
                        # ``judo.unlocked.nodes.periodic_timer`` -- that import
                        # would make ``judo/unlocked`` depend on
                        # ``judo/unlocked/nodes`` (which already depends back on
                        # ``judo/unlocked``), a BUILD dependency cycle. The
                        # concrete timer class is already loaded whenever a graph
                        # uses it, so name matching is sufficient and keeps
                        # detection generic + dependency-free.
                        try:
                            if any(
                                type(c).__name__ == "PeriodicTimer"
                                and "periodic_timer" in (getattr(type(c), "__module__", "") or "")
                                for c in candidates
                                if c is not None
                            ):
                                _PAUSABLE_NAMES.add(getattr(self, "name", ""))
                        except Exception:
                            pass
                    except Exception:
                        pass
                    return res

                _registering_init._unlocked_tap_patched = True  # type: ignore[attr-defined]  # noqa: SLF001
                _Node.__init__ = _registering_init  # type: ignore[method-assign]
            # Patch ``Node.open`` so we can record when each node's
            # ``pre_open`` (the robot startup sequence, etc.) finishes.
            # This is the signal the UI uses to gate Play.
            if not getattr(_Node.open, "_unlocked_tap_patched", False):
                _orig_open = _Node.open

                def _tracking_open(self: Any) -> Any:
                    try:
                        res = _orig_open(self)
                    except BaseException as exc:
                        self._init_error_str = f"{type(exc).__name__}: {exc}"
                        self._started = False
                        raise
                    if getattr(self, "_init_error", None) is not None:
                        self._init_error_str = repr(self._init_error)
                        self._started = False
                    else:
                        self._started = True
                        self._init_error_str = ""
                    return res

                _tracking_open._unlocked_tap_patched = True  # type: ignore[attr-defined]  # noqa: SLF001
                _Node.open = _tracking_open  # type: ignore[method-assign]
            _NODE_PATCHED = True
        except Exception as e:
            import sys as _sys

            print(f"[debug_tap] could not patch Node.exec: {e}", file=_sys.stderr, flush=True)

    _INSTALLED_PID = pid
    _LOG.info("unlocked.debug_tap installed -> %s", _BASE_URL)
    import sys as _sys

    print(f"[debug_tap] installed in pid={pid} url={_BASE_URL}", file=_sys.stderr, flush=True)

    threading.Thread(target=_refresh_loop, name="unlocked-tap-refresh", daemon=True).start()
    threading.Thread(target=_drain_loop, name="unlocked-tap-drain", daemon=True).start()
    threading.Thread(target=_control_loop, name="unlocked-tap-control", daemon=True).start()
    threading.Thread(target=_health_loop, name="unlocked-tap-health", daemon=True).start()


# ---------------------------------------------------------------------------
# Pause / step gate
# ---------------------------------------------------------------------------
# Generic pause strategy: gate every node whose target is a ``PeriodicTimer``.
# Timer ticks drive almost every graph; once they stop, downstream compute /
# publish / control nodes naturally idle on their input wait. This needs zero
# per-graph configuration and works identically for mujoco (where physics
# step + state/mocap publishers all hang off timers) and hardware (where the
# controller + low-level loops are timer-driven; the blocking state-streaming
# and mocap callbacks keep running, which is what you want for a real robot
# anyway).
#
# Extra non-timer nodes can be added via ``UNLOCKED_PAUSE_NODES`` (comma-
# separated substrings) for graphs that need to freeze specific subscribers.
_PAUSE_NODE_SUBSTRINGS: tuple[str, ...] = tuple(
    s.strip() for s in os.environ.get("UNLOCKED_PAUSE_NODES", "").split(",") if s.strip()
)
_PAUSABLE_NAMES: set[str] = set()  # populated at Node construction (timer detection)


# Nodes whose name matches one of these substrings are NEVER paused, even
# if they would otherwise be detected as pausable (e.g. a viser
# timer). Lets read-only UI / inspectors keep ticking so the timeline
# scrubber can stream past states to viser while the rest of the graph is
# frozen. Override / extend with ``UNLOCKED_NEVER_PAUSE_NODES``.
_NEVER_PAUSE_SUBSTRINGS: tuple[str, ...] = tuple(
    s.strip() for s in os.environ.get("UNLOCKED_NEVER_PAUSE_NODES", "viser").split(",") if s.strip()
)


def _is_pausable(name: str) -> bool:
    if any(s in name for s in _NEVER_PAUSE_SUBSTRINGS):
        return False
    if name in _PAUSABLE_NAMES:
        return True
    return any(s in name for s in _PAUSE_NODE_SUBSTRINGS)


def _gate(name: str) -> None:
    """Block while paused, but only for nodes on the controller output path.

    Everything else (state, mocap, sim, MPC, viser) ticks normally so the user can keep inspecting
    the live world while paused.

    A step permit (``server_epoch > last_local_epoch``) bypasses the gate
    exactly once so Step works on the gated nodes too.
    """
    # Every node thread reaches here each tick: mark it managed so it is not
    # mistaken for a rogue (unmanaged) thread in the debug banner. Do this
    # before the pausability early-return so non-pausable scheduler nodes
    # (viser, etc.) also count as managed -- they're under scheduler control.
    _MANAGED_IDS.add(threading.get_ident())
    if not _is_pausable(name):
        return
    with _CTRL_COND:
        while _PAUSED:
            # Never let the pause gate block a node's exit during shutdown: if we're shutting down,
            # release immediately so the scheduler can join this node's thread (else Stop hangs on a
            # paused graph until the SIGKILL escalation).
            if _SHUTDOWN_REQUESTED.is_set():
                return
            ge = _GLOBAL_EPOCH
            if ge > _LAST_GLOBAL.get(name, 0):
                _LAST_GLOBAL[name] = ge
                return
            ne = _NODE_EPOCHS.get(name, 0)
            if ne > _LAST_NODE.get(name, 0):
                _LAST_NODE[name] = ne
                return
            _CTRL_COND.wait(timeout=1.0)


def _control_loop() -> None:
    """Poll ``/control`` and mirror pause/step state into the gate."""
    global _PAUSED, _GLOBAL_EPOCH, _RESET_EPOCH
    global _CLOCK_PAUSED, _CLOCK_RATE, _CLOCK_STEP_PENDING_NS, _CLOCK_LAST_TOTAL_NS
    global _VIRTUAL_ACCUM_NS, _WALL_ANCHOR_NS
    global _SUBORDINATE_PERIODS_DIRTY
    url = f"{_BASE_URL}/control"
    periods_url = f"{_BASE_URL}/control/clock/periods"
    headers = {"X-Run-Token": _RUN_TOKEN} if _RUN_TOKEN else {}
    last_version = -1
    while True:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=2) as resp:
                snap = json.loads(resp.read().decode("utf-8"))
            version = int(snap.get("version", 0))
            # Master-clock mirror is unconditional: pause/rate need to be
            # reflected on every poll, not just on version-bumps, so a
            # missed snapshot doesn't strand the clock in the wrong state.
            new_clock_paused = bool(snap.get("clock_paused", snap.get("paused", False)))
            new_clock_rate = float(snap.get("clock_rate", 1.0))
            new_step_total = int(snap.get("clock_step_total_ns", 0))
            new_virtual_accum = int(snap.get("virtual_accum_ns", 0))
            new_wall_anchor = int(snap.get("wall_anchor_ns", 0))
            with _CLOCK_LOCK:
                _CLOCK_PAUSED = new_clock_paused
                _CLOCK_RATE = new_clock_rate
                # Mirror the server-authoritative virtual clock so ``now_ns``
                # returns the same timeline in every process.
                _VIRTUAL_ACCUM_NS = new_virtual_accum
                _WALL_ANCHOR_NS = new_wall_anchor
                # Diff the server's monotone counter against our high-water
                # mark; any positive delta is a new step we should apply.
                delta = new_step_total - _CLOCK_LAST_TOTAL_NS
                if delta > 0:
                    _CLOCK_STEP_PENDING_NS += delta
                    _CLOCK_LAST_TOTAL_NS = new_step_total
                elif new_step_total < _CLOCK_LAST_TOTAL_NS:
                    # Server reset (e.g. new run); resync.
                    _CLOCK_LAST_TOTAL_NS = new_step_total
            if version != last_version:
                last_version = version
                pending_ge = int(snap.get("global_epoch", 0))
                new_reset_epoch = int(snap.get("reset_epoch", 0))
                reset_pending = new_reset_epoch > _LAST_RESET_APPLIED
                if bool(snap.get("controller_running", False)):
                    _CONTROLLER_RUNNING.set()
                else:
                    _CONTROLLER_RUNNING.clear()
                shutdown_requested_now = bool(snap.get("shutdown_requested", False))
                if shutdown_requested_now and not _SHUTDOWN_REQUESTED.is_set():
                    _SHUTDOWN_REQUESTED.set()
                    _dispatch_safe_shutdown()
                elif not shutdown_requested_now:
                    _SHUTDOWN_REQUESTED.clear()
                with _CTRL_COND:
                    _PAUSED = bool(snap.get("paused", False))
                    _NODE_EPOCHS.clear()
                    _NODE_EPOCHS.update({str(k): int(v) for k, v in (snap.get("node_epochs") or {}).items()})
                    _RESET_EPOCH = new_reset_epoch
                    # Defer the step-permit (``global_epoch``) bump if a
                    # reset is pending: granting permits before the reset
                    # applies would let nodes tick once on the stale state.
                    if not reset_pending:
                        _GLOBAL_EPOCH = pending_ge
                    _CTRL_COND.notify_all()
                if reset_pending:
                    _maybe_apply_reset()
                    # Now safe to publish the deferred step permit so each
                    # node ticks exactly once on the freshly-reset state.
                    with _CTRL_COND:
                        _GLOBAL_EPOCH = pending_ge
                        _CTRL_COND.notify_all()
            # Push any newly-registered subordinate-timer periods to the
            # server so the Step button can compute min(period) across the
            # configured step group.
            with _CLOCK_LOCK:
                if _SUBORDINATE_PERIODS_DIRTY:
                    payload = dict(_SUBORDINATE_PERIODS_NS)
                    _SUBORDINATE_PERIODS_DIRTY = False
                else:
                    payload = None
            if payload is not None:
                try:
                    body = json.dumps({"periods_ns": payload}).encode("utf-8")
                    req = urllib.request.Request(
                        periods_url, data=body, headers={**headers, "Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(req, timeout=2).read()
                except Exception:
                    # Server may be older or briefly unavailable; periods are
                    # re-pushed next poll since we leave the dict populated
                    # in memory.
                    with _CLOCK_LOCK:
                        _SUBORDINATE_PERIODS_DIRTY = True
        except Exception:
            pass
        time.sleep(0.1)


def _collect_health() -> dict[str, Any]:
    """Snapshot per-node readiness for the UI health gate.

    Also rolls up ``shutdown_ready``: True if every registered node
    instance that exposes ``is_shutdown_ready()`` reports True. Pids
    with no such instances vacuously report True (so the server's
    cross-pid aggregate only blocks on pids that own a robot-like node).
    """
    total = 0
    started = 0
    errors: list[dict[str, str]] = []
    pending: list[str] = []
    for n in list(_NODE_REGISTRY):
        if not getattr(n, "_has_pre_open", False):
            continue
        total += 1
        if getattr(n, "_started", False):
            started += 1
            continue
        err = getattr(n, "_init_error_str", "") or ""
        if err:
            errors.append({"name": getattr(n, "name", "?"), "error": err})
        else:
            pending.append(getattr(n, "name", "?"))
    shutdown_ready = True
    seen: set[int] = set()
    for n in list(_NODE_REGISTRY):
        target = getattr(n, "_target", None)
        inst = getattr(target, "__self__", None)
        if inst is None:
            continue
        key = id(inst)
        if key in seen:
            continue
        seen.add(key)
        ready_fn = getattr(inst, "is_shutdown_ready", None)
        if not callable(ready_fn):
            continue
        try:
            if not bool(ready_fn()):
                shutdown_ready = False
                break
        except Exception:
            shutdown_ready = False
            break
    return {
        "ready": started == total,
        "started": started,
        "total": total,
        "pending": pending[:16],  # cap; the UI just needs a hint
        "errors": errors[:16],
        "shutdown_ready": shutdown_ready,
        "unmanaged_threads": _scan_unmanaged_threads(),
    }


def _dispatch_safe_shutdown() -> None:
    """Call ``request_safe_shutdown()`` on each unique node instance.

    Runs once on the rising edge of ``shutdown_requested`` in the
    snapshot. Idempotent on the receiving side
    (:meth:`MujocoSimulation.request_safe_shutdown` is a no-op outside
    the ``running`` phase).
    """
    import sys as _sys

    seen: set[int] = set()
    for node in list(_NODE_REGISTRY):
        target = getattr(node, "_target", None)
        inst = getattr(target, "__self__", None)
        if inst is None:
            continue
        key = id(inst)
        if key in seen:
            continue
        seen.add(key)
        fn = getattr(inst, "request_safe_shutdown", None)
        if not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:
            print(
                f"[debug_tap] request_safe_shutdown on {type(inst).__name__} " f"raised: {exc}",
                file=_sys.stderr,
                flush=True,
            )


def _health_loop() -> None:
    """Push per-node readiness to the server on change.

    The server aggregates this into a ``health`` block on its ``/control``
    snapshot so the UI can show "starting (N/M ready)" and block Play
    until every ``pre_open`` (the robot startup sequence on hardware) has
    completed.
    """
    url = f"{_BASE_URL}/control/health"
    headers = {"Content-Type": "application/json"}
    if _RUN_TOKEN:
        headers["X-Run-Token"] = _RUN_TOKEN
    pid = os.getpid()
    last: dict[str, Any] = {}
    # POST an initial snapshot immediately (even if every count is 0) so the
    # server learns this pid exists before any pre_open completes. Without
    # this, the aggregated ``ready`` flag can race: another process may
    # report ``ready=True`` (its single pre_open finished fast) before this
    # process has registered its own nodes, and the UI briefly shows the
    # whole system as ready.
    first = True
    while True:
        try:
            snap = _collect_health()
            if first or snap != last:
                body = json.dumps({"pid": pid, **snap}).encode("utf-8")
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    resp.read()
                last = snap
                first = False
        except Exception:
            pass
        time.sleep(0.5)


def _maybe_apply_reset() -> None:
    """Call ``reset()`` on each unique node instance that exposes one, if ``reset_epoch`` advanced.

    Runs without requiring the global pause: on hardware we can't safely
    freeze the graph (the joint-control stream would time out). Each
    ``reset()`` implementation is responsible for its own concurrency
    (e.g. ``MujocoSimulation.reset`` takes its state + control locks).
    """
    global _LAST_RESET_APPLIED
    if _RESET_EPOCH <= _LAST_RESET_APPLIED:
        return
    import sys as _sys

    seen: set[int] = set()
    for node in list(_NODE_REGISTRY):
        target = getattr(node, "_target", None)
        inst = getattr(target, "__self__", None)
        if inst is None:
            continue
        key = id(inst)
        if key in seen:
            continue
        seen.add(key)
        reset_fn = getattr(inst, "reset", None)
        if not callable(reset_fn):
            continue
        try:
            reset_fn()
        except Exception as exc:
            print(f"[debug_tap] reset() on {type(inst).__name__} raised: {exc}", file=_sys.stderr, flush=True)
    _LAST_RESET_APPLIED = _RESET_EPOCH
    print(
        f"[debug_tap] applied reset_epoch={_RESET_EPOCH} across {len(seen)} instance(s)", file=_sys.stderr, flush=True
    )


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------
def _refresh_loop() -> None:
    url = f"{_BASE_URL}/tap/queues"
    headers = {"X-Run-Token": _RUN_TOKEN} if _RUN_TOKEN else {}
    while True:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            subscribed = set(data.get("queues") or [])
            # Expand each subscribed (logical) name to all local runtime queue
            # names that "match". A logical name like ``spot_image_output_4``
            # may correspond to a runtime queue like
            # ``mujoco_simulation_5_spot_image_publisher_output_4`` when a
            # builder renames the node. We match by:
            #   * exact name, or
            #   * runtime queue ends with ``_<short_name>_output_<i>`` where
            #     ``short_name`` is built from the subscribed name's stem.
            local_to_sub: dict[str, str] = {}
            local_tapped: set[str] = set()
            known_all = list(_KNOWN_QUEUES)  # snapshot: _tapped_push mutates this from other threads
            for sub in subscribed:
                # Exact match wins and is EXCLUSIVE: queues now carry an exact logical name
                # (``{node}_output_{i}``), so if one matches ``sub`` exactly we must NOT also fuzzy-
                # match sibling queues (e.g. ``foxglove_spot_state_subscriber_output_0`` for
                # ``spot_state_output_0``). Doing so mapped two different queues to the same
                # subscription, so their values overwrote each other in ``_LATEST`` and the inspector
                # flip-flopped between them (spot_state <-> a joint_states dict).
                if sub in known_all:
                    local_to_sub[sub] = sub
                    local_tapped.add(sub)
                    continue
                # No exact match -> fall back to fuzzy suffix match (handles builder-renamed queues,
                # e.g. spot_image_output_4 → *_spot_image*_output_4).
                m = sub.rsplit("_output_", 1)
                if len(m) == 2:
                    stem, idx = m
                    suffix1 = f"_{stem}_output_{idx}"  # exact stem
                    suffix2 = f"_{stem}_publisher_output_{idx}"  # builder *_publisher
                    suffix3 = f"_{stem}_subscriber_output_{idx}"
                    end_idx = f"_output_{idx}"
                    for known in known_all:
                        if known in local_to_sub:
                            continue
                        if known.endswith((suffix1, suffix2, suffix3)):
                            local_to_sub[known] = sub
                            local_tapped.add(known)
                        elif known.endswith(end_idx) and stem in known:
                            local_to_sub[known] = sub
                            local_tapped.add(known)
            with _LOCK:
                prev = set(_TAPPED)
                _TAPPED.clear()
                _TAPPED.update(local_tapped)
                _NAME_MAP.clear()
                _NAME_MAP.update(local_to_sub)
                # Pre-seed flow stats for NEWLY-tapped queues from the rate hint (accumulated while
                # other edges were tapped). This makes a switched/re-tapped edge report an ESTIMATED
                # rate within one drain cycle (~0.1 s) instead of sitting at "measuring..." for a full
                # producer period until its next push. Real pushes then refine the estimate.
                for local in local_tapped:
                    sub = local_to_sub.get(local, local)
                    if sub in _STATS:
                        continue
                    h = _PUSH_HINT.get(local)
                    if h and h.get("ema_period_ns", 0.0) > 0:
                        _STATS[sub] = {
                            "count": 0.0,
                            "ema_period_ns": float(h["ema_period_ns"]),
                            "last_ns": float(h["last_ns"]),
                        }
            if local_tapped != prev:
                import sys as _sys

                summary = [f"{k}→{v}" if k != v else k for k, v in sorted(local_to_sub.items())]
                print(
                    f"[debug_tap] pid={os.getpid()} subscribed={sorted(subscribed)} matched={summary}",
                    file=_sys.stderr,
                    flush=True,
                )
            with _LOCK:
                # Drop slots no longer subscribed (keyed by the *subscribed* name). This includes the
                # flow stats: if we kept a stale ``_STATS`` entry, re-tapping that edge later would
                # resume from its old ``count``/``last_ns`` -- yielding a wrong (frozen-looking) rate
                # and a count that never resets. Clearing here means every (re)tap measures fresh.
                for k in list(_LATEST.keys()):
                    if k not in subscribed:
                        _LATEST.pop(k, None)
                        _DIRTY.discard(k)
                for k in list(_STATS.keys()):
                    if k not in subscribed:
                        _STATS.pop(k, None)
        except Exception:
            pass
        # Poll the subscription frequently so a freshly-tapped edge starts reporting almost
        # immediately (was 1 s -> up to a second of "no stats yet" after selecting an edge). The
        # request is tiny (just the subscribed-name list), so 10 Hz per worker is cheap.
        time.sleep(0.1)


def _drain_loop() -> None:
    url = f"{_BASE_URL}/tap/ingest"
    headers = {"Content-Type": "application/json"}
    if _RUN_TOKEN:
        headers["X-Run-Token"] = _RUN_TOKEN
    while True:
        time.sleep(0.1)
        # Snapshot under the lock, then release BEFORE summarizing. ``_summarize`` can be slow
        # (serializing a big struct/image) and the producer's ``_tapped_push`` grabs the same lock on
        # every push -- holding it across ``_summarize`` stalls high-rate producers. We send STATS for
        # every tapped queue each cycle (so staleness/age keeps growing when a queue stalls -- that's
        # how the editor shows "data blocked"), plus the summarized VALUE for freshly-pushed queues in
        # the full (Debug) tap. Best-effort: a value mutated mid-summarize just yields a torn sample.
        now_mono = time.perf_counter_ns()
        with _LOCK:
            stats_snap = {name: dict(st) for name, st in _STATS.items()}
            value_snap = {} if _STATS_ONLY else {name: _LATEST.get(name) for name in _DIRTY}
            _DIRTY.clear()
        if not stats_snap and not value_snap:
            continue
        batch: list[dict[str, Any]] = []
        names = set(stats_snap) | set(value_snap)
        for name in names:
            st = stats_snap.get(name)
            stats_out = None
            if st is not None:
                ema = st.get("ema_period_ns", 0.0)
                stats_out = {
                    "count": int(st.get("count", 0)),
                    "hz": (1.0e9 / ema) if ema > 0 else None,
                    "age_s": max(0.0, (now_mono - st.get("last_ns", now_mono)) / 1.0e9),
                }
            slot = value_snap.get(name)
            ts = None
            summary = None
            if slot is not None:
                ts, value = slot
                try:
                    summary = _summarize(value)
                except Exception as e:
                    summary = {
                        "kind": "text",
                        "value": f"<summarize error for {type(value).__name__}: {e}>",
                        "type": type(value).__name__,
                    }
            batch.append({"queue": name, "ts": ts if ts is not None else now_s(), "value": summary, "stats": stats_out})
        if not batch:
            continue
        body = json.dumps({"samples": batch}).encode("utf-8")
        # Retry once: under concurrent ingest fan-in the editor can reset a connection; a quick
        # second attempt usually lands. High-rate queues also re-mark themselves dirty on the next
        # push, so a fully-dropped batch self-heals within ~100 ms and isn't worth spamming about.
        err: Exception | None = None
        for _attempt in range(2):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()
                err = None
                break
            except Exception as e:  # - best-effort telemetry
                err = e
                time.sleep(0.05)
        if err is not None:
            _log_ingest_failure(err, batch)


_INGEST_ERR_LAST_LOG: float = 0.0
_INGEST_ERR_SUPPRESSED: int = 0


def _log_ingest_failure(err: Exception, batch: list[dict[str, Any]]) -> None:
    """Log an ingest failure at most once every few seconds (these are transient + self-healing)."""
    global _INGEST_ERR_LAST_LOG, _INGEST_ERR_SUPPRESSED
    import sys as _sys

    now = time.time()
    if now - _INGEST_ERR_LAST_LOG < 5.0:
        _INGEST_ERR_SUPPRESSED += 1
        return
    extra = f" (+{_INGEST_ERR_SUPPRESSED} more suppressed)" if _INGEST_ERR_SUPPRESSED else ""
    _INGEST_ERR_SUPPRESSED = 0
    _INGEST_ERR_LAST_LOG = now
    print(
        f"[debug_tap] pid={os.getpid()} ingest failed (transient, retried): {type(err).__name__}: {err}"
        f" — {len(batch)} sample(s) dropped, will resend on next push{extra}",
        file=_sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Value summarization
# ---------------------------------------------------------------------------
_VECTOR_MAX = 256
_TEXT_MAX = 4096


def _summarize(value: Any) -> dict[str, Any]:
    """Turn an arbitrary value into a renderer-friendly JSON payload."""
    # numpy is optional; everything below degrades gracefully.
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None  # type: ignore[assignment]

    if np is not None and isinstance(value, np.ndarray):
        return _summarize_ndarray(value, np)

    # Bosdyn / protobuf image wrappers: decode raw bytes into a numpy image.
    img_summary = _maybe_summarize_protobuf_image(value, np)
    if img_summary is not None:
        return img_summary

    if isinstance(value, (bool, int, float)):
        return {"kind": "scalar", "value": value, "type": type(value).__name__}
    if isinstance(value, str):
        return {"kind": "text", "value": value[:_TEXT_MAX], "len": len(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        b = bytes(value)
        return {"kind": "bytes", "size": len(b), "head": base64.b64encode(b[:64]).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return _summarize_list(value, np)
    if isinstance(value, dict):
        return _summarize_dict(value)
    # Dataclass fallback: shallow-walk fields (asdict deepcopies and breaks on
    # numpy / SE3Pose / Timestamp).
    if hasattr(value, "__dataclass_fields__"):
        out: dict[str, Any] = {}
        for field_name in value.__dataclass_fields__:
            try:
                out[field_name] = _summarize(getattr(value, field_name))
            except Exception as e:
                out[field_name] = {"kind": "text", "value": f"<error: {e}>"}
        return {"kind": "struct", "type": type(value).__name__, "fields": out}
    # Generic object: shallow-walk public instance attributes so a custom value type (e.g. an SE2/
    # SE3 pose wrapper) serializes its DATA rather than a ``<Class at 0x...>`` repr. That repr's
    # address changes every time the producer re-creates the object (e.g. the state node re-pushing a
    # frozen pose while Paused), which looks like the value is updating when it isn't.
    obj_dict = getattr(value, "__dict__", None)
    if isinstance(obj_dict, dict) and obj_dict:
        out = {}
        for k, v in list(obj_dict.items())[:64]:
            if k.startswith("_"):
                continue
            try:
                out[k] = _summarize(v)
            except Exception as e:
                out[k] = {"kind": "text", "value": f"<error: {e}>"}
        if out:
            return {"kind": "struct", "type": type(value).__name__, "fields": out}
    # Last resort: repr.
    try:
        text = repr(value)
    except Exception as e:
        text = f"<unrepr-able {type(value).__name__}: {e}>"
    return {"kind": "text", "value": text[:_TEXT_MAX], "type": type(value).__name__}


def _maybe_summarize_protobuf_image(value: Any, np: Any) -> dict[str, Any] | None:
    """Detect Bosdyn ImageCapture/ImageResponse/ImageCaptureAndSource and decode."""
    if np is None:
        return None
    # Walk down common wrappers to the inner Image message.
    img = None
    pixel_format = None
    for attr_chain in (("shot", "image"), ("image",), ()):
        obj = value
        ok = True
        for a in attr_chain:
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if ok and hasattr(obj, "data") and hasattr(obj, "rows") and hasattr(obj, "cols"):
            img = obj
            pixel_format = getattr(img, "pixel_format", None)
            break
    if img is None or not getattr(img, "data", b"") or img.rows <= 0 or img.cols <= 0:
        return None
    # Bosdyn PIXEL_FORMAT enum (subset):
    #   1=GREYSCALE_U8, 2=GREYSCALE_U16, 3=RGB_U8, 4=RGBA_U8, 5=DEPTH_U16,
    #   6=GREYSCALE_F32. Default to U8 grayscale if unknown.
    fmt_to_dtype_channels = {
        1: (np.uint8, 1),
        2: (np.uint16, 1),
        3: (np.uint8, 3),
        4: (np.uint8, 4),
        5: (np.uint16, 1),
        6: (np.float32, 1),
    }
    dtype, channels = fmt_to_dtype_channels.get(int(pixel_format) if pixel_format is not None else 1, (np.uint8, 1))
    try:
        arr = np.frombuffer(img.data, dtype=dtype)
        if channels == 1:
            arr = arr.reshape(img.rows, img.cols)
        else:
            arr = arr.reshape(img.rows, img.cols, channels)
    except (ValueError, TypeError):
        return None
    return _summarize_ndarray(arr, np)


def _summarize_dict(value: dict) -> dict[str, Any]:
    try:
        return {"kind": "json", "value": json.loads(json.dumps(value, default=_json_default))}
    except Exception:
        return {"kind": "text", "value": repr(value)[:_TEXT_MAX]}


def _summarize_list(value: list | tuple, np: Any) -> dict[str, Any]:
    # Promote homogeneous numeric lists to a vector for nicer rendering.
    if np is not None and value and all(isinstance(x, (int, float)) for x in value):
        return _summarize_ndarray(np.asarray(value, dtype=float), np)
    try:
        return {"kind": "json", "value": json.loads(json.dumps(list(value), default=_json_default))}
    except Exception:
        return {"kind": "text", "value": repr(value)[:_TEXT_MAX]}


def _summarize_ndarray(arr: Any, np: Any) -> dict[str, Any]:
    shape = list(arr.shape)
    dtype = str(arr.dtype)
    # Image: HxW or HxWxC (C in {1, 3, 4}). Encode to PNG.
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (1, 3, 4)):
        try:
            png = _ndarray_to_png(arr, np)
        except Exception as e:
            return {"kind": "text", "value": f"<image encode failed: {e}>", "shape": shape, "dtype": dtype}
        return {"kind": "image", "shape": shape, "dtype": dtype, "png": png}
    # Vector: 1-D numeric.
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
        n = int(arr.shape[0])
        head = arr[:_VECTOR_MAX].astype(float).tolist()
        stats = {
            "min": float(arr.min()) if n else 0.0,
            "max": float(arr.max()) if n else 0.0,
            "mean": float(arr.mean()) if n else 0.0,
            "n": n,
        }
        return {"kind": "vector", "shape": shape, "dtype": dtype, "values": head, "stats": stats}
    # Fallback: summary stats only.
    flat = arr.ravel()
    return {
        "kind": "ndarray",
        "shape": shape,
        "dtype": dtype,
        "stats": {
            "min": float(flat.min()) if flat.size else 0.0,
            "max": float(flat.max()) if flat.size else 0.0,
            "mean": float(flat.mean()) if flat.size else 0.0,
            "n": int(flat.size),
        },
    }


def _json_default(o: Any) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(o, np.ndarray):
            if o.size <= _VECTOR_MAX:
                return o.tolist()
            return {"__ndarray__": True, "shape": list(o.shape), "dtype": str(o.dtype)}
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
    except ImportError:
        pass
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(o)
    return repr(o)


# ---------------------------------------------------------------------------
# Tiny PNG encoder (stdlib only) -- supports L (HxW), RGB (HxWx3), RGBA (HxWx4).
# Float images are min/max normalized to uint8 grayscale.
# ---------------------------------------------------------------------------
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _ndarray_to_png(arr: Any, np: Any) -> str:
    """Encode a 2-D / 3-D numpy image as PNG and return base64 string."""
    a = arr
    # Normalize floats / non-uint8 to uint8.
    if a.dtype != np.uint8:
        a = a.astype(float)
        if a.size:
            lo = float(a.min())
            hi = float(a.max())
            if hi > lo:
                a = (a - lo) / (hi - lo) * 255.0
            else:
                a = a * 0.0
        a = a.clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        h, w = a.shape
        channels = 1
        color_type = 0  # grayscale
        flat = a
    else:
        h, w, c = a.shape
        if c == 1:
            channels, color_type = 1, 0
            flat = a[..., 0]
        elif c == 3:
            channels, color_type = 3, 2
            flat = a
        elif c == 4:
            channels, color_type = 4, 6
            flat = a
        else:
            raise ValueError(f"unsupported channel count {c}")
    # Cap large images to keep payload small. Use the same stride on both
    # axes so the downsampled image preserves the source aspect ratio.
    max_dim = 384
    if h > max_dim or w > max_dim:
        step = max(1, max(h, w) // max_dim)
        flat = flat[::step, ::step]
        h, w = flat.shape[:2]

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    # Each scanline prefixed with a filter byte (0 = None).
    row_bytes = w * channels
    if channels == 1:
        rows = flat.tobytes()
    else:
        rows = flat.reshape(h, row_bytes).tobytes()
    raw = b"".join(b"\x00" + rows[i * row_bytes : (i + 1) * row_bytes] for i in range(h))
    idat = zlib.compress(raw, 6)
    blob = _PNG_SIG + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
    return base64.b64encode(blob).decode("ascii")
