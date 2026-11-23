# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import gc
import logging
import multiprocessing as mp
import os
import signal
import sys
from threading import Thread
from time import sleep
from typing import Any, Iterable

from judo.unlocked.node import Node, NodeType

_FORK_SAFETY_ENV = "OBJC_DISABLE_INITIALIZE_FORK_SAFETY"


def _ensure_objc_fork_safety_optout() -> None:
    """Guarantee ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`` is in effect on macOS, re-exec'ing if not.

    This scheduler forks its workers (see ``Threaded``), and on macOS a forked child that touches an
    Apple framework aborts with "... may have been in progress in another thread when fork() was
    called" unless the opt-out is set. The Obj-C runtime reads this variable once, in ``_objc_init``
    when ``libobjc`` loads -- i.e. before ``main()``. Setting ``os.environ`` from Python is therefore
    too late to have any effect; the value must already be in the environment the process was
    *launched* with. So if it is missing we set it and re-exec this interpreter, which is the only
    way to make it stick without depending on every launcher (``bazel run``, ``//app:ui``, a plain
    ``python`` invocation) to have exported it.
    """
    if sys.platform == "darwin":
        if os.environ.get(_FORK_SAFETY_ENV):
            return
        # Under pytest the launcher (``--test_env``) owns the environment; re-exec'ing would restart the
        # whole test session.
        if "pytest" in sys.modules:
            os.environ[_FORK_SAFETY_ENV] = "YES"
            return
        os.environ[_FORK_SAFETY_ENV] = "YES"
        # ``sys.orig_argv`` is the full interpreter command line (executable + flags + script + args),
        # so this reproduces the launch exactly. The variable is now set, so the new image returns early
        # above and cannot loop.
        os.execv(sys.executable, sys.orig_argv)


_ensure_objc_fork_safety_optout()


def _install_parent_death_signal() -> None:
    """Ask the kernel to deliver SIGTERM to this process when its parent dies.

    Linux only. Guards every scheduler child against being orphaned when the
    parent ``//judo/unlocked/graph:runner`` is force-killed (SIGKILL, terminal close, etc.) —
    without this the children survive as zombies still bound to viser/foxglove
    ports.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        # PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            logging.debug("prctl(PR_SET_PDEATHSIG) failed: errno=%d", ctypes.get_errno())
    except Exception as exc:
        logging.debug("prctl unavailable: %s", exc)


def _node(node: Node) -> None:
    """Worker loop for a single node thread.

    Repeats wait/exec while the node stays live. ``wait`` blocks with no timeout
    -- an untimed condition wait is an instant kernel wake, whereas a timed one
    degrades to a ~5 ms polling loop on macOS (see SharedMemoryQueueView.wait).
    Because the wait never times out on its own, the scheduler releases it on
    shutdown: spin() marks the node not-live and then calls node.wake(), which
    unblocks this loop so it observes node.live is False and exits (letting spin
    join it before the stages are torn down).
    """
    while node.live:
        if node.wait():
            node.exec()


class Threaded:
    """Process-isolated scheduler that runs one thread per node.

    Concurrency model:
    - One daemon process per scheduler instance
    - One daemon thread per live node inside that process, EXCEPT nodes marked
      a non-default ``NodeType``, which are driven externally (``Node.callback`` /
      ``Node.generator``) and get no thread -- see :meth:`_start_node_threads`
    - Each node thread runs: wait -> exec -> wait -> ...

    Lifecycle model:
    - Start: open output stages, open input stages, run node open callbacks, start threads
    - Run: monitor stop event and node liveness
    - Stop: close nodes, then close input stages, then close output stages

    Lifecycle applies to every member node regardless of ``node_type``, so an externally
    driven node still gets its stages opened, its ``warmup`` published, and its
    ``post_close`` run in ``close_order``.

    Synchronization note: Scheduler orchestrates execution but does not synchronize data
    correspondence across independent input queues; that remains a node/application concern.
    """

    def __init__(
        self,
        name: str,
        nodes: Iterable[Node],
    ):
        self._name = name
        self._nodes = list(nodes)
        # This scheduler is designed around ``fork``: workers inherit the fully
        # materialized graph (nodes hold un-picklable ``_thread.lock`` handles, ONNX
        # sessions, robot clients, shared-memory views) instead of receiving it via
        # pickle. macOS (and Python >=3.14 on Linux) default the start method to
        # ``spawn``, which would pickle ``self.exec`` -> the whole graph and fail with
        # "cannot pickle '_thread.lock' object". Pin a ``fork`` context so behavior is
        # identical across platforms.
        ctx = mp.get_context("fork")
        self._proc = ctx.Process(target=self.exec, name=name, daemon=True)
        self._stop = ctx.Event()
        self._shutdown_clean = ctx.Event()

    def stop(self) -> None:
        """Request scheduler shutdown via process-shared stop event."""
        self._stop.set()

    @property
    def name(self) -> str:
        """The scheduler name."""
        return self._name

    def start(self) -> None:
        """Start scheduler process asynchronously."""
        self._proc.start()

    def join(self, timeout: float | None = None) -> None:
        """Block until scheduler process exits."""
        self._proc.join(timeout)

    @property
    def alive(self) -> bool:
        """Whether the underlying process is still running."""
        return self._proc.is_alive()

    @property
    def shutdown_clean(self) -> bool:
        """Whether orderly cleanup finished, all workers joined, and no lifecycle hook failed.

        Shared with the parent process. This is not a guarantee of error-free execution:
        only node ``pre_open``/``post_close`` errors are tracked by ``lifecycle_error``.
        The parent must also check process exit status and any shutdown escalation.
        """
        return self._shutdown_clean.is_set()

    @property
    def exitcode(self) -> int | None:
        """Exit code of the underlying process, or ``None`` while it is still running.

        ``0`` means ``spin`` returned normally, not necessarily cleanly (e.g. every node
        became not-live because one failed to initialize or raised ``NodeStop``). Check
        ``shutdown_clean`` as well. A negative value ``-N`` means
        the process was killed by signal ``N`` (``-11`` SIGSEGV, ``-9`` SIGKILL, ``-6``
        SIGABRT, ...); signal deaths produce no Python traceback and are the usual reason
        a scheduler appears to vanish "without any cause". A positive value is an explicit
        ``sys.exit`` code.
        """
        return self._proc.exitcode

    def terminate(self) -> None:
        """Send SIGTERM to the underlying process."""
        if self._proc.is_alive():
            self._proc.terminate()

    def kill(self) -> None:
        """Send SIGKILL to the underlying process."""
        if self._proc.is_alive():
            self._proc.kill()

    def exec(self) -> None:
        """Process entrypoint: ignore SIGINT, turn SIGTERM into a graceful stop, run loop.

        SIGINT is left to the parent (the app coordinates shutdown via the shared stop
        event). SIGTERM -- sent by ``terminate()`` -- is converted into a graceful stop
        request rather than an immediate kill so nodes still run their shutdown
        callbacks. This is critical for the Spot control node, whose ``post_close``
        performs the safe-fallback (stow + sit) and power-off sequence; a hard kill
        there would leave the robot powered on and standing with no controller.
        """
        # ``SIG_IGN`` tells the OS to discard the signal entirely -- no handler runs and
        # no ``KeyboardInterrupt`` is raised in this process. Ctrl-C in the terminal
        # delivers SIGINT to every process in the foreground group (parent + every
        # scheduler child); without this, each child would raise ``KeyboardInterrupt``
        # at a random Python frame in parallel with the parent's orderly tiered
        # shutdown, breaking safe_fallback ordering on Spot control.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        # TEMP DIAGNOSTIC: dump every thread's Python traceback on a fatal signal
        # (SIGSEGV/SIGABRT/SIGBUS/SIGFPE), which otherwise leave no output at all. This pinpoints
        # the exact native call that crashes a worker (e.g. the ``--use-mocap`` policy_process
        # SIGSEGV). Writes to this process's stderr; harmless outside a crash.
        import faulthandler

        faulthandler.enable()
        _install_parent_death_signal()
        # Shorten the GIL switch interval (default 5 ms) for this scheduler process. The graph is
        # a chain of one-thread-per-node hand-offs; when a producer node pushes and notifies the
        # next node's thread, that consumer is runnable but cannot get the GIL until the producer
        # releases it -- up to one switch interval. With the default 5 ms, each of the 4 serial
        # hops (timer -> skill -> locomotion -> control) added ~5 ms of pure wake latency,
        # dominating the observation age (the inference itself is sub-ms). 0.5 ms keeps that
        # scheduling jitter ~10x smaller at negligible context-switch overhead (handoffs, not a
        # busy-loop, so switches stay infrequent).
        sys.setswitchinterval(0.0005)
        # Re-install the debug tap in this forked worker (its background
        # threads do not survive ``fork()``). No-op if not enabled.
        try:
            from judo.unlocked import debug_tap as _debug_tap

            _debug_tap.install()
        except Exception:
            pass
        self.spin()

    def spin(self) -> None:
        """Run scheduler loop in-process.

        Initializes node I/O resources, starts worker threads, waits for stop conditions,
        and performs orderly shutdown/cleanup.
        """
        self._shutdown_clean.clear()
        logging.info("Scheduler %s has started", self._name)

        for node in self._nodes:
            node.output_stage.open()
        for node in self._nodes:
            node.input_stage.open()
        for node in self._nodes:
            node.open()

        # Freeze the heap before the hot loop. gc.collect() first sweeps construction
        # garbage, then gc.freeze() promotes every surviving object (the graph, ONNX
        # weights, robot clients, big buffers) into a permanent generation the cyclic GC
        # never rescans -- removing the stop-the-world gen-2 scan of the whole live set (a
        # prime suspect for the ~85 ms stalls) from the steady state. GC stays ENABLED, so
        # objects allocated after this (per-tick temporaries, cycles) are still collected;
        # even multi-hour runs do not grow unboundedly. Per process, after fork, so each
        # scheduler freezes only its own heap.
        gc.collect()
        gc.freeze()

        self._start_node_threads()

        while not self._stop.is_set() and any((node.live for node in self._nodes)):
            sleep(0.2)

        for node in self._nodes:
            node.close()
        # Each node is now not-live, but its worker thread may still be parked in
        # a timeout-free input wait. Release those waits so every thread re-checks
        # liveness and returns; must run after close() (which sets the stop state)
        # and before join() (which would otherwise block forever).
        for node in self._nodes:
            node.wake()
        # Join worker threads before closing the stages so no thread can read a view
        # whose shared memory is being freed (which raises OverflowError in len(view)).
        for t in self.node_threads:
            t.join(timeout=1.0)
        workers_finished = all(not t.is_alive() for t in self.node_threads)
        for node in self._nodes:
            node.input_stage.close()
        for node in self._nodes:
            node.output_stage.close()

        logging.info("Scheduler %s has stopped", self._name)
        if workers_finished and not any(node.lifecycle_error for node in self._nodes):
            self._shutdown_clean.set()

    def _start_node_threads(self) -> None:
        """Create and start one daemon thread per live, scheduler-driven node.

        Nodes whose ``node_type`` is ``CALLBACK`` or ``GENERATOR`` are deliberately skipped: they are driven from
        outside the scheduler via ``Node.callback`` / ``Node.generator``. They remain full
        members of this scheduler -- ``spin`` opens and closes their stages and runs their
        ``open``/``close`` hooks (so ``warmup`` still publishes) -- they simply get no worker
        loop. Giving one a thread would busy-spin, not park: with no inputs connected,
        ``input_stage.wait()`` returns True immediately and ``_node`` would loop on the CPU.
        """
        self.node_threads = [
            Thread(
                target=_node,
                args=(node,),
                name=node.name,
                daemon=True,
            )
            for node in self._nodes
            if node.live and node.node_type is NodeType.UNLOCKED
        ]
        for t in self.node_threads:
            t.start()


class StopAll:
    """Signal handler utility that requests shutdown for multiple schedulers."""

    def __init__(self, schedulers: Iterable) -> None:
        self._schedulers = list(schedulers)

    def __call__(self, signum: Any, _: Any) -> None:
        """Handle a signal by calling stop() on all tracked schedulers."""
        for s in self._schedulers:
            s.stop()
