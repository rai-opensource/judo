# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import traceback
from enum import Enum
from inspect import signature
from time import perf_counter_ns, sleep
from typing import Any, Callable, Generator, Generic, Iterable, ParamSpec, TypeVar, cast

from judo.unlocked.input_stage import InputStage
from judo.unlocked.output_stage import OutputStage
from judo.unlocked.time_stat import SharedTimeStat

# P captures the parameters of the business logic function
P = ParamSpec("P")
# R captures the return type of the business logic function
R = TypeVar("R")


class NodeType(str, Enum):
    """How a node is driven -- and therefore whether the scheduler gives it a worker thread.

    UNLOCKED: unlocked managed execution of node. It takes unlocked driven inputs and returns unlocked driven outputs.

    CALLBACK: node executions is managed outside the node. The node takes no inputs from unlocked, and gets
    inputs from parent execution. Within the unlocked ecosystem these are source nodes, and outside unlocked these are
    sink nodes into unlocked.

    GENERATOR: node execution is managed outside the node. The node takes inputs from unlocked, and the business logic
    output is yielded to the parent execution. Within the unlocked ecosystem these are sink nodes, and outside unlocked
    these are source nodes from unlocked.
    """

    def __str__(self) -> str:
        return str(self.value)  # match ``StrEnum`` (py3.10 has none)

    #: Driven by the scheduler: it runs a worker thread looping ``wait`` -> ``exec``.
    UNLOCKED = "unlocked"
    #: Driven by an external producer calling :meth:`Node.callback` -- e.g. a GUI event handler
    #: publishing straight into the shared-memory queue on its own thread.
    CALLBACK = "callback"
    #: Driven by an external consumer iterating :meth:`Node.generator` -- e.g. a streaming client
    #: that must be handed an iterator and paces the node with its own loop.
    GENERATOR = "generator"


#: The single drive method each node type enables; used to explain a blocked call (see Node._blocked).
_DRIVER_NAME = {NodeType.UNLOCKED: "exec", NodeType.CALLBACK: "callback", NodeType.GENERATOR: "generator"}

#: How long ``close`` waits for an in-flight external drive to finish (see Node._drain). Long
#: enough to cover any sane GUI handler, short enough that a wedged one cannot stall the tiered
#: shutdown -- ``spin`` closes nodes sequentially, so this is a per-node addition to that budget.
_CLOSE_DRAIN_NS = 250_000_000
#: Poll interval for that wait. Fine-grained because the common case finishes immediately and a
#: coarse tick would round every close up to its own length.
_CLOSE_DRAIN_POLL_S = 0.001


class NodeStop(Exception):
    """Signal graceful node termination.

    Raise from a node function to stop future executions of that node.
    """


class NodeRetry(Exception):
    """Signal retry on the current input without advancing input cursors.

    The node thread sleeps for optional ``delay_s`` and then re-enters the wait/exec loop,
    preserving the current input view positions.
    """

    def __init__(self, message: str = "", *, delay_s: float | None = None):
        super().__init__(message)
        self._delay_s = delay_s

    def wait(self) -> None:
        """Sleep for configured retry delay, if any."""
        if self._delay_s is not None:
            sleep(self._delay_s)


class NodeSkip(Exception):
    """Signal skipping the current input and advancing to the next item."""

    def __init__(self, message: str = "", *, delay_s: float | None = None):
        super().__init__(message)
        self._delay_s = delay_s

    def wait(self) -> None:
        """Sleep for configured skip delay, if any."""
        if self._delay_s is not None:
            sleep(self._delay_s)


class Node(Generic[P, R]):
    """Data structure that binds target function, input stage, and output stage.

    A Node runs a target callable in a scheduler-managed thread. Inputs are supplied by
    ``InputStage`` (view/policy pairs over shared-memory queues), and outputs are published by
    ``OutputStage`` (per-channel shared-memory queues).

    Per step contract:
    1. wait for input readiness according to input policies
    2. materialize ``(*args, **kwargs)`` from current views
    3. execute target callable
    4. push outputs (if any)
    5. advance input views via policy ``next``

    Error/control signals:
    - ``NodeStop``: terminate node
    - ``NodeRetry``: keep current input, retry later
    - ``NodeSkip``: drop current input and continue

    Synchronization note: Node does not enforce cross-queue correlation. If multiple upstream
    branches feed this node, application-level IDs are required for matching semantics.
    """

    def __init__(
        self,
        name: str,
        target: Callable[P, R],
        *,
        warmup: Iterable = [],
        pre_open: Callable | None = None,
        post_close: Callable | None = None,
        node_type: NodeType | str = NodeType.UNLOCKED,
        logger: logging.Logger | None = None,
    ):
        assert name, "Node name must not be empty"
        self._name = name
        self._target: Callable[P, R] = target
        self._warmup = list(warmup)
        self._pre_open = pre_open
        self._post_close = post_close
        # Accepts the member or its string value: the graph spec stores the string so it stays
        # JSON, and ``NodeType`` is a StrEnum, so both coerce to the same member here.
        self._node_type = NodeType(node_type)
        # Drive methods already reported as misused, so a hot loop logs once, not per call.
        self._misuse_logged: set[NodeType] = set()
        self._signature = signature(self._target)
        self._output_stage = OutputStage(self._name, self._signature.return_annotation)
        self._input_stage = InputStage(self._name, self._signature)
        self._stop: BaseException | None = None
        self._init_error: BaseException | None = None
        self._in_exec = False
        self._last_update_ns: float = perf_counter_ns()
        if logger is None:
            logger = logging.getLogger(self._name)
        self._logger = logger

        # Externally driven nodes are spun from arbitrary threads, so their stats need a lock;
        # a scheduler-driven node is only ever touched by its own worker.
        _stat_lock = self._node_type is not NodeType.UNLOCKED
        self.wait_time = SharedTimeStat("wait_time", lock=_stat_lock)
        self.exec_time = SharedTimeStat("exec_time", lock=_stat_lock)

    @property
    def name(self) -> str:
        """Get node name."""
        return self._name

    @property
    def node_type(self) -> NodeType:
        """How this node is driven (see :class:`NodeType`)."""
        return self._node_type

    @property
    def is_scheduled(self) -> bool:
        """Whether a scheduler should run a wait/exec worker loop for this node.

        Derived from :attr:`node_type`: only ``NodeType.UNLOCKED`` nodes are scheduler-driven.
        """
        return self._node_type is NodeType.UNLOCKED

    @property
    def warmup(self) -> list:
        """Values published by :meth:`open` before the node is driven for the first time.

        Read-only view of a load-bearing property: a ``KeepLatest`` consumer reports ready only
        once something has been published (``len(view) > 0``), and ``InputStage.ready`` is an
        ``all``, so a channel used as a configuration STORE must be warmed or it holds its
        consumer un-ready indefinitely. Exposed so that invariant can be asserted from outside.
        """
        return list(self._warmup)

    @property
    def live(self) -> bool:
        """Check if node can continue execution.

        Returns False when a stop signal or initialization error has been recorded.
        """
        return self._stop is None and self._init_error is None

    @property
    def lifecycle_error(self) -> bool:
        """Whether a ``pre_open`` or ``post_close`` callback failed.

        This does not track other runtime errors (including target execution errors,
        which are logged and may be retried on subsequent inputs).
        """
        return self._init_error is not None

    @property
    def in_exec(self) -> bool:
        """Check if target callable is currently executing."""
        return self._in_exec

    @property
    def output_stage(self) -> OutputStage:
        """Get output stage (producer side of node outputs)."""
        return self._output_stage

    @property
    def input_stage(self) -> InputStage:
        """Get input stage (consumer side of node inputs)."""
        return self._input_stage

    @property
    def ready(self) -> bool:
        """Check if node is eligible to execute now.

        True only when node is not already in execution and all input policies report ready.
        """
        return not self._in_exec and self._input_stage.ready

    def wait(self) -> bool:
        """Wait for input data according to input policy semantics.

        Blocks with no timeout (see schedule._node); returns False immediately
        if the node is not live so a stopped node exits promptly.
        """
        if not self.live:
            return False
        with self.wait_time:
            return self._input_stage.wait()

    def wake(self) -> None:
        """Release a parked input wait so a stopped node thread can exit.

        Delegates to the input stage, which signals every input view. Called by
        the scheduler during shutdown (see schedule.spin).
        """
        self._input_stage.wake()

    def open(self) -> None:
        """Run node startup callbacks and optionally publish warmup outputs.

        ``pre_open`` is executed first. If it fails, node transitions to init-error state.
        Then each warmup item is pushed through output stage as an initial publish.
        """
        if self._pre_open is not None:
            try:
                self._pre_open()
            except BaseException as error:
                self._init_error = error
                logging.error("Node %s failed to initialize with %s", self._name, error)
                logging.error(
                    "Node %s failed to initialize with %s\n === traceback ===\n%s =================",
                    self._name,
                    error,
                    traceback.format_exc(),
                )
                return
        for w in self._warmup:
            if not isinstance(w, tuple):
                w = (w,)
            self._output_stage.push(w)

    def close(self) -> None:
        """Mark node stopped, run the shutdown callback, then drain an in-flight external drive.

        ``post_close`` is best-effort; exceptions are recorded as init/shutdown errors.

        Ordering matters. ``post_close`` runs BEFORE the drain, because for several nodes it is
        precisely what ends the in-flight exec: ``Mocap.stop`` clears ``_running`` and notifies
        the condition its target is parked on, and ``SpotControl.stop`` releases the control
        loop. Draining first would wait for something only the hook it has not reached yet can
        deliver -- a lock-order inversion that hangs shutdown for good when the stream feeding
        the target has gone quiet. ``spin`` closes nodes SEQUENTIALLY, so one node stuck here
        also denies every later node its ``post_close`` -- including Spot's safe-fallback
        (stow + sit + power-off).
        """
        if self._init_error is not None:
            return
        # Recorded first so an exec/callback already in flight sees ``live`` go False and skips
        # its push (see the ``if not self.live`` guard in ``exec`` / ``callback``). That check,
        # not the drain below, is what actually keeps a stopping node out of the queue.
        self._stop = self._stop or NodeStop("Closed nominally.")
        if self._post_close is not None:
            try:
                self._post_close()
            except BaseException as error:
                self._init_error = error
                logging.error("Node %s failed to stop with %s", self._name, error)
        self._drain()

    def _drain(self) -> None:
        """Wait, briefly, for an externally driven invocation to leave the node.

        Closes the gap between a foreign thread publishing and the scheduler tearing the stages
        down underneath it: ``OutputStage.push`` feeds a lock-free single-producer queue, so a
        push landing after ``output_stage.close()`` writes into released shared memory.

        Only for ``CALLBACK`` / ``GENERATOR``. An ``UNLOCKED`` node is already covered -- ``spin``
        joins its worker thread before closing any stage -- and it is the case that must NOT wait:
        its target is allowed to block indefinitely inside exec (a socket read, a condition wait),
        so waiting on ``_in_exec`` there is the deadlock described in :meth:`close`. Externally
        driven nodes have no thread for the scheduler to join, which is why this exists at all.

        Bounded on purpose. ``_in_exec`` is a plain flag set AFTER the caller has already
        committed to running, so this can narrow the window but never close it; a deadline keeps
        a wedged external driver from holding the whole shutdown hostage, and the timeout is
        reported rather than swallowed.
        """
        if self._node_type is NodeType.UNLOCKED:
            return
        deadline = perf_counter_ns() + _CLOSE_DRAIN_NS
        while self._in_exec and perf_counter_ns() < deadline:
            sleep(_CLOSE_DRAIN_POLL_S)
        if self._in_exec:
            self._logger.warning(
                "Node %s still executing %.0f ms after close; tearing down its stages anyway",
                self._name,
                _CLOSE_DRAIN_NS * 1.0e-6,
            )

    def _blocked(self, method: NodeType) -> bool:
        """Report (once) and block a drive method that contradicts this node's :attr:`node_type`.

        Each ``NodeType`` enables exactly ONE way to drive the node -- ``UNLOCKED``/``exec``,
        ``CALLBACK``/``callback``, ``GENERATOR``/``generator`` -- because they disagree about who
        owns the loop and where inputs come from. Mixing them is a wiring mistake that is
        otherwise silent and confusing: ``exec`` on a callback node reads an input stage nothing
        feeds, and ``callback`` on a scheduler-driven node publishes alongside its worker.

        Blocks rather than raises: these are driven from foreign loops (a GUI handler, a
        streaming client) where an exception would take out the caller's thread, and the
        scheduler's own loop would spin on the traceback. Logged at error level, once per
        method, so the misconfiguration is loud without flooding.
        """
        if self._node_type is method:
            return False
        if method not in self._misuse_logged:
            self._misuse_logged.add(method)
            self._logger.error(
                "Node %s declares node_type=%s so %s() is disabled (it is driven by %s()). "
                "Either drive it with %s() or construct it with node_type=%s.",
                self._name,
                self._node_type,
                _DRIVER_NAME[method],
                _DRIVER_NAME[self._node_type],
                _DRIVER_NAME[self._node_type],
                method,
            )
        return True

    def _stage_gone(self, where: str, error: BaseException) -> None:
        """Record an input stage torn down underneath an in-flight external drive.

        Reachable only in the window where ``close`` lands between a liveness check and the
        stage access that follows it, AND then outruns ``_drain``'s budget -- the externally
        driven equivalent of the join ``spin`` performs for a worker thread. Rare (a stalled
        consumer, a loaded box), but not impossible, and the consequence varies by policy
        rather than being one tidy error: ``Lossless`` raises ``IndexError``, ``Latest`` and
        ``KeepLatest`` raise a bare ``AssertionError`` (an ``AttributeError`` under ``-O``,
        where the assert is stripped), and ``Optional`` does not raise at all -- it reports an
        empty view as ``None`` and feeds that to the target as if it were data.

        Treated as shutdown, not failure: the node is already stopping, so record a stop and
        let the drive loop end normally. That keeps ONE termination contract -- the external
        ``for`` sees ``StopIteration`` whether the shutdown was orderly or raced -- instead of
        leaking a policy-dependent exception into a foreign thread.

        Logged at warning level because reaching it means ``_drain`` was outrun, which is worth
        seeing if it ever stops being rare; at error level if the node was somehow still live,
        since that would be a genuine fault rather than teardown.
        """
        if self.live:
            self._logger.error("Node %s failed accessing its input stage in %s: %s", self._name, where, error)
        else:
            self._logger.warning(
                "Node %s input stage was closed during %s; ending the external drive", self._name, where
            )
        self._stop = self._stop or NodeStop(f"input stage closed during {where}")

    def generator(self) -> Generator[R, None, None]:
        """Run the node as a generator.

        Behavior/Intended use:
        - when called returns a generator
            - run input_stage logic and business logic before returning output in yield
            - "manually" spun by user in a consumer that takes data as needed
        - use in external syncs that expect a generator and not an Unlocked node
        - i.e. When passing spot commands from a policy to grpc command publisher pass this method to have the grpc
        thread subscribe and read from the shm thread.
        - Not thread safe, only one thread should run this at a time.

        Termination: the loop runs while the node is ``live`` and exits when a stop signal or
        init error is recorded (e.g. target raised ``NodeStop``) or when ``wait`` is released
        without data by ``wake()``. Steps that produce no value (``NodeRetry``/``NodeSkip``, or a
        ``None`` return) advance per ``_set_error`` semantics and continue without yielding.
        A consumer that abandons the generator leaves the last item unadvanced; the stage is
        closed by the scheduler anyway, so nothing leaks beyond that item.

        Either way the consumer's ``for`` loop ends normally -- a stop is translated into
        ``StopIteration``, never raised into the consumer's thread. That holds for a raced
        shutdown too: if teardown outruns ``_drain`` and closes the stage mid-step, the drive
        takes the same exit (see :meth:`_stage_gone`).
        """
        if self._blocked(NodeType.GENERATOR):
            return
        try:
            while self.live:
                if not self.wait():
                    continue
                with self.exec_time:
                    self._in_exec = True
                    if not self.live:
                        self._in_exec = False  # type: ignore[unreachable]
                        break  # type: ignore[unreachable]
                    try:
                        args, kwargs = self._input_stage.input_args()
                    except BaseException as e:
                        self._in_exec = False
                        self._stage_gone("input_args", e)
                        break
                    try:
                        result: R | None = cast(Callable[..., R], self._target)(*args, **kwargs)
                    except BaseException as e:
                        self._set_error(e)
                        self._in_exec = False
                        continue
                    if result is None:
                        if self.live:
                            self._input_stage.next()
                        self._in_exec = False
                        continue
                yield result
                if self.live:
                    try:
                        self._input_stage.next()
                    except BaseException as e:
                        self._in_exec = False
                        self._stage_gone("next", e)
                        break
                    self._in_exec = False
                else:
                    self._in_exec = False  # type: ignore[unreachable]
                    break  # type: ignore[unreachable]
        except GeneratorExit:
            self._in_exec = False
        return

    def callback(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Run the node execution as well as output_stage as a callback.

        Behavior/Intended use:
        - when called with correct arguments:
            - business logic is run and then output_stage serializes
            and puts data into shm queue for unlocked nodes to use
            - "manually" spun by user and then made available to other unlocked nodes
        - Wrap a function with Node but use this method to pass it to an external executor as a callback.
        - i.e. When setting up viser callbacks init nodes and pass this method to viser to minimize threads and locking

        Pair with ``node_type=NodeType.CALLBACK`` so the scheduler owns the node's stage lifecycle (and
        publishes its ``warmup``) without giving it a worker thread to spin on.

        Runs on the caller's thread, so an externally driven node is only as thread-safe as
        its target: keep concurrent invocations of the same node serialized.
        """
        if self._blocked(NodeType.CALLBACK):
            return
        with self.exec_time:
            self._in_exec = True
            try:
                try:
                    result: R | None = self._target(*args, **kwargs)
                except BaseException as e:
                    self._set_error(e)
                    return
                if result is None:
                    return
                outputs: tuple[Any, ...] = result if isinstance(result, tuple) else (result,)
                if not self.live:
                    return
                self._output_stage.push(outputs)
            finally:
                self._in_exec = False

    def exec(self) -> None:
        """Execute one node step.

        Behavior:
        - Calls ``input_stage.input_args()`` to read values from current views
        - Invokes target callable
        - Normalizes non-tuple return values to a one-element tuple
        - Pushes outputs unless result is ``None``
        - Advances inputs with ``input_stage.next()`` after successful handling

        Exceptions are routed through ``_set_error``.

        Unlike ``generator``, the stage accesses here need no teardown guard: ``spin`` JOINS
        this node's worker thread before closing any stage, so ``exec`` cannot still be running
        when its views go away. ``_drain``/:meth:`_stage_gone` exist precisely because an
        externally driven node has no thread for the scheduler to join.
        """
        if self._blocked(NodeType.UNLOCKED):
            return
        with self.exec_time:
            self._in_exec = True
            try:
                args, kwargs = self._input_stage.input_args()
                try:
                    result: R | None = cast(Callable[..., R], self._target)(*args, **kwargs)
                except BaseException as e:
                    self._set_error(e)
                    return
                if result is None:
                    self._input_stage.next()
                    return
                outputs: tuple[Any, ...] = result if isinstance(result, tuple) else (result,)
                if not self.live:
                    return
                self._output_stage.push(outputs)
                if not self.live:
                    return  # type: ignore[unreachable]
                self._input_stage.next()
            finally:
                self._in_exec = False

    def _set_error(self, error: Any) -> None:
        """Handle control-flow and failure exceptions raised by node function execution."""
        if isinstance(error, NodeStop):
            self._stop = error
            self._logger.info("Node %s stopped with %s", self._name, error)
        elif isinstance(error, NodeRetry):
            self._logger.warn("Node %s retrying input with %s", self._name, error)
            error.wait()
        elif isinstance(error, NodeSkip):
            self._logger.debug("Node %s skipping input with %s", self._name, error)
            error.wait()
            self._input_stage.next()
        else:
            self._logger.error(
                "Node %s failed with %s\n === traceback ===\n%s =================",
                self._name,
                error,
                traceback.format_exc(),
            )
            self._input_stage.next()
