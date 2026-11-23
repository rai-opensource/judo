# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Tests for externally driven nodes: ``Node.callback`` and ``Node.generator``.

A node whose ``node_type`` is ``CALLBACK`` or ``GENERATOR`` is driven from OUTSIDE the scheduler. ``Threaded`` still
owns
its lifecycle -- stages are opened/closed and ``warmup`` is published -- but it gets no worker
thread, because something else (a GUI handler, a gRPC stream) is its loop.
"""

import threading
import time
import unittest
from typing import Any

from judo.unlocked import Node, NodeType
from judo.unlocked.const import const_cast
from judo.unlocked.node import _CLOSE_DRAIN_NS
from judo.unlocked.policy import Lossless
from judo.unlocked.schedule import Threaded

# Generous upper bound for a cross-thread hand-off; tests wait on an Event, they do not sleep for it.
_TIMEOUT_S = 5.0

#: Concurrency fixture for the ``callback`` serialization contract. Enough callers to make two
#: of them collide inside the single-producer queue path on the very first push (they start on a
#: barrier against a cold queue, so the collision is in ``SharedMemoryQueue._add_frame``), and
#: small enough that the whole case stays well inside a ``short`` test timeout.
_RACE_THREADS = 8
_RACE_CALLS = 40
#: Retries for the "unserialized callers damage the queue" direction. It reproduces on the first
#: attempt in practice; this only keeps a scheduling fluke from turning the case red.
_RACE_ATTEMPTS = 3
#: Keeps per-thread marker ranges disjoint, so a marker identifies its producer unambiguously.
_MARKER_STRIDE = 100_000
#: Body repeats of the marker. Long enough that an interleaved writer overlaps it, short enough
#: that every record still fits one 64 KB frame.
_BODY_LEN = 64

#: Stall held inside a generator's stage access so a shutdown can both land in the check-then-act
#: window AND outrun ``node._CLOSE_DRAIN_NS`` (250 ms) -- the only way to close a stage under an
#: in-flight external drive. Read from the node module so the two cannot drift apart.
_DRAIN_OVERRUN_S = 2.0 * _CLOSE_DRAIN_NS * 1.0e-9


def identity(value: int) -> int:
    """One in, one out."""
    return value


def pair(value: int) -> tuple[int, int]:
    """Fan out to two output channels."""
    return value, value * 10


def only_even(value: int) -> int | None:
    """Return ``None`` for odd values -- the documented "no output this tick" contract."""
    return value if value % 2 == 0 else None


def boom(_: int) -> int:
    """Always raises, to exercise the error path."""
    raise RuntimeError("target failed")


def marked(tid: int, seq: int) -> dict:
    """Self-describing payload: the body repeats the header.

    Lets the sink tell the two failure modes apart. A record that never arrives was LOST; a
    record whose body disagrees with its own header was CORRUPTED -- two serializations
    overlapped in the frame. Counting only arrivals would score corruption as success.
    """
    marker = tid * _MARKER_STRIDE + seq
    return {"marker": marker, "body": [marker] * _BODY_LEN}


class _MarkerSink:
    """Sink for :func:`marked` payloads that records every marker and every damaged record."""

    def __init__(self) -> None:
        self.markers: set[int] = set()
        self.damaged: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, message: dict) -> None:
        try:
            marker = int(const_cast(message["marker"]))
            body = list(const_cast(message["body"]))
        except BaseException as error:
            # A record torn badly enough that pickle cannot reconstruct it.
            with self._lock:
                self.damaged.append(f"undecodable record: {type(error).__name__}: {error}")
            return
        with self._lock:
            self.markers.add(marker)
            if body != [marker] * _BODY_LEN:
                self.damaged.append(f"record {marker}: body does not match its own header")


class _Collector:
    """Sink that records what it receives and signals once a target count is reached."""

    def __init__(self, expected: int = 1) -> None:
        self.values: list[Any] = []
        self._expected = expected
        self.done = threading.Event()

    def __call__(self, value: Any) -> None:
        self.values.append(value)
        if len(self.values) >= self._expected:
            self.done.set()


class ExternallyDrivenNodeTest(unittest.TestCase):
    """Non-default ``NodeType`` lifecycle, ``callback`` publishing and ``generator`` consumption."""

    def setUp(self) -> None:
        self._schedulers: list[Threaded] = []
        self._threads: list[threading.Thread] = []

    def tearDown(self) -> None:
        for sched in self._schedulers:
            sched.stop()
        for thread in self._threads:
            thread.join(timeout=_TIMEOUT_S)

    def _spin(self, *nodes: Node) -> None:
        """Run a scheduler over ``nodes`` in THIS process and block until its threads are up.

        ``Threaded.start`` forks; the tests drive ``spin`` directly so they can poke the
        externally driven nodes from the test thread.
        """
        sched = Threaded(f"test_{len(self._schedulers)}", nodes)
        thread = threading.Thread(target=sched.spin, daemon=True)
        self._schedulers.append(sched)
        self._threads.append(thread)
        thread.start()
        # Wait on the scheduler's own post-open marker rather than sleeping a fixed amount:
        # ``spin`` assigns ``node_threads`` only after every stage has been opened and every
        # ``node.open`` hook (so every ``warmup`` push) has run.
        idle = threading.Event()
        for _ in range(int(_TIMEOUT_S * 100)):
            if getattr(sched, "node_threads", None) is not None:
                return
            idle.wait(0.01)
        self.fail("scheduler did not finish opening node stages in time")

    # -- scheduling ------------------------------------------------------ #

    def test_externally_driven_node_gets_no_worker_thread(self) -> None:
        """The scheduler skips CALLBACK / GENERATOR nodes when spawning workers."""
        external = Node("external", identity, node_type=NodeType.CALLBACK)
        collector = _Collector()
        sink = Node("sink", collector)
        sink.input_stage.connect((external.output_stage[0], Lossless()))

        self._spin(external, sink)

        names = {thread.name for thread in threading.enumerate()}
        self.assertNotIn("external", names, "externally driven node must not get a worker thread")
        self.assertIn("sink", names, "a normal node must still get a worker thread")

    def test_node_type_defaults_to_unlocked(self) -> None:
        """Nodes are scheduler-driven unless a non-default ``node_type`` is requested."""
        self.assertIs(NodeType.UNLOCKED, Node("normal", identity).node_type)
        self.assertTrue(Node("normal2", identity).is_scheduled)
        for node_type in (NodeType.CALLBACK, NodeType.GENERATOR):
            node = Node(f"external_{node_type}", identity, node_type=node_type)
            self.assertIs(node_type, node.node_type)
            self.assertFalse(node.is_scheduled, f"{node_type} must not be scheduler-driven")

    def test_node_type_accepts_its_string_value(self) -> None:
        """``NodeType`` is a StrEnum, so a graph spec's plain string coerces to the member.

        The graph path relies on this: ``materialize`` passes ``NodeType(spec.node_type)``, and the
        spec stores the plain string so it stays JSON.
        """
        self.assertIs(NodeType.CALLBACK, Node("from_str", identity, node_type="callback").node_type)
        self.assertEqual("generator", NodeType.GENERATOR)

    def test_warmup_is_published_without_a_worker_thread(self) -> None:
        """``Threaded`` still runs ``open`` for externally driven nodes, so ``warmup`` reaches consumers."""
        external = Node("external", identity, warmup=[7], node_type=NodeType.CALLBACK)
        collector = _Collector(expected=1)
        sink = Node("sink", collector)
        sink.input_stage.connect((external.output_stage[0], Lossless()))

        self._spin(external, sink)

        self.assertTrue(collector.done.wait(_TIMEOUT_S), "warmup was never published")
        self.assertEqual([7], collector.values)

    # -- node_type guards ---------------------------------------------------- #

    def test_each_node_type_blocks_the_other_drive_methods(self) -> None:
        """A node_type enables exactly one of exec / callback / generator; the others are inert."""
        for node_type in NodeType:
            node = Node(f"guard_{node_type}", identity, node_type=node_type)
            with self.assertLogs(node._name, level="ERROR") as logged:
                if node_type is not NodeType.CALLBACK:
                    node.callback(1)
                if node_type is not NodeType.UNLOCKED:
                    node.exec()
                if node_type is not NodeType.GENERATOR:
                    self.assertEqual([], list(node.generator()), "blocked generator must be empty")
            self.assertTrue(node.live, "a blocked call must not stop the node")
            joined = "\n".join(logged.output)
            self.assertIn("disabled", joined)
            self.assertIn(str(node_type), joined, "the message should name the declared node_type")

    def test_blocked_call_is_logged_once_per_method(self) -> None:
        """The guard must not flood: a scheduler loop could hit it every tick."""
        node = Node("guard_repeat", identity, node_type=NodeType.CALLBACK)
        with self.assertLogs(node._name, level="ERROR") as logged:
            for _ in range(10):
                node.exec()
        self.assertEqual(1, len(logged.output), logged.output)

    def test_allowed_drive_method_is_not_blocked(self) -> None:
        """The guard must not interfere with the node's own node_type."""
        external = Node("allowed", identity, node_type=NodeType.CALLBACK)
        collector = _Collector(expected=1)
        sink = Node("sink", collector)
        sink.input_stage.connect((external.output_stage[0], Lossless()))

        self._spin(external, sink)
        external.callback(42)

        self.assertTrue(collector.done.wait(_TIMEOUT_S), "allowed callback must still publish")
        self.assertEqual([42], collector.values)

    # -- callback -------------------------------------------------------- #

    def test_callback_publishes_from_the_calling_thread(self) -> None:
        """``callback`` runs the target inline and pushes straight into the output queue."""
        external = Node("external", identity, node_type=NodeType.CALLBACK)
        collector = _Collector(expected=3)
        sink = Node("sink", collector)
        sink.input_stage.connect((external.output_stage[0], Lossless()))

        self._spin(external, sink)
        for value in (1, 2, 3):
            external.callback(value)

        self.assertTrue(collector.done.wait(_TIMEOUT_S), f"only got {collector.values}")
        self.assertEqual([1, 2, 3], collector.values, "order must follow call order")

    def test_callback_fans_out_a_tuple_across_channels(self) -> None:
        """A tuple return is spread one element per output channel."""
        external = Node("external", pair, node_type=NodeType.CALLBACK)
        first, second = _Collector(expected=2), _Collector(expected=2)
        sink_a, sink_b = Node("sink_a", first), Node("sink_b", second)
        sink_a.input_stage.connect((external.output_stage[0], Lossless()))
        sink_b.input_stage.connect((external.output_stage[1], Lossless()))

        self._spin(external, sink_a, sink_b)
        external.callback(1)
        external.callback(2)

        self.assertTrue(first.done.wait(_TIMEOUT_S), f"channel 0 got {first.values}")
        self.assertTrue(second.done.wait(_TIMEOUT_S), f"channel 1 got {second.values}")
        self.assertEqual([1, 2], first.values)
        self.assertEqual([10, 20], second.values)

    def test_callback_publishes_nothing_when_the_target_returns_none(self) -> None:
        """``None`` means "no output this tick" -- the odd values must not reach the sink."""
        external = Node("external", only_even, node_type=NodeType.CALLBACK)
        collector = _Collector(expected=2)
        sink = Node("sink", collector)
        sink.input_stage.connect((external.output_stage[0], Lossless()))

        self._spin(external, sink)
        for value in (1, 2, 3, 4):
            external.callback(value)

        self.assertTrue(collector.done.wait(_TIMEOUT_S), f"only got {collector.values}")
        self.assertEqual([2, 4], collector.values)

    def test_callback_does_not_leak_in_exec(self) -> None:
        """``_in_exec`` must be cleared even though ``callback`` returns early on ``None``."""
        external = Node("external", only_even, node_type=NodeType.CALLBACK)
        self._spin(external)

        external.callback(2)
        self.assertFalse(external.in_exec)
        external.callback(3)  # early-returns on the None path
        self.assertFalse(external.in_exec, "in_exec leaked on the None path")

    def test_callback_routes_target_errors_without_raising(self) -> None:
        """A failing target is recorded, not propagated to the external caller."""
        external = Node("external", boom, node_type=NodeType.CALLBACK)
        self._spin(external)

        external.callback(1)  # must not raise into the caller (e.g. a GUI handler)
        self.assertFalse(external.in_exec)

    # -- callback concurrency contract ----------------------------------- #

    def _drive_from_many_threads(self, *, serialize: bool) -> tuple[list[str], set[int]]:
        """Hammer one CALLBACK node from ``_RACE_THREADS`` threads; report damage and losses.

        Returns ``(damage, lost)`` where ``damage`` collects exceptions raised back into the
        callers plus records the sink found corrupted, and ``lost`` is the set of markers that
        were pushed but never arrived.

        The drivers start on a barrier against a cold queue so their first pushes land in
        ``SharedMemoryQueue`` together -- that is where an unserialized collision is certain
        rather than merely likely.
        """
        source = Node("source", marked, node_type=NodeType.CALLBACK)
        sink = _MarkerSink()
        collector = Node("collector", sink)
        collector.input_stage.connect((source.output_stage[0], Lossless()))
        self._spin(source, collector)

        raised: list[str] = []
        serializer = threading.Lock()
        start = threading.Barrier(_RACE_THREADS)

        def drive(tid: int) -> None:
            start.wait()
            for seq in range(_RACE_CALLS):
                try:
                    if serialize:
                        with serializer:
                            source.callback(tid, seq)
                    else:
                        source.callback(tid, seq)
                except BaseException as error:
                    raised.append(f"callback() raised {type(error).__name__}: {error}")
                    return

        drivers = [threading.Thread(target=drive, args=(tid,), daemon=True) for tid in range(_RACE_THREADS)]
        for driver in drivers:
            driver.start()
        for driver in drivers:
            driver.join(timeout=_TIMEOUT_S)

        expected = {tid * _MARKER_STRIDE + seq for tid in range(_RACE_THREADS) for seq in range(_RACE_CALLS)}
        # The sink is a scheduler-driven node, so give it time to drain -- but stop early once
        # everything has landed, or as soon as a caller has already reported a failure.
        deadline = time.monotonic() + _TIMEOUT_S
        while time.monotonic() < deadline and not raised and not sink.markers >= expected:
            time.sleep(0.01)
        return raised + sink.damaged, expected - sink.markers

    def test_unserialized_concurrent_callbacks_damage_the_queue(self) -> None:
        """``callback`` is NOT internally serialized: concurrent callers corrupt/lose records.

        ``Node.callback`` publishes on the caller's thread straight into a lock-free
        SINGLE-producer queue. Two callers therefore interleave inside
        ``SharedMemoryQueue.push``: they race to allocate the same frame (``FileExistsError``
        back into the caller), and they read the same ``Frame._free_offset`` and overwrite each
        other's serialized bytes (a record whose body no longer matches its header).

        This is the hazard behind ``Node.callback``'s "keep concurrent invocations of the same
        node serialized" note, and the reason serialization currently lives with the caller --
        e.g. viser dispatches every GUI handler on a 32-worker thread pool, so nothing upstream
        of the node provides it. The test pins the behavior so the contract cannot be quietly
        assumed away; it must be updated the day ``Node`` takes that lock itself.
        """
        for _ in range(_RACE_ATTEMPTS):
            damage, lost = self._drive_from_many_threads(serialize=False)
            if damage or lost:
                return
        self.fail(
            "unserialized concurrent callbacks published cleanly: either Node.callback now "
            "serializes internally (update this test and the docstring contract) or the fixture "
            "no longer creates contention"
        )

    def test_serialized_concurrent_callbacks_are_lossless(self) -> None:
        """The same load is exact once callers serialize -- the damage above is the race, not the load.

        Same node, same threads, same call count; the only change is a lock held across
        ``callback``. Everything arrives, in one piece. This is the other half of the contract:
        serialization at the caller is SUFFICIENT, so a node fed by an already-serialized event
        loop needs nothing further.
        """
        damage, lost = self._drive_from_many_threads(serialize=True)

        self.assertEqual([], damage, "serialized callers must not damage the queue")
        self.assertEqual(set(), lost, "serialized callers must not lose records")

    # -- generator ------------------------------------------------------- #

    def test_generator_yields_results_to_an_external_consumer(self) -> None:
        """The consumer thread drives the node, parking on the input view itself."""
        source = Node("source", identity, node_type=NodeType.CALLBACK)
        relay = Node("relay", identity, node_type=NodeType.GENERATOR)
        relay.input_stage.connect((source.output_stage[0], Lossless()))

        self._spin(source, relay)

        received: list[int] = []
        got_all = threading.Event()

        def consume() -> None:
            for value in relay.generator():
                received.append(value)
                if len(received) >= 3:
                    got_all.set()

        consumer = threading.Thread(target=consume, daemon=True)
        self._threads.append(consumer)
        consumer.start()

        for value in (4, 5, 6):
            source.callback(value)

        self.assertTrue(got_all.wait(_TIMEOUT_S), f"only got {received}")
        self.assertEqual([4, 5, 6], received)

    def test_generator_skips_none_results(self) -> None:
        """A ``None`` result advances the input without yielding.

        ``generator`` is annotated ``Generator[R, ...]`` and never yields ``None``, so the
        consumer below only ever sees the even values.
        """
        source = Node("source", identity, node_type=NodeType.CALLBACK)
        relay: Node = Node("relay", only_even, node_type=NodeType.GENERATOR)
        relay.input_stage.connect((source.output_stage[0], Lossless()))

        self._spin(source, relay)

        received: list[int] = []
        got_all = threading.Event()

        def consume() -> None:
            for value in relay.generator():
                received.append(value)
                if len(received) >= 2:
                    got_all.set()

        consumer = threading.Thread(target=consume, daemon=True)
        self._threads.append(consumer)
        consumer.start()

        for value in (1, 2, 3, 4):
            source.callback(value)

        self.assertTrue(got_all.wait(_TIMEOUT_S), f"only got {received}")
        self.assertEqual([2, 4], received)

    def test_generator_ends_on_close_and_wake(self) -> None:
        """``close`` + ``wake`` releases the untimed park so the consumer loop can exit.

        This is the shutdown contract for a consumer parked in ``generator()``: the shared-memory
        wait has no timeout, so marking the node not-live is not enough on its own.
        """
        source = Node("source", identity, node_type=NodeType.CALLBACK)
        relay = Node("relay", identity, node_type=NodeType.GENERATOR)
        relay.input_stage.connect((source.output_stage[0], Lossless()))

        self._spin(source, relay)

        finished = threading.Event()

        def consume() -> None:
            for _ in relay.generator():
                pass
            finished.set()

        consumer = threading.Thread(target=consume, daemon=True)
        self._threads.append(consumer)
        consumer.start()
        source.callback(1)

        relay.close()
        relay.wake()

        self.assertTrue(finished.wait(_TIMEOUT_S), "generator did not end on close + wake")

    def test_generator_records_wait_and_exec_time(self) -> None:
        """The node's own stats instrument the two stages, so a consumer needs no extra timing."""
        source = Node("source", identity, node_type=NodeType.CALLBACK)
        relay = Node("relay", identity, node_type=NodeType.GENERATOR)
        relay.input_stage.connect((source.output_stage[0], Lossless()))

        self._spin(source, relay)

        received = threading.Event()

        def consume() -> None:
            for _ in relay.generator():
                received.set()

        consumer = threading.Thread(target=consume, daemon=True)
        self._threads.append(consumer)
        consumer.start()
        source.callback(1)

        self.assertTrue(received.wait(_TIMEOUT_S))
        self.assertGreaterEqual(relay.exec_time.n, 1, "exec_time should cover the target call")
        self.assertGreaterEqual(relay.wait_time.n, 1, "wait_time should cover the blocking wait")

    # -- generator vs. raced shutdown ------------------------------------ #

    @staticmethod
    def _stall(stage: Any, method: str) -> threading.Event:
        """Park ``stage.<method>`` past the close-drain budget; returns an "entered" event.

        Reproduces deterministically what is otherwise a handful of bytecodes: the generator
        re-checks ``live``, then touches the input stage. Stalling INSIDE that gap for longer
        than ``_CLOSE_DRAIN_NS`` lets a shutdown both land in the gap and outrun ``_drain``,
        which is the only way the stage can be closed under an in-flight external drive.
        """
        entered = threading.Event()
        real = getattr(stage, method)

        def stalled(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            time.sleep(_DRAIN_OVERRUN_S)
            return real(*args, **kwargs)

        setattr(stage, method, stalled)
        return entered

    def _consume_through_shutdown(self, stall_method: str) -> list[str]:
        """Drive a generator into ``stall_method``, tear the scheduler down, report escapes."""
        source = Node("source", identity, node_type=NodeType.CALLBACK)
        relay = Node("relay", identity, node_type=NodeType.GENERATOR)
        relay.input_stage.connect((source.output_stage[0], Lossless()))

        self._spin(source, relay)
        entered = self._stall(relay.input_stage, stall_method)

        escaped: list[str] = []
        finished = threading.Event()

        def consume() -> None:
            try:
                for _ in relay.generator():
                    pass
                finished.set()
            except BaseException as error:
                escaped.append(f"{type(error).__name__}: {error}")

        consumer = threading.Thread(target=consume, daemon=True)
        self._threads.append(consumer)
        consumer.start()
        source.callback(1)

        self.assertTrue(entered.wait(_TIMEOUT_S), f"generator never reached input_stage.{stall_method}")
        # The generator is now inside the stalled call with _in_exec set, so this shutdown
        # exhausts the drain budget and closes the stage underneath it.
        self._schedulers[-1].stop()
        self._threads[0].join(timeout=_TIMEOUT_S)
        consumer.join(timeout=_TIMEOUT_S)

        if not finished.is_set() and not escaped:
            escaped.append("consumer neither ended nor raised")
        return escaped

    def test_generator_ends_cleanly_when_shutdown_races_its_input_read(self) -> None:
        """A stage closed during ``input_args`` ends the loop; it does not raise into the consumer.

        ``input_args`` is a check-then-act against the liveness re-check above it, so a
        shutdown that lands in the gap and outruns ``_drain`` leaves it reading a torn-down
        view. What comes back depends on the policy -- ``IndexError`` for ``Lossless``, a bare
        ``AssertionError`` for ``Latest``/``KeepLatest`` -- and none of it belongs in a foreign
        consumer thread. The drive must take the normal stop exit instead.
        """
        self.assertEqual([], self._consume_through_shutdown("input_args"))

    def test_generator_ends_cleanly_when_shutdown_races_its_input_advance(self) -> None:
        """Same contract for the post-yield ``next``, which has the wider of the two windows.

        The consumer has just been resumed from the yield, so a shutdown that began while it
        was suspended there is already well under way by the time the view is advanced.
        """
        self.assertEqual([], self._consume_through_shutdown("next"))

    def test_generator_records_a_stop_when_its_stage_is_torn_down(self) -> None:
        """The raced exit is a stop, not a lifecycle error -- shutdown stays "clean".

        ``_stage_gone`` records ``NodeStop`` rather than routing through ``_init_error``, so a
        stage closed under an in-flight drive reads as teardown and does not flip
        ``lifecycle_error`` (which ``Threaded.shutdown_clean`` is computed from).
        """
        source = Node("source", identity, node_type=NodeType.CALLBACK)
        relay = Node("relay", identity, node_type=NodeType.GENERATOR)
        relay.input_stage.connect((source.output_stage[0], Lossless()))

        self._spin(source, relay)
        entered = self._stall(relay.input_stage, "input_args")

        consumer = threading.Thread(target=lambda: list(relay.generator()), daemon=True)
        self._threads.append(consumer)
        consumer.start()
        source.callback(1)

        self.assertTrue(entered.wait(_TIMEOUT_S))
        self._schedulers[-1].stop()
        self._threads[0].join(timeout=_TIMEOUT_S)
        consumer.join(timeout=_TIMEOUT_S)

        self.assertFalse(relay.live, "a torn-down stage must stop the node")
        self.assertFalse(relay.lifecycle_error, "teardown must not be recorded as a lifecycle error")
        self.assertFalse(relay.in_exec, "_in_exec must be cleared on the raced exit")


if __name__ == "__main__":
    unittest.main()
