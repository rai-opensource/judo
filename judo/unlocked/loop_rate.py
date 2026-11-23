# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Minimal fixed-rate loop pacer.

A tiny replacement for the *pacing* role of ``TimeTracker`` (its ``step`` + ``sleep_remaining``
pair) without the statistics / warning / printing machinery. Call :meth:`Rate.sleep` once at the
end of each loop iteration to hold the loop at a fixed period; use a plain ``bool`` flag for the
loop's stop condition (the pacer deliberately owns pacing only, not lifecycle).
"""

import time

from judo.unlocked import debug_tap

_NS_PER_S = 1_000_000_000
# ``time.sleep`` can overshoot its target by a few milliseconds (notably on macOS for sub-20 ms
# waits). To stay accurate we coarse-sleep to ~this margin before the deadline, then busy-wait the
# remainder. For periods at or below the margin this degenerates to a pure spin -- exactly what we
# want for e.g. the ~2 ms physics step: accurate real-time pacing at the cost of one busy thread.
_SPIN_MARGIN_NS = 2_000_000  # 2 ms


class Rate:
    """Hold a loop at a fixed period by sleeping the leftover time each iteration.

    Usage::

        rate = Rate(period_s=0.02)          # 50 Hz
        while not stop:
            do_work()
            rate.sleep()

    Each :meth:`sleep` blocks until the next period boundary. If an iteration overruns the period
    the schedule resyncs from the current time instead of bursting a train of zero-length cycles
    to catch up, so a momentary stall costs at most one stretched cycle -- matching the old
    ``TimeTracker`` pacing behavior.
    """

    def __init__(self, period_s: float) -> None:
        """Create a pacer for the given loop period in seconds."""
        self._period_ns = max(0, int(period_s * _NS_PER_S))
        # Anchor the first deadline one period out so the first sleep() also paces the loop.
        self._deadline_ns = time.perf_counter_ns() + self._period_ns

    @debug_tap.freezes_when_paused
    def sleep(self) -> None:
        """Block until the next period boundary (resyncs without catch-up if the loop fell behind).

        Decorated with :func:`~judo.unlocked.debug_tap.freezes_when_paused`, so under the graph
        editor's debug clock the loop blocks (advancing no work) while paused and resyncs its
        deadline from *now* on resume. Entirely a no-op outside a Debug session.
        """
        now_ns = time.perf_counter_ns()
        deadline_ns = self._deadline_ns
        if now_ns >= deadline_ns:
            # Behind schedule (a slow iteration, or a long gap since construction): resync from
            # now and skip the sleep rather than fire off a burst of zero-length cycles.
            self._deadline_ns = now_ns + self._period_ns
            return
        coarse_s = (deadline_ns - now_ns - _SPIN_MARGIN_NS) / _NS_PER_S
        if coarse_s > 0:
            time.sleep(coarse_s)
        while time.perf_counter_ns() < deadline_ns:
            time.sleep(0)  # yield the GIL so sibling threads keep running during the spin
        self._deadline_ns = deadline_ns + self._period_ns
