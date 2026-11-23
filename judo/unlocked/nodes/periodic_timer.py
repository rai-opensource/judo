# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from time import perf_counter_ns, sleep

from judo.unlocked import debug_tap
from judo.unlocked.node import Node, NodeStop

_ZERO_NS = perf_counter_ns()


def elapsed_time(time_ns: int | None = None) -> int:
    return (time_ns or perf_counter_ns()) - _ZERO_NS


class PeriodicTimer:
    """Publishing generator."""

    _ADDED_SLEEP_DURATION_NS = 60000
    # Overrun beyond one period past which we RESYNC the deadline to "now" instead of catching up one
    # period at a time. A normal slow tick (GC, a fat ONNX call) overruns by at most a few ms and
    # should catch up gradually so the average rate holds. A debug PAUSE, however, freezes this node
    # at its scheduler gate for seconds while wall time (``perf_counter``) keeps advancing; catching
    # up one period per call would then fire a back-to-back BURST of thousands of ticks on resume
    # (``pause_s / period``), replaying the whole gap at once. For the physics timer that means
    # integrating the entire pause under a stale command in an instant -> the robot falls. 250 ms is
    # far above any legitimate single-tick stall and far below any real pause, so it cleanly
    # separates "slow tick" (catch up) from "was frozen" (resync, drop the missed ticks).
    _CATCHUP_BUDGET_NS = 250_000_000

    def __init__(
        self,
        *,
        frequency: float | None = None,
        period_s: float | None = None,
        duration_s: float | None = None,
    ) -> None:
        assert (frequency is None) ^ (
            period_s is None
        ), "Either 'frequency' or 'period_s' argument must be specified, but not both."
        if period_s is None:
            assert frequency is not None
            period_s = 1.0 / frequency
        self._duration_ns: int | None = None if duration_s is None else int(duration_s * 1.0e9)
        self._period_ns = int(period_s * 1.0e9)
        self._last_update_ns: int | None = None

    def __call__(self) -> int:
        """Published ellapsed time."""
        if self._last_update_ns is None:
            self._start = perf_counter_ns()
            self._last_update_ns = self._start
        else:
            now_ns = perf_counter_ns()
            behind_ns = now_ns - self._last_update_ns
            sleep_time_ns = self._period_ns - behind_ns - self._ADDED_SLEEP_DURATION_NS
            if sleep_time_ns > 0:
                sleep(sleep_time_ns * 1.0e-9)
                self._last_update_ns += self._period_ns
            elif behind_ns - self._period_ns > self._CATCHUP_BUDGET_NS:
                # Far past the deadline -- almost always a debug Pause gap. Resync to now (drop the
                # missed ticks) so resume continues at the real-time rate instead of bursting.
                self._last_update_ns = now_ns
            else:
                # Slightly behind (a slow tick): catch up one period so steady jitter averages out.
                self._last_update_ns += self._period_ns
            if self._duration_ns is not None and (self._last_update_ns - self._start) > self._duration_ns:
                raise NodeStop("Expired.")
        return elapsed_time(self._last_update_ns)

    def update_period_s(self, new_period_s: float) -> None:
        """Update the period of the timer."""
        self._period_ns = int(new_period_s * 1.0e9)


def make_periodic_timer_node(
    name: str,
    *,
    frequency: float | None = None,
    period_s: float | None = None,
    duration_s: float | None = None,
) -> Node:
    timer = PeriodicTimer(frequency=frequency, period_s=period_s, duration_s=duration_s)
    # Advertise this timer's period to the editor's control plane so the
    # debugger's Step advances virtual time by min(period) over the selected
    # timers and the Step-timer picker lists it. No-op when the tap isn't
    # installed (running outside the editor).
    try:
        debug_tap.register_subordinate_period(name, timer._period_ns)  # noqa: SLF001
    except Exception:  # (never let telemetry registration break the graph)
        pass
    return Node(name, timer)


def make_variable_period_timer_nodes(
    name: str,
    *,
    frequency: float | None = None,
    period_s: float | None = None,
    duration_s: float | None = None,
) -> list[Node]:
    timer = PeriodicTimer(frequency=frequency, period_s=period_s, duration_s=duration_s)
    return [
        Node(name, timer),
        Node(name + "_update_period", timer.update_period_s),
    ]
