# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Single virtual-time master clock for the unlocked graph.

The master clock is one Node, intended to live in its own ``timer_scheduler``
process alongside every :class:`SubordinateTimer` in the graph. It owns
``virtual_ns`` -- the only authoritative source of time for nodes that
opt into the master-clock pacing -- and publishes it to subscribers on
every tick.

Three modes drive ``virtual_ns``:

* **Running**: advance by ``resolution_ns`` once per ``resolution_ns / rate``
  of wall time. ``rate`` defaults to 1.0 (virtual time tracks wall time).
* **Paused**: do not advance. Subordinate timers block on their input wait,
  so every downstream node naturally idles.
* **Stepping**: advance freely (no wall sleep) until ``virtual_ns`` reaches
  ``step_target_ns``, then flip back to *Paused*. This is how the editor's
  "Step" button advances by ``min(period)`` over the configured step group.

Control state (``paused``, ``step_target_ns``, ``rate``) is read from
:mod:`judo.unlocked.debug_tap`, which already mirrors the editor's
``/control`` snapshot once per ~100 ms.
"""

from __future__ import annotations

import os
import sys
from time import perf_counter_ns, sleep

from judo.unlocked import debug_tap
from judo.unlocked.node import Node

_DIAG = os.environ.get("UNLOCKED_MASTER_CLOCK_DIAG", "1") != "0"


class MasterClock:
    """Virtual-time generator that gates every :class:`SubordinateTimer`."""

    def __init__(self, *, resolution_s: float = 0.001) -> None:
        self._resolution_ns = max(1, int(resolution_s * 1.0e9))
        self._virtual_ns = 0
        self._step_target_ns = 0
        # Lightweight per-second tick counter so a single stderr line every
        # ~1 s tells the user "yes, the master is actually ticking at N Hz"
        # without flooding logs. Disabled by setting UNLOCKED_MASTER_CLOCK_DIAG=0.
        self._diag_ticks = 0
        self._diag_last_ns = perf_counter_ns()
        self._diag_paused = None  # type: bool | None

    @property
    def virtual_ns(self) -> int:
        """Current virtual-clock value in nanoseconds."""
        return self._virtual_ns

    def __call__(self) -> int | None:
        """Advance ``virtual_ns`` by one resolution unit and publish it.

        Returns ``None`` to skip publishing when paused with no pending step
        (downstream subordinates simply block on their next input).
        """
        # Pull live control state. ``debug_tap`` writes these atomically on
        # every ``/control`` poll; reads here are best-effort (no lock needed
        # since reads are intrinsic and a stale value just costs a cycle).
        rate = debug_tap.get_clock_rate()
        # Drain any pending step bump issued by the editor's Step button.
        pending = debug_tap.consume_clock_step_pending_ns()
        if pending > 0:
            self._step_target_ns = max(self._step_target_ns, self._virtual_ns + pending)

        stepping = self._step_target_ns > self._virtual_ns
        paused = debug_tap.is_clock_paused()
        if _DIAG and paused != self._diag_paused:
            print(
                f"[master_clock] paused={paused} virtual_ns={self._virtual_ns} "
                f"step_target_ns={self._step_target_ns}",
                file=sys.stderr,
                flush=True,
            )
            self._diag_paused = paused
            # Reset the rate window so the first reported Hz measures
            # post-resume ticks against post-resume wall time, not against
            # the long paused interval that preceded it.
            self._diag_ticks = 0
            self._diag_last_ns = perf_counter_ns()

        if paused and not stepping:
            # Sleep briefly so we don't hot-loop the scheduler thread; any new
            # /control snapshot will be picked up on the next iteration.
            sleep(self._resolution_ns * 1.0e-9)
            return None

        if not stepping:
            # Wall-pace one resolution unit at the configured rate. ``rate``
            # < 1.0 runs slower than wall time; > 1.0 runs faster.
            sleep((self._resolution_ns * 1.0e-9) / max(rate, 1.0e-6))

        self._virtual_ns += self._resolution_ns

        if _DIAG:
            self._diag_ticks += 1
            now_ns = perf_counter_ns()
            if now_ns - self._diag_last_ns >= 1_000_000_000:
                rate = self._diag_ticks * 1.0e9 / (now_ns - self._diag_last_ns)
                print(
                    f"[master_clock] ticking at {rate:.1f} Hz "
                    f"(virtual_ns={self._virtual_ns}, paused={paused}, stepping={stepping})",
                    file=sys.stderr,
                    flush=True,
                )
                self._diag_ticks = 0
                self._diag_last_ns = now_ns

        if stepping and self._virtual_ns >= self._step_target_ns:
            # Reached the requested step target -- snap to it and re-pause.
            self._virtual_ns = self._step_target_ns
            self._step_target_ns = 0
            debug_tap.notify_clock_step_consumed()
        return self._virtual_ns


def make_master_clock_node(name: str = "master_clock", *, resolution_s: float = 0.001) -> Node:
    """Build the master-clock Node. Output: ``int`` virtual nanoseconds."""
    return Node(name, MasterClock(resolution_s=resolution_s))
