# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Subordinate timer node driven by :class:`MasterClock`.

A subordinate timer subscribes to the master clock's ``virtual_ns`` output
and republishes it on its own configured period. Subordinates have no wall-
clock awareness: they fire when master crosses their next virtual deadline
and idle otherwise. This is what makes Play/Pause/Step on the master
deterministically gate every downstream node in the graph.

Periods are also reported to the editor via
:func:`judo.unlocked.debug_tap.register_subordinate_period` so the Step
button can compute the smallest advance that fires at least one timer in
the step group (i.e. the next-event step semantics chosen for the editor).
"""

from __future__ import annotations

from judo.unlocked import debug_tap
from judo.unlocked.node import Node


class SubordinateTimer:
    """Republish master ``virtual_ns`` at the subordinate's own period."""

    def __init__(self, name: str, *, period_s: float) -> None:
        assert period_s > 0, "period_s must be positive"
        self._name = name
        self._period_ns = int(period_s * 1.0e9)
        self._next_deadline_ns: int | None = None
        # Announce this timer's period to the editor's step-group machinery.
        # No-op when running outside the editor (debug_tap not installed).
        debug_tap.register_subordinate_period(name, self._period_ns)

    def __call__(self, master_tick_ns: int) -> int | None:
        """Fire if ``master_tick_ns`` has crossed the next virtual deadline."""
        if self._next_deadline_ns is None:
            # First tick: align deadline to the master's current virtual time
            # so the timer fires immediately on the next crossing rather than
            # waiting for the absolute multiple of period from t=0.
            self._next_deadline_ns = master_tick_ns + self._period_ns
            return master_tick_ns
        if master_tick_ns < self._next_deadline_ns:
            return None
        # Advance deadline by exactly one period: at min-period step semantics
        # the master never overshoots a subordinate by more than one period,
        # so we always fire exactly once per crossing.
        self._next_deadline_ns += self._period_ns
        return master_tick_ns


def make_subordinate_timer_node(name: str, *, period_s: float | None = None, frequency: float | None = None) -> Node:
    """Build a subordinate timer Node. Input: master tick (int ns), output: int ns."""
    assert (period_s is None) ^ (frequency is None), "Exactly one of 'period_s' or 'frequency' must be specified."
    if period_s is None:
        assert frequency is not None
        period_s = 1.0 / frequency
    return Node(name, SubordinateTimer(name, period_s=period_s))
