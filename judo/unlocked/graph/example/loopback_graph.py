# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Loopback graph described via the declarative GraphBuilder.

Mirrors `judo/unlocked/example/loopback.py`:
    fast_timer.tick -> loop._tick
    loop.next       -> loop.msg      (self-loop, lossless)
    slow_timer.tick -> subs._tick
    loop.next       -> subs.msg      (latest)
"""

from __future__ import annotations

from judo.unlocked.example.loopback import loop_the_loop, subscriber
from judo.unlocked.graph.builder import GraphBuilder


def build() -> GraphBuilder:
    g = GraphBuilder(name="loopback")
    g.timer("fast_timer", frequency=10_000)
    g.timer("slow_timer", period_s=1.0)
    # loop_the_loop(_tick, msg) -> int : output port from @node_outputs("next").
    g.node("loop", loop_the_loop, warmup=[(0,)])
    g.node("subs", subscriber)

    g.connect("fast_timer.tick", "loop._tick", policy="lossless")
    g.connect("loop.next", "loop.msg", policy="lossless")
    g.connect("slow_timer.tick", "subs._tick", policy="lossless")
    g.connect("loop.next", "subs.msg", policy="latest")
    return g


if __name__ == "__main__":
    print(build().to_json())
