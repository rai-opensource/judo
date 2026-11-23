# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Fibonacci graph described via the declarative GraphBuilder.

Mirrors `judo/unlocked/example/fibonacci.py`:
    timer.tick     -> fibonacci._tick
    fibonacci.b    -> fibonacci.a
    fibonacci.next -> fibonacci.b
"""

from __future__ import annotations

from judo.unlocked.example.fibonacci import fibonacci
from judo.unlocked.graph.builder import GraphBuilder


def build() -> GraphBuilder:
    g = GraphBuilder(name="fibonacci")
    g.timer("timer", frequency=10)
    # fibonacci(_tick, a, b) -> tuple[int, int]; ports from @node_outputs("b", "next").
    g.node("fibonacci", fibonacci, warmup=[(0, 1)])
    g.connect("timer.tick", "fibonacci._tick", policy="lossless")
    g.connect("fibonacci.b", "fibonacci.a", policy="lossless")
    g.connect("fibonacci.next", "fibonacci.b", policy="lossless")
    return g


if __name__ == "__main__":
    print(build().to_json())
