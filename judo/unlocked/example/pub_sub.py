# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import signal

# from operator import add
from judo.unlocked.logging_util import setup_logger
from judo.unlocked import Node
from judo.unlocked.nodes.periodic_timer import make_periodic_timer_node
from judo.unlocked.policy import Latest, Lossless
from judo.unlocked.schedule import StopAll, Threaded

setup_logger(log_level=logging.INFO)


class Publisher:
    """Publishing generator."""

    def __init__(self) -> None:
        self._counter = 0

    def __call__(self, _: int) -> int:
        self._counter += 1
        return self._counter


def subscriber(_: int, msg: int) -> None:
    """Subscriber printout."""
    logging.info("Received %d", msg)


def main() -> None:
    fast = make_periodic_timer_node("fast_timer", frequency=10_000)  # 10 kHz!
    slow = make_periodic_timer_node("slow_timer", period_s=1)
    pub = Node("pub", Publisher())
    sub = Node("sub", subscriber)

    pub.input_stage.connect((fast.output_stage[0], Lossless()))
    sub.input_stage.connect(
        (slow.output_stage[0], Lossless()),
        msg=(pub.output_stage[0], Latest()),
    )

    schedule1 = Threaded(
        "sch_1",
        (
            fast,
            slow,
            pub,
            sub,
        ),
    )
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((schedule1,)))
    schedule1.join()


if __name__ == "__main__":
    main()
