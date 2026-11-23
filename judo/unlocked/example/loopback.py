# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import signal

# from operator import add
from judo.unlocked.logging_util import setup_logger
from judo.unlocked import Node, node_outputs
from judo.unlocked.nodes.periodic_timer import make_periodic_timer_node
from judo.unlocked.policy import Latest, Lossless
from judo.unlocked.schedule import StopAll, Threaded

setup_logger(log_level=logging.INFO, include_thread=True)


@node_outputs("next")
def loop_the_loop(_tick: int, msg: int) -> int:
    """Subscriber printout."""
    return msg + 1


def subscriber(_tick: int, msg: int) -> None:
    logging.info("Received %d", msg)


def main() -> None:
    fast = make_periodic_timer_node("fast_timer", frequency=10_000)  # 10 kHz!
    loop = Node("loop", loop_the_loop, warmup=[(0,)])
    slow = make_periodic_timer_node("slow_timer", period_s=1)
    subs = Node("subs", subscriber)

    loop.input_stage.connect(
        (fast.output_stage[0], Lossless()),
        (loop.output_stage[0], Lossless()),
    )
    subs.input_stage.connect(
        (slow.output_stage[0], Lossless()),
        msg=(loop.output_stage[0], Latest()),
    )

    schedule1 = Threaded("sch_1", (fast, slow, loop, subs))
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((schedule1,)))
    schedule1.join()


if __name__ == "__main__":
    main()
