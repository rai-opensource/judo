# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import signal

# from operator import add
from judo.unlocked.logging_util import setup_logger
from judo.unlocked import Node, node_outputs
from judo.unlocked.nodes.periodic_timer import make_periodic_timer_node
from judo.unlocked.policy import Lossless
from judo.unlocked.schedule import StopAll, Threaded

setup_logger(log_level=logging.DEBUG)


@node_outputs("b", "next")
def fibonacci(_tick: int, a: int, b: int) -> tuple[int, int]:
    """Subscriber printout."""
    next = a + b
    logging.info("Next fibonacci number is %d", next)
    return b, next


def main() -> None:
    f_node = Node("fibonacci", fibonacci, warmup=[(0, 1)])
    # Note timer precission in the logger output (the last two digits in milliseconds)
    timer = make_periodic_timer_node("timer", frequency=10)

    f_node.input_stage.connect(
        (timer.output_stage[0], Lossless()),
        (f_node.output_stage[0], Lossless()),
        (f_node.output_stage[1], Lossless()),
    )

    schedule1 = Threaded(
        "sch_1",
        (
            timer,
            f_node,
        ),
    )
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((schedule1,)))
    schedule1.join()


if __name__ == "__main__":
    main()
