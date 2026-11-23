# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import signal

# from operator import add
from judo.unlocked.logging_util import setup_logger
from judo.unlocked import Node
from judo.unlocked.nodes import make_logger_node, make_periodic_timer_node
from judo.unlocked.policy import Lossless, OneOf
from judo.unlocked.schedule import StopAll, Threaded

setup_logger(log_level=logging.INFO)


def _exactly_one(input: tuple) -> bool:
    result = False
    for data in input:
        if data is None:
            continue
        if result:
            return False
        result = True
    return result


def _mux(*input: str | None) -> str | None:
    assert _exactly_one(input), f"OneOf policy failed with inputs {input}"
    for data in input:
        if data is None:
            continue
        return data
    return None


def _publisher(_: int, data: str, counter: int) -> tuple[str, int]:
    return f"{data} message {counter}", counter + 1


def main() -> None:
    timer_a = make_periodic_timer_node(name="timer_a", frequency=3)
    publisher_a = Node(name="publisher_a", target=_publisher, warmup=[(None, 600)])
    publisher_a.input_stage.connect(
        (timer_a.output_stage[0], Lossless()),
        "Publisher A",
        (publisher_a.output_stage[1], Lossless()),
    )
    timer_b = make_periodic_timer_node(name="timer_b", frequency=5)
    publisher_b = Node(name="publisher_b", target=_publisher, warmup=[(None, 700)])
    publisher_b.input_stage.connect(
        (timer_b.output_stage[0], Lossless()),
        "Publisher B",
        (publisher_b.output_stage[1], Lossless()),
    )
    mux = Node(name="multiplexer_node", target=_mux)
    one_of = OneOf()
    mux.input_stage.connect(
        (publisher_a.output_stage[0], one_of),
        (publisher_b.output_stage[0], one_of),
    )
    logger = make_logger_node(name="logger", level=logging.INFO, msg="%s")
    logger.input_stage.connect((mux.output_stage[0], Lossless()))

    schedule1 = Threaded(
        "sch_1",
        (
            timer_a,
            publisher_a,
            timer_b,
            publisher_b,
            mux,
            logger,
        ),
    )
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((schedule1,)))
    schedule1.join()


if __name__ == "__main__":
    main()
