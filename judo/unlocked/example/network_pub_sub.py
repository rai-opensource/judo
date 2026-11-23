# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import signal
from enum import Enum, auto

import tyro

# from operator import add
from judo.unlocked.logging_util import setup_logger
from judo.unlocked import Node
from judo.unlocked.nodes.network import make_receiver_node, make_sender_node
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


class Role(Enum):
    Publisher = auto()
    Subscriber = auto()


def main_publisher(host: str, port: int) -> None:
    fast = make_periodic_timer_node("fast_timer", frequency=5_000)  # 5 kHz!
    pub = Node("pub", Publisher())
    sender = make_sender_node("sender", host, port)
    pub.input_stage.connect((fast.output_stage[0], Lossless()))
    sender.input_stage.connect((pub.output_stage[0], Lossless()))
    schedule1 = Threaded(
        "sch_1",
        (
            fast,
            pub,
            sender,
        ),
    )
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((schedule1,)))
    schedule1.join()


def main_subscriber(host: str, port: int) -> None:
    receiver = make_receiver_node("receiver", host, port)
    slow = make_periodic_timer_node("slow_timer", period_s=1)
    sub = Node("sub", subscriber)
    sub.input_stage.connect(
        (slow.output_stage[0], Lossless()),
        msg=(receiver.output_stage[0], Latest()),
    )
    schedule1 = Threaded(
        "sch_1",
        (
            receiver,
            slow,
            sub,
        ),
    )
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((schedule1,)))
    schedule1.join()


def main(role: Role, host: str, port: int) -> None:
    if role == Role.Publisher:
        main_publisher(host, port)
    else:
        main_subscriber(host, port)


if __name__ == "__main__":
    tyro.cli(main)
