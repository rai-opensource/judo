# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import signal

import numpy as np

from judo.unlocked.logging_util import setup_logger
from judo.unlocked.benchmark.common import N_TRIALS, SIZES, record
from judo.unlocked.node import Node, NodeStop
from judo.unlocked.nodes.periodic_timer import make_periodic_timer_node
from judo.unlocked.policy import Lossless
from judo.unlocked.schedule import StopAll, Threaded
from judo.unlocked.time_stat import SharedTimeStat, format_binary_prefix

setup_logger(log_level=logging.INFO)

time_stat = SharedTimeStat("pub_sub_stats", lock=False)


class Publisher:
    """Publishing generator."""

    def __init__(self) -> None:
        self._trial_no = 0
        self._size_ind = 0

    def __call__(self, _: int) -> np.array:
        """Publisher."""
        if self._trial_no == N_TRIALS:
            self._trial_no = 0
            self._size_ind += 1
        if self._size_ind == len(SIZES):
            raise NodeStop("Done")

        result = np.random.randint(256, size=SIZES[self._size_ind], dtype=np.ubyte)
        self._trial_no += 1
        time_stat.tic()
        return result


class Subscriber:
    """Latency mesuring subscriber."""

    def __init__(self) -> None:
        self._trial_no = 0
        self._size_ind = 0

    def __call__(self, msg: np.array) -> None:
        """Subscriber."""
        latency = time_stat.toc()
        size = len(msg)
        record("unLocked", size, latency)

        self._trial_no += 1
        if self._trial_no == N_TRIALS:
            print(f"message of size {format_binary_prefix(size, 'B')}:", time_stat)
            self._trial_no = 0
            self._size_ind += 1
            time_stat.clear()

        if self._size_ind == len(SIZES):
            raise NodeStop("Done")


def main() -> None:
    timer = make_periodic_timer_node("timer", frequency=5, duration_s=N_TRIALS * len(SIZES) * 1.1 / 5)
    pub = Node("pub", Publisher())
    sub = Node("sub", Subscriber())
    pub.input_stage.connect((timer.output_stage[0], Lossless()))
    sub.input_stage.connect(msg=(pub.output_stage[0], Lossless()))

    schedule1 = Threaded(
        "sch_1",
        (
            timer,
            pub,
        ),
    )
    schedule2 = Threaded("sch_2", (sub,))
    schedule1.start()
    schedule2.start()
    signal.signal(signal.SIGINT, StopAll((schedule1, schedule2)))
    schedule1.join()
    schedule2.join()


if __name__ == "__main__":
    main()
