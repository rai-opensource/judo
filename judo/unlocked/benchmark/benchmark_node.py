# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import signal

import numpy as np

from judo.unlocked.logging_util import setup_logger
from judo.unlocked.benchmark.common import N_TRIALS, SIZES
from judo.unlocked.node import Node, NodeStop
from judo.unlocked.nodes.periodic_timer import make_periodic_timer_node
from judo.unlocked.policy import Lossless
from judo.unlocked.schedule import StopAll, Threaded
from judo.unlocked.time_stat import SharedTimeStat

setup_logger(log_level=logging.WARNING)


time_stat = SharedTimeStat("pub_sub_stats", lock=False)


class Publisher:
    """Publishing generator."""

    def __init__(self) -> None:
        self._trial_no = 0

    def __call__(self, _: int) -> int:
        """Publish 1080p image"""
        if self._trial_no == N_TRIALS:
            raise NodeStop("Done")
        result = np.random.randint(256, size=SIZES[7], dtype=np.ubyte)  # 1080p @ 24 bit
        self._trial_no += 1
        time_stat.tic()
        return result


class Subscriber:
    """Subscriber."""

    def __init__(self) -> None:
        self._trial_no = 0

    def __call__(self, msg: np.array) -> None:
        """Subscribe to 1080p image"""
        time_stat.toc()
        logging.debug("subscriber received: %s", msg)
        self._trial_no += 1
        if self._trial_no == N_TRIALS:
            raise NodeStop("Done")


def benchmark_node_init() -> None:
    """Benchmark node init time"""
    init_time = SharedTimeStat("init_time", lock=False)

    for _ in range(N_TRIALS):
        with init_time:
            Node("init_node", Publisher())

    print(f"                      Node init time: {init_time}")


def benchmark_pub_sub_intraprocess() -> None:
    """Benchmark data transfer time across nodes in the same process"""
    timer = make_periodic_timer_node("timer", frequency=10, duration_s=1.2 * N_TRIALS / 10)
    pub = Node("intra_pub", Publisher())
    sub = Node("intra_sub", Subscriber())
    pub.input_stage.connect((timer.output_stage[0], Lossless()))
    sub.input_stage.connect(msg=(pub.output_stage[0], Lossless()))
    time_stat.clear()  # reset time_stat

    # single process
    schedule1 = Threaded("sch_intra", (timer, pub, sub))
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((schedule1,)))
    schedule1.join()

    print(f"Intraprocess Node data transfer time: {time_stat}")
    print(f"            {pub.name} Node idle time: {pub.wait_time}")
    print(f"            {pub.name} Node exec time: {pub.exec_time}")
    print(f"            {sub.name} Node idle time: {sub.wait_time}")
    print(f"            {sub.name} Node exec time: {sub.exec_time}")


def benchmark_pub_sub_interprocess() -> None:
    """Benchmark data transfer time across nodes in different processes"""
    timer = make_periodic_timer_node("timer", frequency=10, duration_s=1.2 * N_TRIALS / 10)
    pub = Node("inter_pub", Publisher())
    sub = Node("inter_sub", Subscriber())
    pub.input_stage.connect((timer.output_stage[0], Lossless()))
    sub.input_stage.connect(msg=(pub.output_stage[0], Lossless()))
    time_stat.clear()  # reset time_stat

    # Multiple processes
    schedule1 = Threaded(
        "sch_inter_1",
        (
            timer,
            pub,
        ),
    )
    schedule2 = Threaded("sch_inter_2", (sub,))
    schedule1.start()
    schedule2.start()
    signal.signal(signal.SIGINT, StopAll((schedule1, schedule2)))  # SLF001
    schedule1.join()
    schedule2.join()

    print(f"Interprocess Node data transfer time: {time_stat}")
    print(f"            {pub.name} Node idle time: {pub.wait_time}")
    print(f"            {pub.name} Node exec time: {pub.exec_time}")
    print(f"            {sub.name} Node idle time: {sub.wait_time}")
    print(f"            {sub.name} Node exec time: {sub.exec_time}")


def main() -> None:
    benchmark_node_init()
    benchmark_pub_sub_intraprocess()
    benchmark_pub_sub_interprocess()


if __name__ == "__main__":
    main()
