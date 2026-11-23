# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import multiprocessing as mp
from inspect import signature

import numpy as np

from judo.unlocked.logging_util import setup_logger
from judo.unlocked import InputStage, Node, OutputStage, SharedTimeStat, const_cast
from judo.unlocked.benchmark.common import N_TRIALS, SIZES
from judo.unlocked.policy import Lossless

setup_logger(log_level=logging.INFO)

n_sub = 30
payload = np.random.randint(256, size=SIZES[0], dtype=np.uint8)  # expected 4K image payload size
payload = 0


def _loop(msg: np.ndarray) -> np.ndarray:
    return const_cast(msg)


def benchmark_stages() -> None:
    output_push_time = SharedTimeStat("output_push_time", lock=False)
    input_wait_time = SharedTimeStat("output_push_time", lock=False)
    input_args_time = SharedTimeStat("output_push_time", lock=False)
    input_next_time = SharedTimeStat("output_push_time", lock=False)

    sig = signature(_loop)
    output_stage = OutputStage("benchmark_stages", sig.return_annotation)
    input_stage = InputStage("benchmark_stages", sig)
    input_stage.connect((output_stage[0], Lossless()))

    output_stage.open()
    input_stage.open()

    for _ in range(N_TRIALS):
        with output_push_time:
            output_stage.push((payload,))
        with input_wait_time:
            input_stage.wait()
        with input_args_time:
            input_stage.input_args()
        with input_next_time:
            input_stage.next()

    input_stage.close()
    output_stage.close()

    print(f"                Output stage push time: {output_push_time}")
    print(f"                 Input stage wait time: {input_wait_time}")
    print(f"                 Input stage args time: {input_args_time}")
    print(f"                 Input stage next time: {input_next_time}")


def benchmark_stages_multiprocessing() -> None:
    output_push_time = SharedTimeStat("output_push_time", lock=False)
    input_wait_time = SharedTimeStat("output_push_time", lock=False)
    input_args_time = SharedTimeStat("output_push_time", lock=False)
    input_next_time = SharedTimeStat("output_push_time", lock=False)

    sig = signature(_loop)
    output_stage = OutputStage("benchmark_stages", sig.return_annotation)
    input_stage = InputStage("benchmark_stages", sig)
    input_stage.connect((output_stage[0], Lossless()))

    sync = mp.Barrier(2)

    def push_fn() -> None:
        output_stage.open()
        for _ in range(N_TRIALS):  # publish more not to deprive views of data
            with output_push_time:
                output_stage.push((payload,))
        sync.wait()
        output_stage.close()

    def view_fn() -> None:
        input_stage.open()
        for _ in range(N_TRIALS):  # publish more not to deprive views of data
            with input_wait_time:
                input_stage.wait()
            with input_args_time:
                input_stage.input_args()
            with input_next_time:
                input_stage.next()
        sync.wait()
        input_stage.close()

    processes = [mp.Process(target=push_fn, name="push"), mp.Process(target=view_fn, name="view")]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    print(f"Multiprocessing output stage push time: {output_push_time}")
    print(f" Multiprocessing input stage wait time: {input_wait_time}")
    print(f" Multiprocessing input stage args time: {input_args_time}")
    print(f" Multiprocessing input stage next time: {input_next_time}")


def benchmark_stages_in_node() -> None:
    node_wait_time = SharedTimeStat("node_wait_time", lock=False)
    node_exec_time = SharedTimeStat("node_exec_time", lock=False)

    node = Node("benchmark_stages", _loop)
    node.input_stage.connect((node.output_stage[0], Lossless()))

    node.output_stage.open()
    node.input_stage.open()
    node.open()
    node.output_stage.push((payload,))

    for _ in range(N_TRIALS):
        with node_wait_time:
            node.wait()
        with node_exec_time:
            node.exec()

    node.close()
    node.input_stage.close()
    node.output_stage.close()

    print(f"                        Node wait time: {node_wait_time}")
    print(f"                        Node exec time: {node_exec_time}")


def main() -> None:
    benchmark_stages()
    benchmark_stages_multiprocessing()
    benchmark_stages_in_node()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
