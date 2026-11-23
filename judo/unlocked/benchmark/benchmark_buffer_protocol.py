# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from array import array
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from judo.unlocked.benchmark.common import N_TRIALS, SIZES, record
from judo.unlocked.time_stat import SharedTimeStat, format_binary_prefix


def _benchmark_loop(name: str, destination_factory: Any) -> None:
    time_stat = SharedTimeStat("time_stat", lock=False)
    for size in SIZES:
        destination = destination_factory(size * N_TRIALS)
        destination_memory = memoryview(destination).cast("B")
        for i in range(N_TRIALS):
            payload = np.random.randint(256, size=size, dtype=np.ubyte)
            time_stat.tic()
            destination_memory[i * size : (i + 1) * size] = memoryview(payload).cast("B")
            latency = time_stat.toc()
            record(name, size, latency)

        del destination_memory
        print(f"{name:>13} of size {format_binary_prefix(size, 'B')}:", time_stat)
        time_stat.clear()


def benchmark_bytearray() -> None:
    def bytearray_factory(size: int) -> bytearray:
        return bytearray(size)

    _benchmark_loop("bytearray", bytearray_factory)


def benchmark_array() -> None:
    def array_factory(size: int) -> array:
        return array("B", b"\x00" * size)

    _benchmark_loop("array", array_factory)


def benchmark_numpy_array() -> None:
    def numpy_factory(size: int) -> np.array:
        return np.zeros(size, dtype=np.uint8)

    _benchmark_loop("numpy array", numpy_factory)


def benchmark_shared_memory() -> None:
    class SharedMemoryFactory:
        def __init__(self) -> None:
            self._shm: SharedMemory | None = None

        def __call__(self, size: int) -> memoryview:
            self._close()
            self._shm = SharedMemory("benchmark_latency", create=True, size=size)
            # SharedMemory.buf is only None after close() is called
            assert self._shm.buf is not None
            return self._shm.buf

        def __del__(self) -> None:
            self._close()

        def _close(self) -> None:
            if self._shm is not None:
                self._shm.close()
                self._shm.unlink()
                self._shm = None

    _benchmark_loop("shared memory", SharedMemoryFactory())


def main() -> None:
    benchmark_bytearray()
    benchmark_array()
    benchmark_numpy_array()
    benchmark_shared_memory()


if __name__ == "__main__":
    main()
