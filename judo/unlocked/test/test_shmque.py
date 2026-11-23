# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import multiprocessing as mp
import random
import unittest
from ctypes import c_bool
from dataclasses import dataclass
from time import sleep
from typing import Any, Callable

import numpy as np
from parameterized import parameterized_class
from petname import generate

from judo.unlocked import SIZEOF_SIZE_T, Frame, Memory, SharedMemoryQueue

_instance_name: str = generate(2, "_")


@dataclass
class MyPair:
    first: float
    second: float


random.seed(785672498529)


def int_gen(_: Any, size: int) -> list:
    return [random.randint(0, 1_000_000_000) for _ in range(size)]


def pair_gen(_: Any, size: int) -> list:
    return [MyPair(first=random.uniform(0, 1), second=random.uniform(0, 1)) for _ in range(size)]


def array_gen(_: Any, size: int) -> list:
    return [np.random.randint(1_000_000_000, size=(10, 10)) for _ in range(size)]


def int_eq(_: Any, a: int, b: int) -> bool:
    return a == b


def pair_eq(_: Any, a: MyPair, b: MyPair) -> bool:
    return a.first == b.first and a.second == b.second


def array_eq(_: Any, a: MyPair, b: MyPair) -> bool:
    return np.array_equal(a, b)


type_gen_params = [
    ("scalar_" + _instance_name, int_gen, int_eq),
    ("dataclass_" + _instance_name, pair_gen, pair_eq),
    ("array_" + _instance_name, array_gen, array_eq),
]


class TestSharedMemory(unittest.TestCase):
    """Test if the memory is shared between two views."""

    def test_shared_array(self) -> None:
        t = np.arange(10)
        t_mem: Memory = Memory(t)
        t_bytes = bytearray(len(t_mem))
        t_mem.encode(t_bytes)
        tt1 = Memory.decode(t_bytes)
        tt2 = Memory.decode(t_bytes)
        self.assertIsNot(tt1, tt2)
        self.assertTrue(np.array_equal(tt1, tt2), msg=f"{tt1} is not equal to {tt2}")
        tt1[...] = 42
        self.assertTrue(np.array_equal(tt1, tt2), msg=f"{tt1} is not equal to {tt2}")


@parameterized_class(("name_suffix", "gen", "eq"), type_gen_params)
class TestShmque(unittest.TestCase):
    """Test shared memory queue."""

    name_suffix: str
    eq: Callable
    gen: Callable

    def test_memory(self) -> None:
        t = self.gen(1)[0]
        t_mem: Memory = Memory(t)
        t_bytes = bytearray(len(t_mem))
        t_mem.encode(t_bytes)
        tt = Memory.decode(t_bytes)
        self.assertIsNot(t, tt)
        self.assertTrue(self.eq(t, tt), msg=f"{t} is not equal to {tt}")

    def test_frame(self) -> None:
        values = self.gen(5)
        frame: Frame = Frame(name="test_frame_shm_" + self.name_suffix, create=True, size=1024 * 10)
        frame_view: Frame = Frame(name="test_frame_shm_" + self.name_suffix, create=False)

        self.assertEqual(len(frame), 0)
        self.assertEqual(len(frame_view), 0)
        self.assertGreaterEqual(frame.size(), 1024 * 10)
        self.assertEqual(frame.free(), frame.size() - SIZEOF_SIZE_T)
        self.assertEqual(frame.used(), SIZEOF_SIZE_T)

        for val in values:
            frame.push(val)

        self.assertEqual(len(frame), 5)
        self.assertEqual(len(frame_view), 5)
        self.assertEqual(frame.free() + frame.used(), frame.size())
        self.assertGreater(frame.used(), SIZEOF_SIZE_T)

        for index in range(5):
            self.assertTrue(self.eq(frame[index], values[index]), msg=f"error at index {index}")
            self.assertTrue(self.eq(frame_view[index], values[index]), msg=f"view error at index {index}")

    def test_frame_multiprocessing(self) -> None:
        def producer(name: str, values: list, cond: Any, events: list) -> None:
            fail = False
            frame: Frame = Frame(name=name + self.name_suffix, create=True, size=1024 * 10)
            frame_created_ev.set()
            try:
                for val in values:
                    frame.push(val)
                self.assertEqual(len(frame), 5)
                with cond:
                    cond.notify_all()
                while not all([event.is_set() for event in events]):
                    sleep(random.uniform(0.05, 0.1))
            except Exception as e:
                print(e)
                fail = True
            if fail:
                exit(1)

        def consumer(name: str, values: list, cond: Any, ind: int, events: list) -> None:
            # Ensure the producer has created the shared resource
            frame_created_ev.wait()
            fail = False
            frame: Frame = Frame(name=name + self.name_suffix, create=False)
            try:
                with cond:
                    cond.wait_for(lambda: len(frame) == 5)
                self.assertEqual(len(frame), 5)
                for index in range(5):
                    self.assertTrue(self.eq(frame[index], values[index]), msg=f"view error at index {index}")
                events[ind].set()
                while not all([event.is_set() for event in events]):
                    sleep(random.uniform(0.05, 0.1))
            except Exception as e:
                print(e)
                events[ind].set()
                fail = True
            if fail:
                exit(1)

        name = "test_frame_multiprocessing_"
        values = self.gen(5)
        sub_num = 10
        manager = mp.Manager()
        cond = manager.Condition()
        events = [manager.Event() for _ in range(sub_num)]
        frame_created_ev = manager.Event()

        procs = [mp.Process(target=producer, args=(name, values, cond, events), daemon=True)] + [
            mp.Process(target=consumer, args=(name, values, cond, ind, events), daemon=True) for ind in range(sub_num)
        ]

        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

        self.assertTrue(all((proc.exitcode == 0 for proc in procs)))

    def test_shmque(self) -> None:
        n = 10
        values = self.gen(n)
        queue: SharedMemoryQueue = SharedMemoryQueue(name="test_shmque_" + self.name_suffix, min_rate=1)
        view1 = queue.make_view()
        view2 = queue.make_view()
        queue.open()
        view1.open()
        view2.open()

        test_time = 0.1
        delay = 2 * test_time / n

        for val in values:
            sleep(random.uniform(0, delay))
            queue.push(val)

        sleep(0.01)
        self.assertEqual(len(queue), n)
        self.assertEqual(len(view2), n)
        self.assertEqual(len(view2), n)

        for index in range(n):
            random_index = random.randint(0, n - 1)
            if view1.begin <= random_index and random_index < view1.end:
                self.assertTrue(
                    self.eq(view1[random_index], values[random_index]), msg=f"view error at index {random_index}"
                )
            else:
                with self.assertRaises(IndexError):
                    view1[random_index]

            self.assertTrue(self.eq(view1[view1.begin], values[index]), msg=f"view error at index {index}")
            self.assertTrue(self.eq(view1.top(), values[index]), msg=f"view error at index {index}")
            view1.pop()

        self.assertEqual(len(view1), 0)
        self.assertEqual(len(view2), n)
        self.assertEqual(len(queue), n)

        view2.pop(n)

        self.assertEqual(len(view2), 0)

        queue.flush()
        self.assertLess(len(queue), n)

        view1.close()
        view2.close()
        queue.close()

    def test_shmque_stress(self) -> None:
        random.seed(47299561901461)

        sub_num = 10
        act_num = 100
        test_time = 0.1
        delay = 2 * test_time / act_num

        queue: SharedMemoryQueue = SharedMemoryQueue(name="test_shmque_stress_" + self.name_suffix, min_rate=1)
        views = [queue.make_view() for _ in range(sub_num)]

        values = self.gen(act_num)

        success = mp.RawArray(c_bool, [False] * (sub_num + 1))
        barrier = mp.Barrier(sub_num + 1)

        def publisher() -> None:
            queue.open()
            barrier.wait()
            for val in values:
                sleep(random.uniform(0, delay))
                queue.push(val)
            barrier.wait()
            queue.close()
            success[0] = True

        def subscriber(ind: int) -> None:
            view = views[ind]
            try:
                view.open()
                barrier.wait()
                for val in values:
                    while not view.wait():
                        pass
                    if not self.eq(view.top(), val):
                        print(f"Subscriber {self.name_suffix}_{ind} expected {val}, but got {view.top()}")
                        return
                    index = random.randint(view.begin, view.end - 1)
                    if not self.eq(view[index], values[index]):
                        print(
                            f"Subscriber {self.name_suffix}_{ind} at {index} "
                            f"expected {values[index]}, but got {view[index]}"
                        )
                        return
                    view.pop()
            finally:
                view.close()
                barrier.wait()
            success[ind + 1] = True

        procs = [mp.Process(target=publisher)] + [
            mp.Process(target=subscriber, args=(sub_ind,)) for sub_ind in range(sub_num)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
        self.assertTrue(all(success))


if __name__ == "__main__":
    unittest.main()


class TestProducerClosed(unittest.TestCase):
    """A consumer mid-step when its producer closes the queue (shutdown) must not try to attach unlinked frames."""

    def test_pop_after_producer_close(self) -> None:
        queue: SharedMemoryQueue = SharedMemoryQueue(name="test_closed_" + _instance_name, min_rate=1)
        view = queue.make_view()
        queue.open()
        view.open()
        for i in range(3):
            queue.push(i)
        self.assertEqual(view.top(), 0)  # attaches the frames pushed so far
        for i in range(3, 6):
            queue.push(i)  # new frames the view has not attached yet
        queue.close()  # unlinks every frame

        view.pop(2)
        self.assertEqual(len(view), 0)
        view.close()
