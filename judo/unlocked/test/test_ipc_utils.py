# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import multiprocessing as mp
import unittest
from ctypes import c_bool, c_int
from random import uniform
from threading import Thread
from time import perf_counter, sleep

from judo.unlocked import Barrier, BroadcastEvent, Mutex


class TestIPCUtils(unittest.TestCase):
    """Test IPC utils."""

    def test_mutex_thread(self) -> None:
        mutex = Mutex("test_mutex")
        locked = True

        def test_fn() -> None:
            nonlocal locked
            with mutex:
                locked = False

        thread = Thread(target=test_fn)
        with mutex:
            thread.start()
            sleep(0.1)
            self.assertTrue(locked)
        thread.join()
        sleep(0.1)
        self.assertFalse(locked)

    def test_mutex_mp(self) -> None:
        mutex = Mutex("test_mutex")
        n_proc = 10
        delay = 0.1 / n_proc

        def run() -> None:
            with mutex:
                sleep(delay)

        procs = [mp.Process(target=run, name=f"proc_{i}") for i in range(n_proc)]

        tic = perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=1.0)
        toc = perf_counter() - tic

        self.assertGreater(toc, 0.1)
        self.assertLess(toc, 0.2)

    def test_barrier(self) -> None:
        n_iter = 1000
        n_proc = 10
        delay = 0.1 / n_iter
        barrier = Barrier("test_barrier", n_proc)

        array = mp.RawArray(c_int, n_proc)
        success = mp.RawArray(c_bool, [False] * n_proc)

        def run(proc_ind: int) -> None:
            scs = True
            for _ in range(n_iter):
                with barrier:
                    sleep(uniform(0, delay))
                    array[proc_ind] += 1
                sleep(uniform(0, delay))
                if not all([a == array[proc_ind] for a in array]):
                    scs = False
            success[proc_ind] = scs

        procs = [mp.Process(target=run, args=(i,), name=f"proc_{i}", daemon=True) for i in range(n_proc)]

        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=1.0)

        self.assertTrue(all(success))

    def test_broadcast_event(self) -> None:
        event = BroadcastEvent("test_event")

        n_iter = 1000
        n_cons = 20
        delay = 0.1 / n_iter

        prod_success = mp.RawValue(c_bool, False)
        cons_success = mp.RawArray(c_bool, [False] * n_cons)

        def producer() -> None:
            while not all(cons_success):
                sleep(delay)
                event.notify()
            prod_success.value = True

        def consumer(ind: int) -> None:
            for _ in range(n_iter):
                with event:
                    pass
            cons_success[ind] = True

        prod_proc = mp.Process(target=producer, name="prod")
        cons_proc = [mp.Process(target=consumer, args=(i,), name=f"cons_{i}") for i in range(n_cons)]

        for p in cons_proc:
            p.start()
        prod_proc.start()

        prod_proc.join()
        for p in cons_proc:
            p.join(timeout=1)

        self.assertTrue(prod_success.value)
        self.assertTrue(all(cons_success))


if __name__ == "__main__":
    unittest.main()
