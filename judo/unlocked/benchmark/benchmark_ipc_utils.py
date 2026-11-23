# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import multiprocessing as mp
import traceback
from ctypes import c_bool

from judo.unlocked.ipc import Barrier, BroadcastEvent, Event, Mutex
from judo.unlocked.time_stat import SharedTimeStat


def benchmark_mutex() -> None:
    mutex = Mutex("benchmark_mutex")
    lock_time = SharedTimeStat("mutex_lock_time", lock=False)
    total_time = SharedTimeStat("mutex_lock_time", lock=False)

    n_trials = 10_000

    for _ in range(n_trials):
        lock_time.tic()
        total_time.tic()
        with mutex:
            lock_time.toc()
        total_time.toc()

    print(f"            Mutex lock time: {lock_time}")
    print(f"      Mutex round-trip time: {total_time}")


def benchmark_event() -> None:
    event = Event("benchmark_event")
    notify_time = SharedTimeStat("event_notify_time", lock=False)
    ping_pong_time = SharedTimeStat("event_ping_pong_time", lock=False)

    n_trials = 10_000

    def ping() -> None:
        for _ in range(n_trials):
            ping_pong_time.tic()
            notify_time.tic()
            event.notify()
            ping_pong_time.toc()

    def pong() -> None:
        for _ in range(n_trials):
            with event:
                notify_time.toc()

    ping_proc = mp.Process(target=ping, name="ping")
    pong_proc = mp.Process(target=pong, name="pong")

    ping_proc.start()
    pong_proc.start()

    ping_proc.join(timeout=30)
    pong_proc.join(timeout=30)

    print(f"          Event notify time: {notify_time}")
    print(f"             Ping-pong time: {ping_pong_time}")


def benchmark_barrier() -> None:
    n_sub = 10
    barrier = Barrier("benchmark_barrier", n_sub + 1)
    notify_time = SharedTimeStat("barrier_notify_time", lock=False)
    ping_pong_time = SharedTimeStat("barrier_ping_pong_time", lock=False)

    n_trials = 10_000

    def first() -> None:
        for _ in range(n_trials):
            ping_pong_time.tic()
            notify_time.tic()
            with barrier:
                notify_time.toc()
            ping_pong_time.toc()

    def rest() -> None:
        for _ in range(n_trials):
            with barrier:
                pass

    procs = [mp.Process(target=first, name="first")] + [mp.Process(target=rest, name=f"rest_{i}") for i in range(n_sub)]

    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)

    print(f"        Barrier notify time: {notify_time}")
    print(f"             Ping-pong time: {ping_pong_time}")


def benchmark_broadcast() -> None:
    broadcast = BroadcastEvent("benchmark_broadcast")
    notify_time = SharedTimeStat("broadcast_notify_time", lock=False)
    ping_pong_time = SharedTimeStat("broadcast_ping_pong_time", lock=False)

    n_trials = 10_000
    n_sub = 10

    done = mp.RawArray(c_bool, [False] * n_sub)

    def first() -> None:
        try:
            while not all(done):
                ping_pong_time.tic()
                notify_time.tic()
                broadcast.notify()
                ping_pong_time.toc()
        except BaseException:
            pass

    def rest(ind: int) -> None:
        try:
            for _ in range(n_trials):
                with broadcast:
                    if ind == 0:
                        notify_time.toc()
        except BaseException:
            if ind == 0:
                traceback.print_exc()
        finally:
            done[ind] = True

    procs = [mp.Process(target=first, name="first")] + [
        mp.Process(target=rest, args=(i,), name=f"rest_{i}") for i in range(n_sub)
    ]
    for p in procs[1:]:
        p.start()
    procs[0].start()
    try:
        for p in procs:
            p.join(timeout=30)
    except BaseException:
        pass

    print(f"Broadcast Event notify time: {notify_time}")
    print(f"             Ping-pong time: {ping_pong_time}")


def main() -> None:
    benchmark_mutex()
    benchmark_event()
    benchmark_barrier()
    benchmark_broadcast()


if __name__ == "__main__":
    main()
