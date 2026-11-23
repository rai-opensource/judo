# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from ctypes import c_uint64
from multiprocessing import Semaphore
from multiprocessing.sharedctypes import RawValue
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# We are preparing for using POSIX Semaphores to be able to synch with C++/Rust/Zig code.
# (POSIX Semaphores)[https://github.com/osvenskan/posix_ipc/blob/develop/USAGE.md#the-semaphore-class]
# We also will create custom Test Semaphores that will simulate random arrival times.
class NamedSemaphore:
    """Posix semaphore."""

    def __init__(self, name: str, value: int = 1):
        self.__name__ = name
        self._semaphore = Semaphore(value)

    def acquire(self) -> None:
        """Acquire semaphore."""
        self._semaphore.acquire()

    def release(self, n: int = 1) -> None:
        """Release semaphore."""
        if n == 1:  # it's a bit faster to do if then a for loop of 1
            self._semaphore.release()
        else:
            for _ in range(n):
                self._semaphore.release()


class UseGenericSemaphore(Generic[T]):
    """Templated semaphore."""

    __SemaphoreT__: type = NamedSemaphore

    def __class_getitem__(cls, SemaphoreT: type) -> type:
        UseSemaphore = cls
        UseSemaphore.__SemaphoreT__ = SemaphoreT
        return UseSemaphore


class Mutex(UseGenericSemaphore):
    """Mutex."""

    def __init__(self, name: str):
        self._name = name
        self._s = self.__SemaphoreT__(name + "_s", 1)

    def __enter__(self) -> None:
        self._s.acquire()

    def __exit__(self, *args: Any) -> None:
        self._s.release()

    def acquire(self) -> None:
        """Manually acquire mutex."""
        self._s.acquire()

    def release(self) -> None:
        """Manually release mutex."""
        self._s.release()


class Event(UseGenericSemaphore):
    """Event syncronizes a calling process with one and only one of waiting processes."""

    def __init__(self, name: str):
        self._name = name
        self._s1 = self.__SemaphoreT__(name + "_s1", 0)
        self._s2 = self.__SemaphoreT__(name + "_s2", 0)

    def __enter__(self) -> None:
        self._s1.acquire()

    def __exit__(self, *args: Any) -> None:
        self._s2.release()

    def notify(self) -> None:
        """Notify one waiting process and wait for it to wake up."""
        self._s1.release()
        self._s2.acquire()


class BroadcastEvent(UseGenericSemaphore):
    """Broadcast event syncronizes a calling process with all (any number) of waiting processes."""

    def __init__(self, name: str):
        self._name = name
        self._counter = RawValue(c_uint64, 0)
        self._mutex = Mutex[self.__SemaphoreT__](name + "_mutex")  # type: ignore[misc]
        self._t = Mutex[self.__SemaphoreT__](name + "_turn")  # type: ignore[misc]

        self._s1 = self.__SemaphoreT__(name + "_s1", 0)
        self._s2 = self.__SemaphoreT__(name + "_s2", 1)
        self._s3 = self.__SemaphoreT__(name + "_s3", 0)

    def __enter__(self) -> None:
        # Block until notifier is done
        with self._t:
            # Count number of listeners
            with self._mutex:
                self._counter.value += 1

        # Wait for the notice
        self._s1.acquire()

    def __exit__(self, *args: Any) -> None:
        # Release listeners
        with self._mutex:
            self._counter.value -= 1
            if self._counter.value == 0:
                self._s2.release()
                self._s3.release()

        self._s2.acquire()
        self._s2.release()

    def notify(self) -> None:
        """Notify all waiting process and wait for all of them to wake up."""
        # Notifier block
        with self._t:
            with self._mutex:
                n = self._counter.value
            # if no listeners are waiting---leave
            if n == 0:
                return
            # notify
            self._s2.acquire()
            self._s1.release(n)
            self._s3.acquire()


class Barrier(UseGenericSemaphore):
    """Barrier syncronizes exactly n processes."""

    def __init__(self, name: str, n: int):
        self._name = name
        self._n = n
        self._counter = RawValue(c_uint64, 0)
        self._mutex = Mutex[self.__SemaphoreT__](name + "_mutex")  # type: ignore[misc]
        self._s1 = self.__SemaphoreT__(name + "_s1", 0)
        self._s2 = self.__SemaphoreT__(name + "_s2", 1)

    def __enter__(self) -> None:
        with self._mutex:
            self._counter.value += 1
            if self._counter.value == self._n:
                self._s2.acquire()
                self._s1.release(self._n + 1)

        self._s1.acquire()

    def __exit__(self, *args: Any) -> None:
        with self._mutex:
            self._counter.value -= 1
            if self._counter.value == 0:
                self._s1.acquire()
                self._s2.release(self._n + 1)

        self._s2.acquire()

    # A slower, but perhaps a safer solution, in which threads are unlocked one-by-one.
    # def __enter__(self) -> None:
    #     with self._mutex:
    #         self._counter.value += 1
    #         if self._counter.value == self._n:
    #             self._s2.acquire()
    #             self._s1.release()
    #
    #     self._s1.acquire()
    #     self._s1.release()
    #
    # def __exit__(self, *args) -> None:
    #     with self._mutex:
    #         self._counter.value -= 1
    #         if self._counter.value == 0:
    #             self._s1.acquire()
    #             self._s2.release()
    #
    #     self._s2.acquire()
    #     self._s2.release()
