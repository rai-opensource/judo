# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import queue
import threading
from ctypes import c_double, c_uint64
from multiprocessing.sharedctypes import RawValue
from typing import Any

from judo.unlocked.shmque import SharedMemoryQueueView

# Sentinel pushed by a OneOf trigger thread to release a blocking OneOf.wait()
# get() on shutdown (there is no timeout to fall back on -- see
# SharedMemoryQueueView.wait).
_WAKE = object()


class Lossless:
    """Lossless policy"""

    def ready(self, view: SharedMemoryQueueView) -> bool:
        """Check if the input is ready."""
        return len(view) > 0

    def wait(self, view: SharedMemoryQueueView) -> bool:
        """Wait for the input."""
        return view.wait()

    def get(self, view: SharedMemoryQueueView) -> Any:
        """Get the next element."""
        return view.top()

    def next(self, view: SharedMemoryQueueView) -> None:
        """Advance the input."""
        view.pop()


class Latest:
    """Lossless policy"""

    def __init__(self) -> None:
        self._drop = 0
        self._n = RawValue(c_uint64, 0)
        self._min = RawValue(c_uint64, -1)
        self._max = RawValue(c_uint64, 0)
        self._mu = RawValue(c_double, 0.0)
        self._s2 = RawValue(c_double, 0.0)

    def ready(self, view: SharedMemoryQueueView) -> bool:
        """Check if the input is ready."""
        return len(view) > 0

    def wait(self, view: SharedMemoryQueueView) -> bool:
        """Wait for the input."""
        return view.wait()

    def get(self, view: SharedMemoryQueueView) -> Any:
        """Get the next eleent."""
        self._drop = len(view) - 1
        assert self._drop >= 0
        # accumulate dropped message statistics
        self._n.value += 1
        self._min.value = min(self._min.value, self._drop)
        self._max.value = max(self._max.value, self._drop)
        delta = self._drop - self._mu.value
        self._mu.value = self._mu.value + delta / self._n.value
        delta2 = self._drop - self._mu.value
        self._s2.value = self._s2.value + delta * delta2
        # return the latest message
        return view[view.begin + self._drop]

    def next(self, view: SharedMemoryQueueView) -> None:
        """Advance the input."""
        view.pop(self._drop + 1)


class Optional:
    """Lossless policy"""

    def __init__(self) -> None:
        self._drop = 0
        self._n = RawValue(c_uint64, 0)
        self._min = RawValue(c_uint64, -1)
        self._max = RawValue(c_uint64, 0)
        self._mu = RawValue(c_double, 0.0)
        self._s2 = RawValue(c_double, 0.0)

    def ready(self, view: SharedMemoryQueueView) -> bool:
        """Check if the input is ready."""
        return True

    def wait(self, view: SharedMemoryQueueView) -> bool:
        """Wait for the input."""
        return True

    def get(self, view: SharedMemoryQueueView) -> Any:
        """Get the next eleent."""
        self._drop = len(view) - 1
        if self._drop < 0:
            return None
        # accumulate dropped message statistics
        self._n.value += 1
        self._min.value = min(self._min.value, self._drop)
        self._max.value = max(self._max.value, self._drop)
        delta = self._drop - self._mu.value
        self._mu.value = self._mu.value + delta / self._n.value
        delta2 = self._drop - self._mu.value
        self._s2.value = self._s2.value + delta * delta2
        # return the latest message
        return view[view.begin + self._drop]

    def next(self, view: SharedMemoryQueueView) -> None:
        """Advance the input."""
        if self._drop < 0:
            return
        view.pop(self._drop + 1)


class KeepLatest:
    """Lossless policy"""

    def __init__(self) -> None:
        self._drop = 0
        self._n = RawValue(c_uint64, 0)
        self._min = RawValue(c_uint64, -1)
        self._max = RawValue(c_uint64, 0)
        self._mu = RawValue(c_double, 0.0)
        self._s2 = RawValue(c_double, 0.0)

    def ready(self, view: SharedMemoryQueueView) -> bool:
        """Check if the input is ready."""
        return len(view) > 0

    def wait(self, view: SharedMemoryQueueView) -> bool:
        """Wait for the input."""
        return len(view) > 0 or view.wait()

    def get(self, view: SharedMemoryQueueView) -> Any:
        """Get the next eleent."""
        self._drop = len(view) - 1
        assert self._drop >= 0
        # accumulate dropped message statistics
        self._n.value += 1
        self._min.value = min(self._min.value, self._drop)
        self._max.value = max(self._max.value, self._drop)
        delta = self._drop - self._mu.value
        self._mu.value = self._mu.value + delta / self._n.value
        delta2 = self._drop - self._mu.value
        self._s2.value = self._s2.value + delta * delta2
        # return the latest message
        return view[view.begin + self._drop]

    def next(self, view: SharedMemoryQueueView) -> None:
        """Advance the input."""
        if self._drop <= 0:
            return
        view.pop(self._drop)


class OneOf:
    """Trigger on one of the inputs.

    This policy implies optional for all inputs but one, that is,
    only one input will contain a value, the rest will be none.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue
        self._next: str | None = None
        self._threads: dict[str, threading.Thread] = {}

    def ready(self, view: SharedMemoryQueueView) -> bool:
        """Check if any input is ready."""
        if len(view) > 0:
            return True
        if not hasattr(self, "_queue"):
            return False
        return not self._queue.empty()

    def wait(self, view: SharedMemoryQueueView) -> bool:
        """Wait for any input."""
        if view.name not in self._threads:
            if not hasattr(self, "_queue"):
                self._queue = queue.Queue()
            self._threads[view.name] = threading.Thread(
                name=f"{view.name}_trigger", target=self._trigger_on_view, args=(view,), daemon=True
            )
            self._threads[view.name].start()
            return False

        if self._next is not None:
            return True

        # Blocking get -- no timeout (see SharedMemoryQueueView.wait). The
        # per-view trigger threads push a view name on publish, or the _WAKE
        # sentinel when released on shutdown, so this never parks forever.
        got = self._queue.get()
        if got is _WAKE:
            self._next = None
            return False
        self._next = got
        return True

    def get(self, view: SharedMemoryQueueView) -> Any:
        """Get the next element or None if view has not triggered yet."""
        if view.name != self._next:
            return None
        return view.top()

    def next(self, view: SharedMemoryQueueView) -> None:
        """Advance the input."""
        if view.name != self._next:
            return
        self._next = None
        view.pop()

    def _trigger_on_view(self, view: SharedMemoryQueueView) -> None:
        # Loop until shutdown: wait_on_queue_publish() returns True on a real
        # publish (forward the view name) and False when wake() releases it; in
        # that case push the _WAKE sentinel so a OneOf.wait() blocked on get()
        # unblocks and returns False, then exit the thread cleanly.
        while True:
            if view.wait_on_queue_publish():
                self._queue.put_nowait(view.name)
            else:
                self._queue.put_nowait(_WAKE)
                return
