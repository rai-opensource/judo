# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from collections import deque
from multiprocessing import Event
from typing import Callable, Generator, Generic, TypeVar

from judo.unlocked.const import Const, const

T = TypeVar("T")


class Queue(Generic[T]):
    """Shared queue."""

    def __init__(self) -> None:
        self._read: deque[T] = deque()
        self._writable: tuple[deque[T], deque[T]] = (deque(), deque())
        self._write_index = 0
        self._offset = 0
        self._views: list[QueueView[T]] = []

    @property
    def _write(self) -> deque[T]:
        return self._writable[self._write_index]

    @property
    def _buffer(self) -> deque[T]:
        return self._writable[self._write_index ^ 1]

    # @property.setter
    # def _buffer(self, value: deque[T]) -> None:
    #     self._writable[self._write_index ^ 1] = value

    @property
    def begin(self) -> int:
        """The first available data index."""
        return self._offset

    @property
    def end(self) -> int:
        """Past last available data index."""
        return self._offset + len(self)

    def __getitem__(self, index: int) -> Const | T:
        index -= self._offset
        if index < 0:
            raise IndexError("Out of bounds")
        if index >= len(self._read):
            self._swap()
        return const(self._read[index])

    def _swap(self) -> None:
        self._write_index ^= 1
        self._read.extend(self._buffer)
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._read) + len(self._write)

    def flush(self) -> None:
        """Manually flush unused elements."""
        min_begin = min(self.end, *(view.begin for view in self._views))
        if min_begin == self._offset:
            return
        if min_begin - self._offset > len(self._read):
            self._swap()

        assert min_begin - self._offset <= len(self._read)
        [self._read.popleft() for _ in range(min_begin - self._offset)]
        self._offset = min_begin

    def push(self, value: T) -> None:
        """Push an element into a queue."""
        self._write.append(value)
        # With small number of subscribers (views) it is much faster to
        # i) notify subscriber views directly from publish thread, e.g.,
        for view in self._views:
            view.notify()
        # , than ii) spin a separate thread and notify views from there, e.g.,
        # Thread(target=lambda : [view.notify() for view in self._views]).begin()

    def pop(self) -> T:
        """Remove top element from the queue."""
        if not self._read:
            self._swap()
        result = self._read.popleft()
        self._offset += 1
        return result

    def top(self) -> Const | T:
        """Get top element from the queue."""
        if not self._read:
            self._swap()
        return const(self._read[0])

    def clear(self) -> None:
        """Clear the queue."""
        for view in self._views:
            view.clear()
        self._write_index ^= 1
        self._buffer.clear()
        self._read.clear()
        self._offset = 0

    def make_view(self) -> "QueueView":
        """Make a queue view."""
        queue_view = QueueView(self, begin=self.begin, end=self.end)
        self._views.append(queue_view)
        return queue_view


class QueueView(Generic[T]):
    """The view fot the shared queue."""

    def __init__(self, queue: Queue[T], begin: int = 0, end: int = 0, max_len: None | int = None):
        self._queue = queue
        self._begin = begin
        self._end = end
        self._max_len = max_len
        self._has_data = Event()
        self._on_ready: list[Callable] = []

    @property
    def begin(self) -> int:
        """The first available data index."""
        return self._begin

    @property
    def end(self) -> int:
        """Past last available data index."""
        return self._end

    def __getitem__(self, index: int) -> Const | T | tuple:
        if index < self._begin or index >= self._end:
            raise IndexError("Out of bounds")
        return self._queue[index]

    def notify(self) -> None:
        """Notify view that new data is available."""
        if not self.ready:
            [callback() for callback in self._on_ready]
        self._has_data.set()

    def wait(self, timeout: float | None = None) -> None:
        """Wait for new data."""
        self._has_data.wait(timeout)

    @property
    def ready(self) -> bool:
        """Check if data is available."""
        return len(self) > 0 or self._has_data.is_set()

    def pop(self) -> Const | T | tuple:
        """Remove top element from the view."""
        if self._begin == self._end:
            raise IndexError("Queue is empty")
        self._begin += 1
        return self._queue[self._begin - 1]

    def top(self) -> Const | T | tuple:
        """Get view top element."""
        if self._begin == self._end:
            raise IndexError("Queue is empty")
        return self._queue[self._begin]

    def last(self) -> Const | T | tuple:
        """Get view last element."""
        if self._begin == self._end:
            raise IndexError("Queue is empty")
        return self._queue[self._end - 1]

    def flush(self, *, flush_queue: bool = False) -> None:
        """Manually flush the view."""
        self._has_data.clear()
        self._begin = self._end
        self._end = self._queue.end
        if self._max_len is not None:
            self._begin = max(self._begin, self._end - self._max_len)
        if flush_queue:
            self._queue.flush()

    def __iter__(self) -> Generator:
        return (self._queue[i] for i in range(self.begin, self._end))

    def __len__(self) -> int:
        return self._end - self._begin

    def clear(self) -> None:
        """Clear the view."""
        self._has_data.clear()
        self._begin = self._queue.begin
        self._end = self._begin

    def register(self, on_ready: Callable) -> None:
        """Register the callback on available data."""
        self._on_ready.append(on_ready)
