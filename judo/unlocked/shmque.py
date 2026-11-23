# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Lock-free shared memory FIFO queue with zero-copy read support for inter-process communication.

Implements a lock-free, variable-size shared memory queue for high-performance single-machine
robotics pipelines. Multiple consumer processes read from a single producer's queue with
independent cursors, minimal synchronization overhead, and zero-copy access to large data.

=== ARCHITECTURE ===

**Frames**: Named shared memory blocks (queue_name_frame_0, frame_1, ...) that group messages
into fixed-size chunks to amortize allocation overhead at high throughput rates. Each frame uses
packed layout (data forward, offsets backward) and pickle protocol 5 with out-of-band buffers,
enabling zero-copy tensor/array access—consumers reconstruct objects directly into shared memory
without heap copies. Frames are reclaimed deterministically once all consumers advance past them,
keeping memory proportional to the slowest consumer and enabling consistent performance.

**Shared Index Array (INDEX_ARRAY_SIZE)**: A shared memory c_size_t array that enables lock-free
coordination between producer and consumers without mutexes. Holds one read-cursor slot per
consumer view plus four queue-global bounds (begin, end, frame_begin, frame_end) in the last
four slots; the consumer capacity (MAX_QUEUE_VIEWS) is a compile-time constant and can be
raised if needed.
Word-sized atomic stores to x86_64 shared memory make updates immediately visible to all
processes—each consumer sees consistent queue bounds, while independent view cursors allow
asynchronous, multi-rate reading without blocking or expensive synchronization.

=== KEY DESIGN ===

1. **Independent Consumer Cursors**: Each view (consumer) maintains its own begin index, advancing
   independently at different rates (e.g., perception 10 Hz, control 50 Hz).

2. **Variable-Size Frames**: Grow dynamically (64 KB minimum, self-tuning) to amortize allocation
   overhead. Reclaimed when all consumers advance past them.

3. **Memory Reclamation**: Producer keeps data alive until min(all_view.begin) passes it. Explicit
   flush() on each push() ensures deterministic reclamation (no GC pauses).

4. **Lockless Synchronization**: Indices are plain word-sized atomic stores. A Condition variable
   (which still carries an internal Lock, slated for removal) notifies consumers on each push().

=== MAIN CLASSES ===

**Memory**: Serializes/deserializes objects using pickle protocol 5 with out-of-band buffers.
encode() copies to shared memory; decode() reconstructs views into shared memory (no copies).

**Frame**: Fixed-size shared memory block holding multiple serialized items with offset table.
Supports O(1) random access via bisect on cumulative counts.

**SharedMemoryQueue**: Producer-owned bookkeeping managing frame lifecycle, item serialization,
and multi-consumer coordination via shared INDEX_ARRAY_SIZE.

**SharedMemoryQueueView**: Consumer-side read cursor with independent begin index. Lazily syncs
frames, uses bisect to locate items, wraps results in const() for immutability.
"""

import logging
import multiprocessing as mp
import multiprocessing.synchronize
import pickle
import struct
import sys
import time
from bisect import bisect
from ctypes import Array, c_size_t, sizeof
from itertools import accumulate, pairwise
from multiprocessing.shared_memory import SharedMemory
from threading import Event
from typing import Any, Generic, TypeVar

from judo.unlocked.const import Const, const
from judo.unlocked.shm_session import queue_frame_name, register_segment

SIZEOF_SIZE_T = sizeof(c_size_t)
SIZE_T_MAX = c_size_t(-1).value
MIN_FRAME_SIZE = 64 * 1024  # 64 kb
MAX_QUEUE_VIEWS = 1000
INDEX_ARRAY_SIZE = c_size_t * (MAX_QUEUE_VIEWS + 4)  # ctype*N: a new class representing an array of N elements of ctype
_QUEUE_BEGIN_IND = -4
_QUEUE_END_IND = -3
_QUEUE_FRAME_BEGIN_IND = -2
_QUEUE_FRAME_END_IND = -1
T = TypeVar("T")


# Due to a bug in Python 3.10 (https://bugs.python.org/issue39959)
# we need to patch resource tracker for shared memory resource tracking
# TODO (dmitry): remove this hack when we switch to Python 3.13 or later versions
def _patch(name: str, rtype: str) -> None:
    """NoOp to patch resource tracker."""


mp.resource_tracker.register = _patch  # type: ignore[attr-defined]
mp.resource_tracker.unregister = _patch  # type: ignore[attr-defined]


# inspired by https://github.com/joblib/joblib/issues/1094
class Memory(Generic[T]):
    """Memory class for serialization and deserialization of python objects into and from a memory buffer."""

    def __init__(self, obj: Any):
        """Serialization: Prepare a serialized shared-memory representation of a Python object.

        This class uses pickle protocol 5 (PEP 574) with out-of-band buffers: large buffer-backed
        objects (e.g. NumPy arrays and torch tensors) store their raw bytes outside the main
        pickle stream. The pickle stream keeps only buffer-index references, and ``buffer_callback``
        lets us collect those raw buffers in ``self._buffers``.

        self._buffers layout:
        +-------------------+----------------------+----------------------+-----+-------------+
        | preamble          | out-of-band buffer 0 | out-of-band buffer 1 | ... | pickle dump |
        +-------------------+----------------------+----------------------+-----+-------------+

        preamble layout:
        +-----------------------+-----------------+-----------------+-----+-----------------+
        | num items in _buffers | len of buffer 0 | len of buffer 1 | ... | len of buffer n |
        +-----------------------+-----------------+-----------------+-----+-----------------+

        """
        # Concrete example used throughout the steps below:
        #   obj = {
        #       "words": ["cat", "dog"],
        #       "scores": [0.1, 0.9],
        #       "image1": np.array([1, 2, 3], dtype=np.float32),  # out-of-band buffer
        #       "image2": np.array([1, 2, 3], dtype=np.float32),  # out-of-band buffer
        #   }
        #
        # Step 1: pickle.dumps(protocol=5) writes the main pickle stream and, for each out-of-band
        # buffer-backed object (image1, image2), invokes _on_buffer(), which appends a memoryview
        # of the raw bytes to self._buffers.
        #   self._buffers -> [memoryview_of_image1, memoryview_of_image2]
        #   dump          -> pickle bytes for {dict/list structure + words + scores
        #                    + ref_to_image1 + ref_to_image2}
        self._buffers: list[memoryview] = []
        dump = pickle.dumps(obj, protocol=5, buffer_callback=self._on_buffer)

        # Step 2: append the pickle stream itself as the last buffer.
        #   self._buffers -> [memoryview_of_image1, memoryview_of_image2, memoryview(dump)]
        self._buffers.append(memoryview(dump))

        # Step 3: build the preamble = [num_buffers][size_0][size_1]...[size_n], each a native
        # size_t ("N"). decode() uses it to find each segment's offset and length.
        self._preamble = b"".join(
            [struct.pack("N", len(self._buffers))] + [struct.pack("N", len(buffer)) for buffer in self._buffers]
        )

        # Step 4: prepend memoryview(preamble). Final on-wire order written by encode():
        #   [memoryview(preamble), memoryview_of_image1, memoryview_of_image2, memoryview(dump)]
        self._buffers = [memoryview(self._preamble), *self._buffers]

        # Step 5: total byte size encode() will copy into the destination memory
        # (len(memoryview) is the size of the underlying data in bytes).
        # Bytes layout in destination memory:
        #   [bytes(preamble)][bytes(image1)][bytes(image2)][bytes(dump)]
        self._size = sum(map(len, self._buffers))

    def _on_buffer(self, buffer: pickle.PickleBuffer) -> None:
        """Callback invoked by pickle.dumps() for each out-of-band buffer (numpy array, torch tensor, bytearray)."""
        self._buffers.append(buffer.raw())

    def __len__(self) -> int:
        return self._size

    def encode(self, memory: bytearray | memoryview) -> None:
        """Copy serialized data from self._buffers into a memory block (e.g., a Frame's shared-memory buffer)."""
        if not isinstance(memory, memoryview):
            memory = memoryview(memory).cast("B")
        assert self._size <= len(
            memory
        ), f"Out-of-range error: encoding object of size {self._size} into a memory blocke of size {len(memory)}."
        offset = 0
        for buffer in self._buffers:
            size = len(buffer)
            # buffer is a memoryview; this copies its data into the target memory.
            memory[offset : offset + size] = buffer
            offset += size
        assert offset == self._size

    @classmethod
    def decode(cls, memory: bytearray | memoryview) -> Any:
        """Deserialize from a buffer into an object view (object data may be shared between multiple views)."""
        if not isinstance(memory, memoryview):
            memory = memoryview(memory)
        # First size_t in preamble: number of serialized buffers.
        num_buffers = struct.unpack("N", memory[:SIZEOF_SIZE_T])[0]
        # Next num_buffers size_t values: byte length of each buffer segment.
        buffers_size: list[int] = [
            struct.unpack("N", memory[(offset + 1) * SIZEOF_SIZE_T : (offset + 2) * SIZEOF_SIZE_T])[0]
            for offset in range(num_buffers)
        ]
        # Data starts after (num_buffers + 1) size_t entries in the preamble.
        buffers_offset = accumulate(buffers_size, initial=(num_buffers + 1) * SIZEOF_SIZE_T)

        # Build zero-copy views into each serialized segment.
        buffers: list[memoryview] = [memory[begin:end] for begin, end in pairwise(buffers_offset)]

        return pickle.loads(buffers[-1], buffers=buffers)


class Frame(Generic[T]):
    """A frame is a fixed size shared memory block that is used by variable size shared memory queue.

    Frame layout
    +--------+--------+-------+--------+------------------------+----------+-------+----------+----------+---------+
    | data_1 | data_2 |  ...  | data_n | //////// free //////// | offset_n |  ...  | offset_3 | offset_2 |    n    |
    +--------+--------+-------+--------+------------------------+----------+-------+----------+----------+---------+
    ^        ^        ^       ^        ^                        ^          ^       ^          ^          ^         ^
    0        |    offset_3    |     free_offset                 |    size - 4(n-1) |      size - 8       |      size
         offset_2          offset_n                        size - 4n           size - 12             size - 4

    WARNING: Note that offset_1 for the first data item data_1 is always 0, hence not stored in the frame.
    Its slot in the offset table -- the last number of the frame buffer, is used to store the number of items n instead,
    followed by the offsets of the remaining items, starting at offset_2.
    """

    def __init__(self, *, name: str, create: bool, size: int | None = None):
        logging.debug("Opening frame%s %s", "" if create else " view", name)
        assert not create or (size is not None), "Cannot specify size for shared memory that has already been created."
        self.name = name
        self._shm: SharedMemory
        self._time = time.perf_counter()
        shm_size = (size + SIZEOF_SIZE_T) if create and size is not None else 0
        self._shm = SharedMemory(name=name, create=create, size=shm_size)
        assert self._shm.buf is not None
        self._buffer = self._shm.buf
        self._is_view = not create

        self._len = 0
        self._size = len(self._buffer)
        self._free = self._size - SIZEOF_SIZE_T
        self._used = SIZEOF_SIZE_T
        self._free_offset = 0

        if self._is_view:
            # Views (read-only consumers) do not modify frames. Reassign production-only methods
            # to __deleted__, defined below, which raises RuntimeError if accidentally called, preventing silent bugs.
            self.free = self.__deleted__  # type: ignore[assignment, method-assign]
            self.used = self.__deleted__  # type: ignore[assignment, method-assign]
            self.push = self.__deleted__  # type: ignore[assignment, method-assign]
            self.push_memory = self.__deleted__  # type: ignore[assignment, method-assign]

    def __deleted__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Not implemented for views")

    def __len__(self) -> int:
        return struct.unpack("N", self._buffer[-SIZEOF_SIZE_T:])[0]

    def __del__(self) -> None:
        logging.debug("Closing frame%s %s", " view" if self._is_view else "", self.name)
        try:
            self._shm.close()
            if not self._is_view:
                logging.debug("Unlinking frame%s %s", " view" if self._is_view else "", self.name)
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    # Another process (or interpreter shutdown order) already
                    # unlinked this segment. Harmless — the kernel ref count
                    # is what actually frees the memory.
                    pass
        except BufferError:
            # A memoryview / numpy slice into the mmap is still alive (e.g. a
            # policy returned a view that hasn't been dropped yet at shutdown).
            # Mark the underlying mmap as closed so SharedMemory.__del__ does
            # not retry and print "Exception ignored in: ..." noise. The OS
            # frees the segment when the last reference is dropped.
            logging.debug(
                "frame%s %s still has live views; deferring close", " view" if self._is_view else "", self.name
            )
            try:
                self._shm._mmap = None  # type: ignore[attr-defined]  # noqa: SLF001
            except AttributeError:
                pass

    def free(self) -> int:
        """Free memory size in bytes."""
        return self._free

    def fits_size(self, size: int) -> bool:
        """Check if an object of a given size fits into the frame."""
        return size + SIZEOF_SIZE_T <= self._free

    def used(self) -> int:
        """Used memory size in bytes."""
        return self._used

    def size(self) -> int:
        """Frame total memory size in bytes."""
        return self._size

    def data_offset(self, index: int) -> int:
        """Memory offset in bytes where an [index] object is stored."""
        if index < 0 or index > len(self):
            raise IndexError(f"Out-of-range error: attempt to access index {index} within the range [0, {len(self)})")
        if index == 0:
            # offset of the first item is always 0: its data starts a  t the beginning of the buffer.
            # hence the offset is not stored in the frame,
            return 0
        if index == len(self):
            return self._free_offset
        return struct.unpack("N", self._buffer[-(index + 1) * SIZEOF_SIZE_T : -index * SIZEOF_SIZE_T])[0]

    def time(self) -> float:
        """Time at which frame has been created."""
        return self._time

    def __getitem__(self, index: int) -> T:
        return Memory.decode(self._buffer[self.data_offset(index) :].toreadonly())

    def push(self, el: T) -> None:
        """Push elelement into the frame."""
        # Serialize the data into a Memory object, which prepares the pickle buffers and preamble.
        el_mem: Memory = Memory(el)
        # Copy the serialized data into the frame's shared memory buffer, updating internal offset bookkeeping.
        self.push_memory(el_mem)

    def push_memory(self, mem: Memory) -> None:
        """Push a pre-serialized Memory object into the frame.

        push(el) = Memory(el) + push_memory(mem). Under pickle protocol 5, Memory(el) is cheap:
        pickle.dumps() doesn't copy buffer-backed payloads, it just exposes them via out-of-band
        buffer pointers. The actual copy happens in mem.encode() via memcpy (a system call, not
        the Python interpreter), which is still much cheaper than pre-protocol-5 pickle.
        The pre-built Memory object is reused by the caller to measure the serialized size
        (for frame allocation) and then copy the data.
        """
        assert not ((self._len == 0) ^ (self._free_offset == 0)), f"What?!?!? {self._len} {self._free_offset}"
        record_size = len(mem) + (0 if self._len == 0 else SIZEOF_SIZE_T)
        assert (
            record_size <= self._free
        ), f"Out-of-range error: Record size {record_size} is greater than remaining buffer size {self._free}"

        # Copy mem's serialized data into the frame's shared memory buffer at the current free offset.
        mem.encode(self._buffer[self._free_offset :])
        if self._len > 0:
            # Store the offset of the new item in the offset table, growing backward from the end of the frame buffer.
            # The first item (when self._len==0) has no offset stored since it always 0.
            self._buffer[-(self._len + 1) * SIZEOF_SIZE_T : -self._len * SIZEOF_SIZE_T] = struct.pack(
                "N", self._free_offset
            )
        self._len += 1
        self._free_offset += len(mem)

        # Store the number of items n in the last slot of the frame buffer.
        self._buffer[-SIZEOF_SIZE_T:] = struct.pack("N", self._len)
        self._free -= record_size
        self._used += record_size


class SharedMemoryQueueView(Generic[T]):
    """Consumer-side read cursor over a SharedMemoryQueue.

    Each view represents one independent consumer. It does not own queue data and cannot
    create or destroy frames in shared memory. Instead, it:

    - tracks its own logical read position as an item index via ``begin``,
    - mirrors producer-created frames locally as read-only handles in ``_frames``,
    - stores cumulative item bounds in ``_frame_bounds`` to map a global item
      index to ``(frame, index_in_frame)``,
    - tracks the global frame index of ``_frames[0]`` in ``_frame_begin``.

    Core semantics:

        - ``begin`` is a view-dependent item index stored in that view's slot in ``indices_shm``.
        - ``end`` is a queue-global item index past the last available item (exclusive).
        - ``frame_begin`` is the global frame index of the first frame of this view (``_frame_begin``).
        - ``frame_end`` is queue-global and equals ``queue_frame_end`` (past last frame, exclusive).
        - ``top()`` and ``__getitem__()`` return immutable proxies via ``const(...)`` to prevent
            accidental mutation of shared-memory-backed data.
        - Closing a view only releases this process's frame handles and marks its cursor as inactive;
            frame unlink/reclamation remains producer-owned.
    """

    def __init__(self, ind: int, name: str, *, has_data: mp.synchronize.Condition):
        self._ind = ind
        self._name = name
        self._view_name = f"{name}_view_{ind}"
        self._has_data = has_data
        self._indices_shm: SharedMemory = SharedMemory(name=self._name)
        assert self._indices_shm.buf is not None
        self._indices: Array[c_size_t] = INDEX_ARRAY_SIZE.from_buffer(self._indices_shm.buf)
        self.begin = 0
        del self._indices
        self._indices_shm.close()

        # View-owned frame handles and cumulative item counts.
        self._frames: list[Frame] = []
        self._frame_bounds = [0]

        # Global absolute frame index that self._frames[0] corresponds to.
        self._frame_begin = 0

        self._data_available = Event()

        # Same-process shutdown flag for the timeout-free waits below. During
        # teardown there is no producer to notify a parked view, so spin() calls
        # wake(), which sets this and notifies _has_data. Plain attribute (not
        # shared memory): the waker (spin) and parked worker threads share the
        # process. Written/read only under _has_data, so a wake() racing a
        # thread entering the wait cannot be lost.
        self._wake = False

    @property
    def queue_name(self) -> str:
        """Get associated queue name."""
        return self._name

    @property
    def name(self) -> str:
        """Get the name of this view."""
        return self._view_name

    def open(self) -> None:
        """Open a view.

        A view should be opened before reading the data.
        """
        logging.debug("[%s] Opening", self._view_name)
        self._indices_shm = SharedMemory(name=self._name)  # The same SharedMemory in SharedMemoryQueue.
        assert self._indices_shm.buf is not None
        self._indices = INDEX_ARRAY_SIZE.from_buffer(self._indices_shm.buf)
        while self.queue_begin == SIZE_T_MAX:
            time.sleep(0.001)
        self.begin = self.queue_begin

    def close(self) -> None:
        """Close a view.

        A view must be closed in order to clear all resources.
        """
        logging.debug("[%s] Closing %d frames", self._view_name, len(self._frames))
        del self._frames[:]

        logging.debug("[%s] Setting begin index to max value", self._view_name)
        self.begin = SIZE_T_MAX
        del self._indices
        self._indices_shm.close()
        logging.debug("[%s] Closed", self._view_name)

    def _sync_frames(self) -> None:
        if self.frame_end == SIZE_T_MAX:
            # The producer closed the queue (shutdown) and unlinked its frames; there is nothing left to attach.
            return
        # open frames
        view_frame_end = self.frame_begin + len(self._frames)

        if view_frame_end == self.frame_end:
            self._frame_bounds[-1] = self.queue_end
            return

        logging.debug(
            "[%s] Adding %d to %d frames", self._view_name, self.frame_end - view_frame_end, len(self._frames)
        )
        self._frames.extend(
            [
                Frame(name=queue_frame_name(self._name, frame_count), create=False)  # only attach to existing frame.
                for frame_count in range(view_frame_end, self.frame_end)
            ]
        )
        self._frame_bounds = list(
            accumulate(
                (len(frame) for frame in self._frames),
                initial=self._frame_bounds[0],
            )
        )

    def _not_empty(self) -> bool:
        return hasattr(self, "_indices") and self.end != SIZE_T_MAX and len(self) > 0

    def wait(self) -> bool:
        """Block until this view has unread data, or wake() releases it.

        Untimed on purpose: on macOS a timed Condition.wait() has no
        sem_timedwait, so CPython polls with a 1..20 ms backoff and only sees a
        notify when the current sleep chunk ends, adding ~5 ms per cross-process
        hop (measured); an untimed wait is an instant kernel wake (~0.1 ms).

        The under-lock re-check is mandatory: push() bumps end then
        notify_all()s under this same lock, so a notify firing between the
        lockless fast-path check and the park would be lost without re-checking
        the predicate after acquiring the lock -- the waiter would hang forever.
        _wake lets shutdown release the park through the same predicate.
        """
        if self._not_empty():
            return True
        with self._has_data:
            while not self._not_empty() and not self._wake:
                self._has_data.wait()
        return self._not_empty()

    def wait_on_queue_publish(self) -> bool:
        """Block until the next publish, or False when wake() fires on shutdown.

        Untimed (see wait). OneOf trigger threads loop on the result so they
        exit on shutdown instead of parking forever.
        """
        with self._has_data:
            if self._wake:
                return False
            self._has_data.wait()
            return not self._wake

    def wake(self) -> None:
        """Release threads parked in wait/wait_on_queue_publish on shutdown.

        Sets _wake and notifies, both under _has_data so it cannot race a thread
        entering the wait. Same-process scheduler teardown path only.
        """
        with self._has_data:
            self._wake = True
            self._has_data.notify_all()

    @property
    def begin(self) -> int:
        """The first available data index."""
        return self._indices[self._ind]

    @begin.setter
    def begin(self, begin: int) -> None:
        """The first available data index."""
        self._indices[self._ind] = begin

    @property
    def end(self) -> int:
        """Past last available data index."""
        return self.queue_end

    @property
    def frame_begin(self) -> int:
        """The first available frame index."""
        return self._frame_begin

    @property
    def frame_end(self) -> int:
        """Past last available frame index."""
        return self.queue_frame_end

    def __len__(self) -> int:
        # During shutdown the producing process may have torn down the
        # shared-memory indices, leaving ``end < begin`` (giant unsigned
        # garbage) or making the underlying mmap raise BufferError. Return 0
        # in that window so policies fall through their ``len < 1`` branches.
        # Note: ``__len__`` must return a value that fits in Py_ssize_t — if
        # we return e.g. 2**64-1, Python itself raises OverflowError at the
        # ``len(view)`` call site. Clamp aggressively.
        try:
            n = self.end - self.begin
        except Exception:
            return 0
        if not isinstance(n, int) or n <= 0 or n > sys.maxsize:
            return 0
        return n

    @property
    def queue_begin(self) -> int:
        """The first available data index."""
        return self._indices[_QUEUE_BEGIN_IND]

    @property
    def queue_end(self) -> int:
        """Past last available data index."""
        return self._indices[_QUEUE_END_IND]

    @property
    def queue_frame_begin(self) -> int:
        """The first available frame index."""
        return self._indices[_QUEUE_FRAME_BEGIN_IND]

    @property
    def queue_frame_end(self) -> int:
        """Past last available frame index."""
        return self._indices[_QUEUE_FRAME_END_IND]

    def __getitem__(self, index: int) -> T | Const[T]:
        if index < self.begin or index >= self.end:
            raise IndexError(f"Index {index} is out of bounds [{self.begin}, {self.end})")
        self._sync_frames()
        local_frame_idx = bisect(self._frame_bounds, index) - 1  # local frame index in _frames!
        index_in_frame = index - self._frame_bounds[local_frame_idx]  # item index within the frame.
        assert index_in_frame >= 0, f"index in frame is negative {index_in_frame}"
        frame = self._frames[local_frame_idx]
        assert index_in_frame < len(frame), f"index in frame {index_in_frame} is out of bounds {len(frame)}"
        return const(frame[index_in_frame])

    def top(self) -> T | Const[T]:
        """The top element (FIFO) order."""
        if len(self) < 1:
            raise IndexError("The queue is empty")
        self._sync_frames()
        index = self.begin
        # Usually the top element is in the first frame, but we still search the frame index here for robustness.
        local_frame_idx = 1
        while index >= self._frame_bounds[local_frame_idx]:
            local_frame_idx += 1
        index_in_frame = index - self._frame_bounds[local_frame_idx - 1]
        assert index_in_frame >= 0, f"index in frame is negative {index_in_frame}"
        frame = self._frames[local_frame_idx - 1]
        assert index_in_frame < len(frame), f"index in frame {index_in_frame} is out of bounds {len(frame)}"
        return const(frame[index_in_frame])

    def pop(self, num_items: int = 1) -> None:
        """Remove the top `num_items` elements."""
        if self.queue_end == SIZE_T_MAX:
            # The producer closed the queue while this consumer was mid-step (shutdown); nothing left to advance.
            return
        self._sync_frames()
        view_length = len(self)
        if num_items > view_length:
            raise IndexError(f"Attempting to remove {num_items} elements from the queue of length {view_length}")
        self.begin += num_items
        logging.debug(
            "[%s] Advanced internal counter to %d out of %d",
            self._view_name,
            self.begin,
            self.end,
        )

        # close frames
        if self.begin < self._frame_bounds[1]:
            return

        num_frames_to_drop = min(
            bisect(self._frame_bounds, self.begin) - 1,
            len(self._frames) - 1,
        )
        if num_frames_to_drop > 0:
            logging.debug(
                "[%s] Removing %d out of %d frames",
                self._view_name,
                num_frames_to_drop,
                len(self._frames),
            )
            del self._frames[:num_frames_to_drop]
            del self._frame_bounds[:num_frames_to_drop]
            self._frame_begin += num_frames_to_drop


class SharedMemoryQueue(Generic[T]):
    """Producer-owned bookkeeping for a frame-backed FIFO queue in shared memory.

    This class is mostly index bookkeeping, not the payload storage itself.
    Queue messages are serialized into shared-memory Frame blocks, and each frame stores
    multiple messages. The queue is therefore an ordered collection of frames plus
    indices describing the globally visible message and frame ranges.

    The producer appends messages to frames, allocates new frames when needed, and
    pops old frames in FIFO order once all consumers have advanced past them.

    Global indices are stored in a shared index array (indices_shm / INDEX_ARRAY_SIZE) so all
    SharedMemoryQueueView consumers see the same queue state:

    - per-view read cursors: one slot per view, indexed by view index,
    - queue item bounds: queue.begin and queue.end,
    - queue frame bounds: queue.frame_begin and queue.frame_end.

    Views share this index segment for coordination, while frame creation and
    frame unlink/reclamation remain producer-owned responsibilities.
    """

    def __init__(self, name: str, *, min_rate: int = MIN_FRAME_SIZE, fps: int = 1):
        self._name = name
        self._min_rate = min_rate  #  i.e., 1 MB/s the default value
        self._fps = fps  # expected new frames per second (default is 1)
        self._rate_estimate = self._min_rate  # B/s

        # Frames in queue; each is a shared memory block holding multiple serialized items.
        self._frames: list[Frame] = []
        # Cumulative counts of items per frame; used with bisect to locate which frame holds a given item.
        self._frame_bounds = [0]

        self._has_data = mp.Condition()

        self._view_count = 0

        # The shared index array is the sole coordination point between producer and all consumers.
        # Layout: self._indices[0 : _QUEUE_BEGIN_IND] holds one read-cursor slot per consumer view;
        # the last four slots (_QUEUE_BEGIN_IND.._QUEUE_FRAME_END_IND) hold queue-global bounds
        # (queue.begin/end, frame.begin/end). This avoids any other IPC mechanism (pipes, sockets,
        # locks): each consumer reads queue.end to know how many messages are available, and the
        # producer computes min(self._indices[:_QUEUE_BEGIN_IND]) to find the slowest consumer and
        # determine which frames are safe to reclaim.
        register_segment(self._name, self._name)
        self._indices_shm: SharedMemory = SharedMemory(name=self._name, create=True, size=sizeof(INDEX_ARRAY_SIZE))

        # Map the ctypes array directly onto _indices_shm.buf (not a copy) so any write is
        # immediately visible to every process that has attached to the same segment. On x86_64,
        # aligned word-sized stores are atomic, so individual slot updates need no mutex.
        assert self._indices_shm.buf is not None
        indices = INDEX_ARRAY_SIZE.from_buffer(self._indices_shm.buf)

        # self._indices_shm should remained closed until open() is called, at which point self._indices
        # will be assigned as the live view used for all subsequent reads and writes.
        # So `indices` is just a **temporary** view used only to initialize all slots to SIZE_T_MAX.
        # It must be deleted before closing the self._indices_shm to avoid BufferError.
        for i in range(len(indices)):
            indices[i] = SIZE_T_MAX
        del indices  # release the temporary view before closing the shm handle
        self._indices_shm.close()
        self._indices: Array[c_size_t]  # assigned in open()

    def __del__(self) -> None:
        try:
            self._indices_shm.unlink()
        except FileNotFoundError:
            # Already unlinked (e.g. by close() or external cleanup).
            pass
        except BufferError:
            logging.error("Failed to close queue %s", self.name)

    # def __getitem__(self, index: int) -> Const | T:
    #     if index < self.begin or index >= self.end:
    #         raise IndexError(f"Index {index} is out of bounds [{self.begin}, {self.end})")
    #     i = bisect(self._frame_bounds, index) - 1
    #     index_in_frame = index - self._frame_bounds[i]
    #     assert index_in_frame >= 0, f"index in frame is negative {index_in_frame}"
    #     frame = self._frames[i]
    #     assert index_in_frame < len(frame), f"index in frame {index_in_frame} is out of bounds {len(frame)}"
    #     return const(frame[index_in_frame])

    @property
    def name(self) -> str:
        """Get them name of this queue."""
        return self._name

    def __len__(self) -> int:
        try:
            n = self.end - self.begin
        except Exception:
            return 0
        if not isinstance(n, int) or n <= 0 or n > sys.maxsize:
            return 0
        return n

    @property
    def begin(self) -> int:
        """The first available data index."""
        return self._indices[_QUEUE_BEGIN_IND]

    @begin.setter
    def begin(self, begin: int) -> None:
        """The first available data index."""
        self._indices[_QUEUE_BEGIN_IND] = begin

    @property
    def end(self) -> int:
        """Past last available data index."""
        return self._indices[_QUEUE_END_IND]

    @end.setter
    def end(self, end: int) -> None:
        """Past last available data index."""
        self._indices[_QUEUE_END_IND] = end

    @property
    def frame_begin(self) -> int:
        """The first available frame index."""
        return self._indices[_QUEUE_FRAME_BEGIN_IND]

    @frame_begin.setter
    def frame_begin(self, begin: int) -> None:
        """The first available frame index."""
        self._indices[_QUEUE_FRAME_BEGIN_IND] = begin

    @property
    def frame_end(self) -> int:
        """Past last available frame index."""
        return self._indices[_QUEUE_FRAME_END_IND]

    @frame_end.setter
    def frame_end(self, end: int) -> None:
        """Past last available frame index."""
        self._indices[_QUEUE_FRAME_END_IND] = end

    def open(self) -> None:
        """Open a queue.

        A queue should be opened before pushing any data.
        """
        logging.debug("[%s] Opening", self._name)
        self._indices_shm = SharedMemory(name=self._name)
        assert self._indices_shm.buf is not None
        self._indices = INDEX_ARRAY_SIZE.from_buffer(self._indices_shm.buf)
        assert self.begin == SIZE_T_MAX, f"{self._name} has corrupted begin value {self.begin}"
        self.begin = 0
        self.end = 0
        self.frame_begin = 0
        self.frame_end = 0

    def close(self) -> None:
        """Close a queue.

        A queue must be closed in order to clear all resources.
        """
        logging.debug("[%s] Closing with %d frames", self._name, len(self._frames))
        self.begin = SIZE_T_MAX
        self.end = SIZE_T_MAX
        self.frame_begin = SIZE_T_MAX
        self.frame_end = SIZE_T_MAX

        self._frame_bounds = [0]
        del self._frames[:]

        del self._indices
        self._indices_shm.close()
        logging.debug("[%s] Closed", self._name)

    def flush(self) -> None:
        """Manually flush unused frames."""
        if len(self._frames) == 0:
            return
        min_begin = min(self._indices[:_QUEUE_BEGIN_IND])
        min_begin = min(min_begin, self.end)
        assert min_begin >= self.begin
        self.begin = min_begin
        # we do not delete the last frame for preventing frequent allocations
        n = min(bisect(self._frame_bounds, self.begin) - 1, len(self._frames) - 1)
        if n == 0:
            return
        logging.debug("[%s] Flushing %d out of %d frames", self._name, n, len(self._frames))
        self.frame_begin += n
        del self._frames[:n]
        del self._frame_bounds[:n]

    def make_view(self) -> SharedMemoryQueueView:
        """Make a queue view."""
        view: SharedMemoryQueueView = SharedMemoryQueueView(
            ind=self._view_count,
            name=self._name,
            has_data=self._has_data,
        )
        self._view_count += 1
        return view

    def _add_frame(self, size: int = 0) -> None:
        self.flush()
        if self._frames:
            self._rate_estimate = int(
                sum((frame.used() for frame in self._frames)) / (time.perf_counter() - self._frames[0].time())
            )
        logging.debug("[%s] Current rate estimate %d B/s", self._name, self._rate_estimate)
        self._rate_estimate = max(self._fps * self._rate_estimate, self._fps * self._min_rate, size)
        name = queue_frame_name(self._name, self.frame_end)
        register_segment(self._name, name)
        frame: Frame = Frame(name=name, create=True, size=self._rate_estimate)
        self._frames.append(frame)
        self._frame_bounds.append(self._frame_bounds[-1])
        self.frame_end += 1

    def push(self, el: T) -> None:
        """Push an element into a queue."""
        el_mem: Memory = Memory(el)
        el_size = len(el_mem)
        if not self.is_open():
            # TODO(duy): raise an exception instead of silently returning?
            return
        if not self._frames or not self._frames[-1].fits_size(el_size):
            self._add_frame(el_size)
        self._frames[-1].push(el)
        self._frame_bounds[-1] += 1
        self.end += 1
        with self._has_data:
            # Wake all consumers waiting on this condition so they can re-check queue state and read new data.
            self._has_data.notify_all()

    def is_open(self) -> bool:
        """Check if the queue is open."""
        return self.begin != SIZE_T_MAX
