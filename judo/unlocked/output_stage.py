# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import hashlib
import time
from typing import Any, Iterator

from judo.unlocked.annotation import is_tuple_t
from judo.unlocked.const import const_cast
from judo.unlocked.shm_session import session_queue_name
from judo.unlocked.shmque import SharedMemoryQueue


class OutputStage:
    """Producer side of a node's outputs: maps return values to `SharedMemoryQueue`s.

    Mapping from the function's return annotation:
    - A single return value -> one queue ``{name}_output_0``.
    - A tuple return -> queues ``{name}_output_0..{name}_output_{N-1}``.
    - ``None`` annotation -> no queues.
    """

    def __init__(self, name: str, return_annotation: Any):
        self._name = name
        self._hash = hashlib.sha1((name + f"_{time.time()}").encode("utf-8")).hexdigest()[:20]
        self._return_annotation: tuple[type, ...]
        if return_annotation is None:
            self._return_annotation = tuple()
        elif is_tuple_t(return_annotation):
            self._return_annotation = return_annotation.__args__
        else:
            self._return_annotation = (return_annotation,)

        # the following was meant to satisfy mypy, but it actually made matters worse
        # queue_annotations = tuple((Queue[t] for t in return_annotation))
        # queue_tuple_annotation = tuple[queue_annotations]
        # self._output_queues: queue_tuple_annotation = tuple((Queue[t]() for t in return_annotation))
        # Tuple mapping output index -> SharedMemoryQueue producer endpoint.
        self._output_queues: tuple = tuple(
            (SharedMemoryQueue(session_queue_name(f"{self._hash}_{i}")) for i, t in enumerate(self._return_annotation))
        )
        # Tag each queue with a stable *logical* name (``{node}_output_{i}``). The shared-memory name
        # is a per-run hash (with an ownership prefix inside a runtime session), which the editor can't derive;
        # this attribute lets the tap map the runtime queue back to the node/port the editor subscribes
        # to (``{node}_output_{index}``). Best-effort: never let tagging break queue construction.
        for _i, _q in enumerate(self._output_queues):
            try:
                _q._logical_name = f"{self._name}_output_{_i}"  # noqa: SLF001
            except Exception:
                pass
        self._len = len(self._output_queues)

    @property
    def types(self) -> tuple[type, ...]:
        """Get declared return types for the outputs.

        Returns:
            Tuple of return types in declaration order. For a function returning ``tuple[T0, T1]``,
            this returns ``(T0, T1)``.
        """
        return self._return_annotation

    def __len__(self) -> int:
        """Get number of outputs."""
        return self._len

    def __iter__(self) -> Iterator[SharedMemoryQueue]:
        return iter(self._output_queues)

    def __getitem__(self, channel: int) -> SharedMemoryQueue:
        """Get the producer queue for a specific output channel.

        Args:
            channel: Zero-based output channel index.

        Returns:
            SharedMemoryQueue for the selected output.

        Raises:
            IndexError: If index is out of range ``[0, len(self))``.
        """
        if channel < 0 or channel >= self._len:
            raise IndexError(f"Channel index is out of range: {self._name}")
        return self._output_queues[channel]

    def open(self) -> None:
        """Open all output queues for publishing.

        Must be called before ``push()``. Initializes shared-memory state and producer
        indices for each channel queue.
        """
        for queue in self._output_queues:
            queue.open()

    def close(self) -> None:
        """Close all output queues and release shared-memory resources.

        Should be paired with ``open()`` during scheduler shutdown.
        """
        for queue in self._output_queues:
            queue.close()

    def push(self, result: tuple) -> None:
        """Publish one node invocation result into the output queues.

        Args:
            result: Tuple of values in output order. Its length must exactly match
                the number of output queues.

        Behavior:
            - ``result[i]`` is pushed to output queue ``i``.
            - ``None`` values are skipped (no push for that output).
            - ``const`` wrapper is removed from all published values.

        Invariant:
            ``len(result) == len(self._output_queues)`` must hold.
        """
        assert len(result) == len(self._output_queues), f"Output queue {self._name} got wrong results"
        for r, q in zip(result, self._output_queues, strict=True):
            if r is None:
                continue
            q.push(const_cast(r))
