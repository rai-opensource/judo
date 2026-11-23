# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from inspect import Signature
from itertools import chain
from typing import Any, Iterable

from judo.unlocked.const import const
from judo.unlocked.shmque import SharedMemoryQueue, SharedMemoryQueueView


class InputStage:
    """Manages node function inputs by coupling views and policies over shared memory queues.

    InputStage binds a function signature to input sources, enabling asynchronous multi-consumer
    patterns over shared memory. Each connected input is represented as a (view, policy) pair:

    - View: an independent consumer cursor into a SharedMemoryQueue. Multiple views can read
      from the same queue, each with its own begin index (cursor position). Data is only
      physically reclaimed when all consumers have advanced past it.

    - Policy: defines consumption semantics for that view (e.g., Lossless takes oldest unread;
      Latest drops backlog and takes newest; Optional returns None when empty). Policies control
      both visibility (ready/wait) and advancement (get/next) over a view.

    This design enables:
    - Multiple independent consumers reading the same queue at different rates
    - Fine-grained backpressure control per consumer via policy
    - Asynchronous pipelines where inputs to a node may not be synchronized in time

    Note: InputStage itself does NOT synchronize inputs across multiple queues. If a node
    consumes from multiple producers (e.g., segmentation and detection from the same image),
    correlation must be handled at the node level (e.g., via frame IDs in output payloads).
    """

    def __init__(self, name: str, signature: Signature):
        self._name = name
        self._signature = signature
        self._arg_len = 0
        self._arg_inputs: tuple[tuple[SharedMemoryQueueView, Any], ...] = ()
        self._kwarg_inputs: dict[str, tuple[SharedMemoryQueueView, Any]] = {}

    # def __len__(self) -> int:
    #     """Number of inputs."""
    #     return len(self._parameters)

    @property
    def all_inputs(self) -> Iterable:
        """Get all input view and policy pairs.

        Returns an iterable of (view, policy) tuples, combining positional and keyword inputs.
        Used internally for bulk operations (open, close, wait, ready, next).
        """
        return chain(self._arg_inputs, self._kwarg_inputs.values())

    @property
    def ready(self) -> bool:
        """Check if all inputs have data available according to their policies.

        Returns True iff for every (view, policy) pair, policy.ready(view) is True.

        Semantics depend on policy:
        - Lossless: view has unread items (len(view) > 0)
        - Latest: view has unread items
        - Optional: always True (can return None)
        - KeepLatest: view has unread items
        - OneOf: any view has unread items or a trigger is pending

        This is a non-blocking check; use wait() to block until data arrives.
        """
        return all((policy.ready(view) for view, policy in self.all_inputs))

    def wait(self) -> bool:
        """Block until all inputs have data available.

        Returns:
            True once every input is ready; False if any input wait returned
            False (e.g. the scheduler called wake() to release it on shutdown).

        Semantics: Sequential; waits on each (view, policy) pair in order. For
        queue views this parks on the shared-memory queue condition variable
        (notified when a producer calls queue.push()), with no timeout -- see
        SharedMemoryQueueView.wait. On shutdown InputStage.wake() signals every
        view so a thread parked on any input here returns promptly.
        """
        for view, policy in self.all_inputs:
            if not policy.wait(view):
                return False
        return True

    def wake(self) -> None:
        """Release any thread parked in wait() on these inputs (shutdown).

        Signals every input view condition so a blocking, timeout-free wait
        returns and its node thread can observe the stopped state and exit.
        """
        for view, _ in self.all_inputs:
            view.wake()

    def open(self) -> None:
        """Open all input shared memory views for reading.

        Must be called before any data access. For queue views, this attaches to shared
        memory segments and initializes the view's cursor. For constant values, this is a no-op.

        Should be called during InputStage initialization, typically by the Scheduler.
        """
        for view, _ in self.all_inputs:
            view.open()

    def close(self) -> None:
        """Close all input shared memory views and release resources.

        Should be called during shutdown to clean up shared memory handles and allow
        the queue producer to reclaim frames. Must be paired with open().
        """
        for view, _ in self.all_inputs:
            view.close()

    def connect(self, *args: Any, **kwargs: Any) -> None:
        """Bind input sources to function arguments via views and policies.

        Args:
            *args: Positional arguments corresponding to the node function's positional parameters.
                   Each arg can be:
                   - A constant value (int, str, tensor, etc.). The caller passes the bare
                     value — no wrapping required. `connect` itself wraps it in a `_ValueView`
                     paired with a `_ValuePolicy` (always ready, always returns the same value).
                   - A tuple (queue, policy) where queue is SharedMemoryQueue and policy is a Policy
                     instance (Lossless, Latest, Optional, KeepLatest, OneOf): creates a view into
                     the queue and couples it with the given policy

            **kwargs: Keyword arguments corresponding to the node function's keyword parameters.
                      Same format as positional args above.

        Example:
            >>> node_fn = lambda img, scale=1.0: ...
            >>> input_stage = InputStage("my_node", signature(node_fn))

            >>> img_queue = SharedMemoryQueue("images")
            >>> input_stage.connect(
            ...     (img_queue, Lossless()),  # img parameter: queue input with Lossless policy
            ...     scale=2.0,                # scale parameter: constant value
            ... )

            In this case:
            - _arg_inputs[0] = (view_into_img_queue, Lossless())
            - _kwarg_inputs["scale"] = (_ValueView(2.0), _ValuePolicy())

        Invariant: After connect(), all_inputs contains tuples of (view, policy) ready for
        the node's execution loop: wait() -> input_args() -> node_fn() -> next()
        """
        _args = [_make_view_with_policy(a) for a in args]
        _kwargs = {k: _make_view_with_policy(v) for k, v in kwargs.items()}
        bindings = self._signature.bind(*_args, **_kwargs)
        self._arg_inputs = bindings.args
        self._kwarg_inputs = bindings.kwargs

    def input_args(self) -> tuple[list[Any], dict[str, Any]]:
        """Retrieve current input values by delegating to each policy's get() method.

        Returns:
            (args, kwargs): tuple of (list of values, dict of values) ready to pass to the node
                           function as *args, **kwargs.

        Semantics: For each (view, policy) pair, this calls policy.get(view), which:
        - Lossless: returns view.top() (oldest unread item)
        - Latest: returns latest item (skips backlog)
        - Optional: returns latest item or None if view is empty
        - KeepLatest: returns latest item, keeps one buffered
        - OneOf: returns value from whichever view triggered, None for others

        This is called after wait() succeeds and before next().
        """
        args = [policy.get(view) for view, policy in self._arg_inputs]
        kwargs = {name: policy.get(view) for name, (view, policy) in self._kwarg_inputs.items()}
        return args, kwargs

    def next(self) -> None:
        """Advance all views after the node function has consumed the current inputs.

        Semantics: For each (view, policy) pair, this calls policy.next(view), which:
        - Lossless: pops one item (advances view.begin by 1)
        - Latest: pops all dropped items + 1 (skips the backlog)
        - Optional: pops all if any exist, else no-op
        - KeepLatest: pops all but one (keeps latest buffered for next tick)
        - OneOf: pops the triggered view, others are no-op

        Expected to be called after the node function completes successfully.
        This ensures that on the next step, the node function receives the next batch of inputs.
        """
        for view, policy in self.all_inputs:
            policy.next(view)


def _make_view_with_policy(arg: Any) -> tuple:
    if not isinstance(arg, tuple):
        return _ValueView(arg), _ValuePolicy()
    if not isinstance(arg[0], SharedMemoryQueue):
        return _ValueView(arg), _ValuePolicy()
    queue, policy = arg
    return queue.make_view(), policy


class _ValueView:
    def __init__(self, value: Any) -> None:
        self._value = value

    @property
    def queue_name(self) -> str:
        return str(self._value)

    @property
    def value(self) -> Any:
        return const(self._value)

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def wake(self) -> None:
        # Constant inputs never block, so nothing to release. Present so
        # InputStage.wake() can call view.wake() uniformly across all inputs.
        pass


class _ValuePolicy:
    def ready(self, value_view: _ValueView) -> bool:
        return True

    def wait(self, value_view: _ValueView) -> bool:
        return True

    def next(self, value_view: _ValueView) -> None:
        pass

    def get(self, value_view: _ValueView) -> Any:
        return value_view.value
