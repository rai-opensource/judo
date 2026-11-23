<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# unLocked

`judo.unlocked` is the Python module for **unLocked**, a dataflow runtime for Python. Wrap any callable in a `Node`, connect nodes with shared-memory queues, pick an input policy, and run. The framework handles serialization, transport, and scheduling — user code is just functions.

See the design docs for the full rationale and architecture details:

- [Lock-free shared-memory IPC middleware](https://www.notion.so/theaiinstitute/Lock-free-shared-memory-IPC-middleware-240ed28d3849809dac78c32a8c7ec7a5)
- [Debugging with unLocked](https://www.notion.so/theaiinstitute/Debugging-with-unLocked-2afed28d384980d6a107eede613631bd)

## Design Goals

**No boilerplate.** A node is a plain Python callable. There are no base classes to inherit from, no schemas to define, and no manual serialize/deserialize calls.

**Shared-memory transport.** Data moves through POSIX shared memory (`/dev/shm`), not sockets. Send and receive are memory writes and reads. Syscalls still happen inside the `SharedMemory*` classes to allocate new frames, but frame size is tuned against the estimated throughput so that frame allocation amortizes to roughly one syscall per second.

**Deterministic scheduling (planned).** The architecture admits a single-threaded deterministic execution order (BFS over the compute graph) for reproducible testing and debugging. This scheduler is not implemented yet — only the `Threaded` scheduler ships today.

**Visualization nodes.** Foxglove and Viser are wired in as regular nodes.

**Sim and real in one pipeline.** Today, a simulated robot and a real robot can run inside the same asynchronous node graph, which makes it easy to swap or mix them during bring-up and evaluation. The north star is to extend this so the same graph drives training of hierarchical policies with async execution and simulated delays — that part is not implemented yet.

## Why Not an Existing Tool?

Zenoh and ZMQ are message-passing transports, not dataflow runtimes — they require explicit pub/sub wiring and do not support arbitrary Python objects over shared memory without marshalling to C types or byte arrays. ROS 2 and LCM require IDL schemas and have no deterministic single-thread scheduler. None of them integrate recording and visualization at the framework level, and none are designed to run inside a training loop.

## unLocked vs ROS/ROS2 and Other Middleware

unLocked is a **dataflow runtime for Python**, not a general-purpose message bus. That framing drives every trade-off below.

### Benefits

- **No boilerplate, no IDL.** A node is just a Python function — parameters are inputs, the return annotation is outputs. ROS 2 / LCM require `.msg`/IDL schemas and codegen; unLocked has none. Any picklable Python object flows through directly.
- **Zero-copy shared memory by default.** Sends/receives are memory writes/reads, not socket round-trips or serialization to C types. numpy arrays and torch tensors are written *next to* the pickle stream (PEP 574) and read as `memoryview`s over the same segment. ROS 2 only gets true zero-copy with careful loaned-message + fixed-size type setups; Zenoh/ZMQ are message transports that still marshal.
- **Explicit per-input consumption semantics.** The `Policy` on each edge (`Lossless`, `Latest`, `Optional`, `KeepLatest`, `OneOf`) precisely controls what a node reads each step. In ROS you emulate this with queue depths, `message_filters`, and custom callback logic; here freshness-vs-completeness is a first-class, declared choice.
- **Sim and real in one graph.** A simulated and a real robot can run in the same async node graph, making swap/mix during bring-up trivial — a designed-in goal, not a bolt-on.
- **Framework-level recording & visualization.** Foxglove and Viser are ordinary nodes, and MCAP recording is part of the graph. No separate rosbag/bridge layer.
- **Declarative layer + editor.** `GraphBuilder` gives serialize/materialize/round-trip, plus a browser editor with live edge stats, master-clock pause/step, and dynamic graph regeneration — tighter than the typical `rqt_graph` + rosbag workflow.
- **Lightweight footprint.** A pure Python module in-repo; no ROS distro, no DDS discovery/config, no separate build ecosystem to adopt.

### Downsides (where ROS/ROS2 wins)

- **Python-only (today).** No C++/Rust/multi-language node story like ROS 2's rclcpp/rclpy. For C++ real-time nodes, ROS 2 is far ahead.
- **Single-host, shared-memory-bound.** `/dev/shm` transport is same-machine only. ROS 2 / Zenoh / DDS give networked, multi-machine pub/sub with discovery out of the box; unLocked has no distributed transport.
- **Tiny ecosystem.** ROS has thousands of packages (nav2, MoveIt, drivers, TF, perception). unLocked has none of that — you build it yourself.
- **Threaded scheduler only.** The deterministic single-thread BFS scheduler is *planned, not implemented*; the full reproducibility story isn't there yet.
- **Fewer safety rails.** No mature QoS profiles, lifecycle-managed nodes, or security (SROS2). Shutdown tiers/grace exist, but the framework is younger and less battle-tested.
- **Pickle-based.** Convenient, but ties you to Python and to trusting your own producers (don't unpickle untrusted data); there is no cross-language schema contract.

**Bottom line:** choose unLocked for a low-boilerplate, high-throughput, single-host **Python** dataflow pipeline that moves large tensors zero-copy and blends sim + real with built-in recording/visualization and a graph editor. Stay with ROS 2 when you need multi-language nodes, multi-machine networking, mature QoS/security/lifecycle, or the vast existing package ecosystem.

## Pitfalls & Things to Watch

- **Thread count.** The only scheduler today (`Threaded`) runs **one thread per node**. Many nodes ⇒ many threads ⇒ GIL contention and context-switch overhead for CPU-bound Python work. Group related work into fewer nodes, or split heavy nodes across processes so each gets its own interpreter.
- **Threads *outside* a Node are invisible to the runtime.** A thread you spawn yourself won't freeze on debug pause and isn't scheduled/paced — a real correctness/safety hazard (physics or a command stream can keep moving while the graph "looks paused"). Prefer Nodes; if you must spawn a thread, mark it managed (`mark_managed_thread()` / a `@freezes_when_paused` pacer). The editor flags unmanaged threads as dashed pills.
- **Policy choice is behavior, not tuning.** `Lossless` blocks and preserves every message (can build backpressure/latency if a consumer is slow); `Latest` favors freshness and *drops* older data; `KeepLatest` reuses stale data when nothing new arrives. Picking the wrong one silently changes correctness.
- **Memory reclamation is gated by the slowest consumer.** A frame is freed only after the slowest live view passes it. One slow/stalled consumer on a high-rate edge keeps `/dev/shm` frames alive and grows memory. Watch for laggards.
- **Frame sizing / throughput.** Allocation amortizes to ~one syscall/sec only when frame size matches the estimated rate. Badly mismatched rates mean more syscalls (too small) or wasted shared memory (too large).
- **Crash cleanup.** Killed processes can leak `/dev/shm/*_output_*` segments and hold ports; the runner sweeps these on start, and `//judo/unlocked/graph:runner -- --kill` clears dangling children.
- **Feedback loops need `warmup`.** Cyclic graphs (e.g. a control loop) must seed initial values or they deadlock waiting on inputs.
- **Same-host assumption.** No networking — architect around single-machine deployment.

## Core Concepts

**`Node`.** Wraps a Python callable. The callable's parameter list defines its inputs; the return type annotation defines its outputs. A node returning `tuple[float, bool]` produces two output values; a node returning `int` produces one.

```python
def fn(x: int, y: str) -> tuple[float, bool]:  # node business logic: 2 inputs, 2 outputs
    ...
```

**`OutputStage`.** Binds each of the node's return values to a `SharedMemoryQueue` that downstream nodes can connect to. Access the queue for an output by index: `node.output_stage[0]`.

**`InputStage`.** Binds each of the node's input arguments. `input_stage.connect(...)` takes one entry per positional argument; each entry is either a `(SharedMemoryQueue, Policy)` pair — in which case `connect` consumes the queue and produces a `SharedMemoryQueueView` bound to that argument — or a plain Python value, which is bound directly as a constant:

```python
node.input_stage.connect(
    5,                                       # constant: 5 -> first argument
    (upstream.output_stage[1], Latest()),    # queue view (Latest policy) -> second argument
)
```

**`Policy`.** Decides which value an input reads from its `SharedMemoryQueueView` on each execution step of the node function. The framework provides:

- `Lossless` — consume one value per step in FIFO order; block until one is available.
- `Latest` — block for at least one value, then jump to the newest and drop everything older.
- `Optional` — non-blocking; return the newest value if any is available, otherwise pass `None` into the function for this step.
- `KeepLatest` — return the newest value, but do **not** consume it: if nothing newer arrives by the next step, the same value is reused.
- `OneOf` — used across several inputs sharing the same `OneOf` instance: the node executes as soon as any one of those inputs has a new value; the triggering input receives that value, the other `OneOf`-bound inputs receive `None`.

**`SharedMemoryQueue` and `SharedMemoryQueueView`.** A `SharedMemoryQueue` is the lock-free, append-only buffer behind every output value. The queue is split into fixed-size `Frame`s — shared-memory blocks under `/dev/shm` allocated lazily as the producer fills them. Each frame packs multiple messages with an offset table for O(1) random access. Each consumer holds its own `SharedMemoryQueueView`: an independent read cursor that maps a message index to `(frame, offset)` and deserializes in place. A frame is released only after the slowest live view has advanced past every message in it.

Messages are serialized with Pickle protocol 5 (PEP 574). Out-of-band buffers (numpy arrays, torch tensors, bytearrays) are written next to the pickle stream rather than copied into it, so consumers deserialize messages as zero-copy `memoryview`s over the same shared memory the producer wrote.

**`Threaded`.** A scheduler that takes a collection of nodes and runs each on its own thread until any node raises `NodeStop`. Multiple `Threaded` instances can run side by side in the same process or across processes.

## Usage Example

Fibonacci sequence with feedback loop and timer-driven scheduling:

```python
import logging
import signal

from judo.unlocked.logging_util import setup_logger
from judo.unlocked import Node
from judo.unlocked.nodes.periodic_timer import make_periodic_timer_node
from judo.unlocked.policy import Lossless
from judo.unlocked.schedule import StopAll, Threaded

setup_logger(log_level=logging.INFO)


def fibonacci(_: int, a: int, b: int) -> tuple[int, int]:
    """Compute next Fibonacci number."""
    next_val = a + b
    logging.info("Next fibonacci number is %d", next_val)
    return b, next_val


# Create nodes: timer runs at 10 Hz, fibonacci maintains state via warmup + feedback
timer = make_periodic_timer_node("timer", frequency=10)
fib_node = Node("fibonacci", fibonacci, warmup=[(0, 1)])

# Connect: timer triggers fibonacci; fibonacci feeds back to itself
fib_node.input_stage.connect(
    (timer.output_stage[0], Lossless()),
    (fib_node.output_stage[0], Lossless()),  # feedback: previous 'b'
    (fib_node.output_stage[1], Lossless()),  # feedback: previous result
)

# Run on a single thread
schedule = Threaded("main", (timer, fib_node))
schedule.start()
signal.signal(signal.SIGINT, StopAll((schedule,)))
schedule.join()
```

Or use one of the built-in examples:

```bash
python -m judo.unlocked.example.fibonacci      # classic example
python -m judo.unlocked.example.pub_sub        # async pub/sub
python -m judo.unlocked.example.loopback       # feedback loop
python -m judo.unlocked.example.multiplex      # aggregate queues
```

## Graph Builder & Editor UI

The wiring above is the low-level, imperative API. For anything beyond a small graph, author it
declaratively with **`GraphBuilder`** instead: describe nodes, shared instances, timers, edges, and
processes as plain data, then serialize / materialize / run it.

```python
from judo.unlocked.graph.builder import GraphBuilder

def build() -> GraphBuilder:
    g = GraphBuilder("demo")
    g.timer("clock", frequency=10)                   # emits a "tick" port at 10 Hz
    g.node("logger", log_tick, inputs=["tick"])      # a plain callable
    g.connect("clock.tick", "logger.tick", policy="latest")
    return g
```

Ports are inferred from the function signature and return annotation, edge policies are the same
five above, and the spec round-trips to JSON/YAML. A **browser editor** (`judo-graph-ui`) renders
any builder as a live graph with configurable build args, an inspector, master-clock pause/step, and
live per-edge stats.

See [`graph/README.md`](graph/README.md) for the full builder API and editor documentation.

## Key Components

- `node.py`: node data structures.
- `input_stage.py`: input binding and policy wiring.
- `output_stage.py`: output queue creation and publishing.
- `policy.py`: subscription semantics (`Lossless`, `Latest`, `Optional`, `KeepLatest`, `OneOf`).
- `shmque.py`: shared-memory queue internals.
- `schedule.py`: scheduler and node execution runtime (`Threaded`).
- `nodes/periodic_timer.py`: timer source nodes.
- `nodes/logger.py`: logging sink nodes

## Operational Notes

- Return annotations define output channel count.
- `warmup` seeds feedback loops before steady-state execution.
- `Latest` favors freshness; `Lossless` favors completeness.
- Memory reclamation depends on the slowest active consumer view.
