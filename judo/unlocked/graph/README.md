<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# unLocked Graph — Builder API & Editor UI

`judo.unlocked.graph` is the **declarative layer** on top of the [unLocked](../README.md) dataflow
runtime. Instead of hand-wiring `Node` objects and `input_stage.connect(...)` calls imperatively,
you describe a graph as plain data with `GraphBuilder`. The same description can then be:

- **serialized** to JSON / YAML for the React Flow editor,
- **materialized** into live `Node` / `Threaded` objects for execution,
- **round-tripped** — load a graph edited in the browser and run it.

This is *not* a replacement for the imperative `Node` API; it sits on top of it.

```
builder.py     GraphBuilder — author / serialize / materialize a graph
runner.py      run a builder (or a saved graph JSON) as scheduler processes
example/       minimal runnable graphs (fibonacci, loopback)
ui/            the browser editor: stdlib HTTP server + React Flow front-end
```

---

## Quick start

```bash
# Run a built-in example graph
python -m judo.unlocked.graph.runner \
    --builder=judo.unlocked.graph.example.loopback_graph:build

# Open the editor (seeds from a builder module:callable)
judo-graph-ui                       # defaults to judo.app.graph:build
```

A minimal builder:

```python
from judo.unlocked.graph.builder import GraphBuilder

def build() -> GraphBuilder:
    g = GraphBuilder("demo")
    g.timer("clock", frequency=10)                  # emits a "tick" port at 10 Hz
    g.node("logger", log_tick, inputs=["tick"])     # a plain callable
    g.connect("clock.tick", "logger.tick", policy="latest")
    return g
```

---

## Builder API

### Core concepts

| Concept      | What it is                                                                           |
| ------------ | ------------------------------------------------------------------------------------ |
| **node**     | a callable wrapped as a runtime `Node`; its params are input ports, its return is output ports |
| **edge**     | a `src.port -> dst.port` connection carrying a **policy** (how the input reads its queue) |
| **instance** | a shared object used by method nodes or constructors, built once per materialization |
| **process**  | a scheduling group (a `Threaded` scheduler); nodes are tagged with `process=`          |
| **tier**     | shutdown ordering — tier 0 stops first; within a tier processes stop in parallel       |

### Authoring methods

- **`g.node(node_id, target, *, inputs=, outputs=, process=, instance=, bound_inputs=, warmup=, pre_open=, post_close=, close_order=, **construct_kwargs)`**
  Register a node. `target` is a **real Python reference** (class / function / method) — converted
  to its `"module:qualname"` string for serialization, so authoring code stays IDE-navigable
  (go-to-def, find-refs, rename) and type-checked. Returns the node id for chaining.

- **`g.instance(instance_id, factory, *, process=, **construct_kwargs)` → handle**
  Register a shared object. `handle.node(...)` adds nodes bound to that object and pins them to the
  same process (the shared object lives in one address space). Pass the **factory**, not a built
  object — the spec must stay JSON-serializable. `handle.ref()` passes the shared object to another
  constructor in that same process (see [Shared constructor dependencies](#shared-constructor-dependencies)).

- **`g.timer(node_id, *, frequency= | period_s=, duration_s=, process=)`**
  Sugar for a periodic-timer node emitting a single `tick` port.

- **`g.connect(src, dst, *, policy="lossless", policy_kwargs=, extra_args=)`**
  Wire `src_node.out_port` to `dst_node.in_port`. Port names are validated against each node.

- **`g.process(name, *, grace_s=, stop_before=, role=, tier=)`**
  Configure a process's shutdown ordering and teardown budget. `role` picks a default grace
  (`safety_critical` = 30 s, `recorder` = 120 s, else 3 s); `stop_before` adds explicit stop-before
  DAG edges; `tier` hard-floors the tier.

### Port inference — why you rarely declare `inputs=` / `outputs=`

For a **plain callable target**, both port lists are inferred, so you usually pass neither:

- **Inputs** come from the parameter names (`_infer_inputs`). A leading `self`/`cls` is dropped and
  bound to the shared instance — which is why a method target must be added via `instance.node(...)`.
- **Outputs** come from a `@node_outputs` decorator first, else the return annotation
  (`_infer_outputs`):
  - `-> None` → no ports
  - `-> SpotState` → `["spot_state"]` (single typed return, snake-cased)
  - `-> tuple[SpotState, np.ndarray]` → `["spot_state_0", "ndarray_1"]` (fixed-length tuple)

You **must** declare them only when inference genuinely can't work, or to override it:

| Situation                                    | Why declaration is needed                                              |
| -------------------------------------------- | --------------------------------------------------------------------- |
| **Factory target** (returns a `Node`)        | its signature describes *construction* args, not runtime ports — pass both `inputs=` and `outputs=` |
| **`*args` parameter**                        | no fixed arity — supply `expand={param: [ports...]}` or explicit `inputs=` |
| **Variable-length tuple return** (`tuple[X, ...]`) | port *count* is unknown — pass `outputs=` (or `@node_outputs`)   |
| **Rename / reshape**                         | `inputs=` / `outputs=` optionally rename the inferred ports            |

This is why `g.timer(...)` hardcodes `inputs=[], outputs=["tick"]` — `make_periodic_timer_node` is a
factory, so its ports can't be inferred.

### Policies (edge semantics)

The `policy=` on an edge decides which value an input reads from its queue each step:

| Policy        | Behavior                                                                       |
| ------------- | ------------------------------------------------------------------------------ |
| `lossless`    | one value per step, FIFO, blocks until available (completeness)                |
| `latest`      | block for ≥1, then jump to newest and drop older (freshness)                   |
| `optional`    | non-blocking; newest if any, else `None`                                       |
| `keep_latest` | newest value, but not consumed — reused until something newer arrives          |
| `one_of`      | shared across inputs; fires as soon as any one has a new value (others `None`)  |

### Constants and lifecycle

- **`bound_inputs={port: value}`** pins an input to a literal at materialize time; no edge may
  connect to it. The editor renders it as an inline constant chip.
- **`pre_open` / `post_close`** are lifecycle hooks (bound to the instance when the node has one).
- **`close_order`** sequences open/close *within* a process (lower opens/closes first) — e.g. a
  safety-critical node's `post_close` (safe-fallback) runs before the state/mocap producers it
  shares a process with tear down.

### Serialize · materialize · round-trip

```python
g.to_json()      # / to_yaml() / to_dict() — name, nodes, edges, instances, processes, shutdown_tiers
GraphBuilder.from_yaml(...)   # / from_dict(...) — rebuild a graph (from_dict(json.loads(text)) for JSON)
g.materialize()  # -> {node_id: Node}, fully wired (instances built once, Hydra-resolved kwargs)
g.build_processes()  # -> ([Threaded, ...], nodes)
g.run()          # materialize, schedule, block until SIGINT, then tiered safe shutdown
```

Instance `kwargs` may hold nested Hydra `{_target_: ...}` sub-configs, built by
`hydra.utils.instantiate` at materialize time (with `_partial_: true` support), so a ctor can take
live sub-objects declaratively without a bespoke `make_*` factory.

### Shared constructor dependencies

`handle.ref()` returns a fresh JSON-safe dictionary, exactly `{"_instance_ref_": "instance_id"}`.
Pass it as a **direct constructor kwarg** to `g.instance(...)` or a factory-style `g.node(...)`:

```python
sim = g.instance("sim", make_simulation, process="simulation")
link = g.instance("link", NetworkSimRobotLink, process="simulation", simulation=sim.ref(), robot_prefix="spot")
link.node("link_step", NetworkSimRobotLink.command_step)
g.node("camera", make_camera_node, process="simulation", simulation=sim.ref(),
       inputs=["tick"], outputs=["image"])
```

#### Three rules

1. **Share one object.** In the example above, `link` and `camera` receive the exact same simulation
   object. It is created once each time the graph is materialized.
2. **Use the same process.** Set `process="simulation"` on the simulation and every consumer.
   References do not move nodes into the same process automatically. Method nodes must also use
   their instance's process. The defaults (`<instance_id>_process` / `<node_id>_process`) are separate.
3. **Pass references directly to constructors.** Use `simulation=sim.ref()`, not a reference nested
   inside a list, dictionary, or Hydra config. References cannot be used as method-node kwargs or
   runtime input bindings; use edges for runtime data.

#### What happens when the graph runs?

- Dependencies are constructed first, regardless of their order in JSON/YAML.
- Hydra builds ordinary constructor kwargs first; shared objects are injected afterward. This
  preserves object identity, including for dataclasses. The serialized spec contains reference
  markers, never live objects.
- Invalid references raise `ValueError` before factories are imported or called. Validation rejects
  unknown ids, cycles, malformed markers, nested references, and references across processes.
  Cross-process references are rejected in dry-run and `materialize()` too.

**Why require one process?** The runner constructs the full graph in the parent, then `Threaded`
forks the process groups. Objects copied into different processes are no longer shared. The editor
uses this same runner; neither path generates Python or trims the spec to one process.

#### Saving and editing graphs

- Every saved instance includes its `process`, even if it has no method nodes.
- When extracting part of a graph, keep every instance it depends on, including indirect
  dependencies, and preserve their `process` fields. Keeping only instances named by nodes is not
  enough.
- The editor preserves references and process ownership. Reference fields are read-only; rebuilding
  uses the builder's new references rather than old editor values.

**Older JSON:** A referenced instance without an explicit process can infer it when all its method
nodes use one process. If it has no method nodes, specify `process` explicitly. Older graphs without
references keep their existing runtime behavior.

### Running a builder

```bash
# From a module:callable, with kwargs
python -m judo.unlocked.graph.runner \
    --builder=judo.app.graph:build \
    --builder-kwarg robot=spot --builder-kwarg use_mocap=true

# From a saved graph (e.g. edited in the editor)
python -m judo.unlocked.graph.runner --graph-json=/path/to/graph.json

# Inspect without running
python -m judo.unlocked.graph.runner --builder=judo.app.graph:build --dry-run
```

#### Shared-memory cleanup

Linux and macOS use the same cleanup strategy. Each graph run creates a session with:

- **An ownership lock** that remains held while the parent or any forked scheduler worker
  retains its inherited descriptor. A surviving worker keeps the session live even if its
  parent dies.
- **A manifest** listing the exact names of its shared-memory segments. Queue index and frame
  names are recorded **before allocation**, including frames created by worker threads.

**When cleanup runs:**

1. Before materializing the graph, the runner checks for abandoned sessions.
2. It reclaims a session's segments only if it can acquire that session's ownership lock.
3. On exit, it attempts the same cleanup for its own session. `--kill` uses the same liveness
   check for shared-memory cleanup.

Segments are removed by name with POSIX `shm_unlink`, not by scanning `/dev/shm`.
Concurrent apps and restarts with new graph namespaces remain isolated. PID reuse and paused
workers do not trigger cleanup. Legacy or unowned segments are left untouched because their
ownership cannot be established safely.

**Where bookkeeping is stored:**

| Platform | Persistent metadata directory |
| --- | --- |
| Linux | `~/.local/state/starfish/starfish-sessions-<uid>/` |
| macOS | `~/Library/Application Support/starfish/starfish-sessions-<uid>/` |

These directories hold the locks and manifests, not the queue data. Only their location differs
by platform. Shared-memory names stay at 28 bytes regardless of channel or frame counter size,
and the same lifecycle tests exercise real POSIX segments on both platforms without platform
mocking.

---

## Editor UI

`judo-graph-ui` starts a dependency-free stdlib HTTP server (port `8770`) serving a React Flow
editor (`ui/index.html` + `ui/app.js`). It is **generic** — it works for any `--builder module:callable`.

- `--builder judo.app.graph:build` seeds the initial graph.
- The **Run** button launches `python -m judo.unlocked.graph.runner --graph-json=...` with the edited graph.

### Feature overview

- **Configurable build args.** `GET /builder/args` introspects the builder's signature (types +
  hints) and renders controls automatically: `bool` → toggle, `int`/`float` → number, `str` → text,
  `Literal[...]` → dropdown. A builder may expose `param_choices()` for dynamic dropdowns and
  `param_meta()` for extra UI metadata. The last config is persisted and restored on reload.

- **Dynamic graph generation.** Changing an arg posts `POST /builder/build {kwargs}`; the server
  `importlib.reload()`s the builder module, calls `builder(**kwargs)`, serializes with `to_dict()`,
  and the browser re-renders. Selecting a different `robot` / `task` can produce a structurally
  different graph from the same builder.

- **Inspector.** Select a node to edit its constructor kwargs via a typed editor (schema resolves
  from a class `config_schema()` → `CONFIG_TYPE` dataclass → signature). App-level fields (e.g.
  `task`, `robot`) render disabled — they re-propagate on rebuild. Select an edge to set its policy
  and `expected_hz`, and (during a run) see live stats.

- **Instances as container boxes.** Nodes sharing an `instance=` are grouped into one labelled box
  (e.g. `foxglove` / `viser` / `spot_control`), so the diagram shows one box per shared object.

- **Hiding items.** An eye toggle dims a node / edge / process (visual only, wiring unchanged) — used
  to focus on the core flow by hiding infrastructure (foxglove, viser, recorders). "Show All" restores.

- **Build & Run.**
  - **Build** (`POST /build/graph`) bazel-builds the runner so later Runs take the fast path.
  - **Run** (`POST /run/graph`) writes the in-memory spec to a tempfile and launches the run target;
    the default (Debug) run installs the debug tap (edge values + master-clock pause/step), while a
    plain Run installs a lightweight stats-only tap (flow rate/count only, no clock gating).
  - **Stop / Kill** — graceful SIGINT (then escalate) vs immediate SIGKILL.

### Debug features

- **Debug tap.** When `UNLOCKED_TAP_URL` is set, `debug_tap.install()` monkey-patches
  `SharedMemoryQueue.push` and starts daemon threads. It always collects per-edge **flow stats**
  (count, EMA period → `actual_hz`, last-push time); in Debug mode it also captures the latest
  deserialized **value** per subscribed edge (pickle-5, zero-copy). The server keeps a rolling
  history for a time-travel scrubber while paused.

- **Master-clock pause / step (generic).** A server-authoritative virtual clock is mirrored to every
  child. **Pause** gates pausable nodes (timers); **Step** issues permits (step-all or a single node).
  Nodes read time via `debug_tap.now_ns()`; loop pacers use `@freezes_when_paused` to freeze on pause.
  This controls **time itself**, so it applies to any unLocked graph.

- **Edge stats.** Each edge shows `expected_hz` (configured, persisted to `edge_expected_hz.json`
  beside the builder module) vs `actual_hz` (measured), plus message count / queue depth / staleness,
  and flags rate drift so you can spot a stalled producer at a glance.

### Pitfall — threads outside an unLocked Node

Any thread the scheduler does **not** manage will **not** freeze on Pause — physics / mocap / a
command stream could keep moving while the graph "looks paused" (a safety bug). The tap scans for
these (`_scan_unmanaged_threads`) and the editor draws thread **pills** on the process frame:

- **↻ managed** (solid) — pause-aware (freezes). Make a thread managed by calling
  `debug_tap.mark_managed_thread()` or pacing it with a `@freezes_when_paused` method (e.g. `Rate.sleep`).
- **⚠ unmanaged** (dashed) — rogue; won't freeze. Double-click to open its source.

Rule of thumb: prefer Nodes; if you must spawn a thread, mark it managed.

---

## Files

| File            | Purpose                                                                  |
| --------------- | ------------------------------------------------------------------------ |
| `builder.py`    | `GraphBuilder` — authoring, port inference, serialize, materialize, run   |
| `runner.py`     | run a builder or saved graph JSON; shm cleanup; `--dry-run` report        |
| `example/`      | minimal runnable graphs (`fibonacci_graph`, `loopback_graph`)             |
| `ui/server.py`  | stdlib HTTP server: build args, run/debug control, tap store, edge config |
| `ui/app.js`     | React Flow front-end (nodes, edges, process frames, inspector, thread pills) |
| `ui/index.html` | editor shell + styling                                                    |
