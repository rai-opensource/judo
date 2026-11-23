# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Declarative graph builder for unLocked.

Captures node definitions and connections as plain data so the same description can be:
  - serialized to JSON for the React Flow editor,
  - materialized into live `Node`/`Schedule` objects for execution,
  - round-tripped (load JSON edited in the browser, run it).

This is *not* a replacement for the existing imperative API; it sits on top of it.
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from judo.unlocked.annotation import NODE_OUTPUTS_ATTR
from judo.unlocked.node import Node, NodeType
from judo.unlocked.policy import KeepLatest, Latest, Lossless, OneOf, Optional

_POLICY_REGISTRY: dict[str, Callable[..., Any]] = {
    "lossless": Lossless,
    "latest": Latest,
    "keep_latest": KeepLatest,
    "optional": Optional,
    "one_of": OneOf,
}


@dataclass
class _NodeSpec:
    id: str
    target: str  # "module:qualname"
    kwargs: dict[str, Any] = field(default_factory=dict)
    warmup: list[Any] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    # Optional lifecycle hooks ("module:qualname" strings).
    pre_open: str | None = None
    post_close: str | None = None
    # Optional grouping annotations (cosmetic + used by codegen / scheduler).
    process: str | None = None  # process group id (see GraphBuilder._processes)
    instance: str | None = None  # shared-instance id (see GraphBuilder._instances)
    # Input ports pre-bound to literal values. The port name is still listed in
    # ``inputs``; the editor renders the constant inline and no edge may
    # connect to it. At materialize time these are prepended to
    # ``input_stage.connect(...)`` in port-declaration order.
    bound_inputs: dict[str, Any] = field(default_factory=dict)
    # Open/close sequence WITHIN a process (lower runs first). The scheduler opens + closes its
    # nodes front-to-back, so a lower ``close_order`` means this node's ``pre_open`` runs earlier
    # AND its ``post_close`` runs earlier. Used so a safety-critical node (e.g. Spot control, whose
    # ``post_close`` runs safe-fallback) tears down BEFORE the state / mocap producers it shares a
    # process with -- keeping them alive to publish the safe-fallback motion (see app.py's node
    # order). Ties keep declaration order (stable sort).
    close_order: int = 0
    # How the node is driven (``NodeType`` value, stored as its string so the spec stays JSON).
    # Anything other than ``unlocked`` marks an externally driven node: the scheduler manages its
    # stages and open/close hooks (so ``warmup`` still publishes) but spawns no worker thread for
    # it. Such a node is spun by ``Node.callback`` / ``Node.generator`` from outside the graph --
    # e.g. a viser GUI handler publishing straight into the shm queue on its own thread. It
    # declares no inputs, so a worker loop would spin instead of park (see NodeType).
    node_type: str = NodeType.UNLOCKED.value


@dataclass
class _EdgeSpec:
    src: str  # "node_id.out<i>"
    dst: str  # "node_id.in<j>"
    policy: str = "lossless"
    # Positional args prepended to ``input_stage.connect(...)`` before the
    # ``(queue, Policy())`` tuple. Used for foxglove subscribers that take a
    # topic string or similar constant.
    extra_args: list[Any] = field(default_factory=list)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class _InstanceSpec:
    """A shared object used by method nodes and/or constructor references."""

    id: str
    factory: str  # "module:qualname" of class / factory
    kwargs: dict[str, Any] = field(default_factory=dict)
    # Persist ownership even when this instance has no method nodes. None is legacy JSON.
    process: str | None = None


@dataclass
class _ProcessSpec:
    """A scheduling group (currently always Threaded)."""

    name: str
    kind: str = "Threaded"
    members: list[str] = field(default_factory=list)
    # Shutdown ordering + grace (see GraphBuilder.process / _shutdown_tiers). ``stop_before`` lists
    # other process names this one must be stopped BEFORE (stop-before DAG edges); ``tier`` is an
    # optional hard floor (else the tier is derived); ``grace_s`` / ``role`` set the teardown budget.
    stop_before: list[str] = field(default_factory=list)
    tier: int | None = None
    grace_s: float | None = None
    role: str | None = None  # "safety_critical" | "recorder" | None -> default grace


# Per-role default shutdown budget (seconds). Safety-critical processes run safe-fallback
# (stow + sit + power-off, ~10-20 s on hardware); the MCAP recorder flushes its footer.
_DEFAULT_SHUTDOWN_GRACE_S: dict[str | None, float] = {
    "safety_critical": 30.0,
    "recorder": 120.0,
    None: 3.0,
}


class GraphBuilder:
    """Builds a graph description and (optionally) materializes it."""

    def __init__(self, name: str = "graph") -> None:
        self._name = name
        self._nodes: dict[str, _NodeSpec] = {}
        self._edges: list[_EdgeSpec] = []
        self._instances: dict[str, _InstanceSpec] = {}
        self._processes: dict[str, _ProcessSpec] = {}
        # Free-form, JSON-serializable graph-level metadata that round-trips through
        # ``to_dict``/``from_dict`` (so it survives the editor's serialize -> run path). Used e.g.
        # for ``post_run`` hooks (see :meth:`add_post_run_hook`) that :func:`run_builder` invokes
        # after the graph stops.
        self._metadata: dict[str, Any] = {}
        # Runtime-only status; deliberately not serialized with the graph description.
        self.shutdown_clean: bool = False

    def add_post_run_hook(self, target: str, **kwargs: Any) -> None:
        """Register a ``"module:qualname"`` callable to run after the graph stops.

        The hook is stored in the graph metadata (so it survives serialization and fires on both the
        CLI and editor run paths) and is invoked by :func:`judo.unlocked.graph.runner.run_builder`
        once ``run()`` returns, as ``fn(since=<run_start_epoch>, **kwargs)``. It should return an
        ``int`` process-exit code (falsy = ok). Keeps ``run_builder`` generic -- the core layer never
        imports the hook's module until it resolves it by name at call time.

        Reserve ``include_shutdown_status=True`` in kwargs to opt into an additional
        ``shutdown_clean`` bool argument. The runner removes that reserved key; without
        opt-in the hook's arguments are unchanged. Status covers orderly cleanup and
        lifecycle hooks, not all node runtime errors.
        """
        self._metadata.setdefault("post_run", []).append({"target": target, "kwargs": kwargs})

    # -- authoring ---------------------------------------------------------

    def node(
        self,
        node_id: str,
        target: Callable[..., Any],
        *,
        warmup: list[Any] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        expand: dict[str, list[str]] | None = None,
        pre_open: str | Callable[..., Any] | None = None,
        post_close: str | Callable[..., Any] | None = None,
        process: str | None = None,
        instance: str | None = None,
        bound_inputs: dict[str, Any] | None = None,
        close_order: int = 0,
        node_type: NodeType | str = NodeType.UNLOCKED,
        **construct_kwargs: Any,
    ) -> str:
        """Register a node.

        ``target`` is the callable that the Node will execute or a factory that returns a ``Node``
        (passed as a real Python reference, which is converted to its ``"module:qualname"`` form for
        serialization).

        Construction arguments for the target are passed **inline** as keyword
        arguments (mirroring :meth:`instance`), e.g.
        ``g.node("clock", make_master_clock_node, resolution_s=1e-3, inputs=[], outputs=["tick"])``.
        They become the node's serialized kwargs. The reserved keywords above
        (``warmup``/``inputs``/``outputs``/``expand``/``pre_open``/``post_close``/
        ``process``/``instance``/``bound_inputs``) therefore cannot also be
        construction-argument names.

        Port inference: when ``inputs=`` is omitted the input ports are taken
        from the target's parameter names, dropping a leading ``self``/``cls``
        (which is bound to the shared instance at materialize time). Output ports
        come from a :func:`judo.unlocked.node_outputs` decorator on the target,
        else from the return annotation (``-> None`` → no outputs, fixed-length
        ``-> tuple[...]`` → one ``<element_type>_<index>`` port per element, any
        other typed return → a single port named after the type, e.g.
        ``-> SpotState`` → ``spot_state``).

        ``inputs=`` may be passed to rename/reshape the input ports.
        ``outputs=`` likewise names or renames the output ports: it overrides a
        :func:`judo.unlocked.node_outputs` decorator, names the outputs of an
        otherwise-undecorated target, and supplies the port count for a
        **variable-length tuple return** (e.g. a camera node whose port count is
        fixed at construction), where the count cannot be inferred.

        ``expand`` maps a target's ``*args`` parameter to the concrete port names
        it should fan out to, e.g. ``expand={"images": ["img_left", "img_right"]}``
        for ``def __call__(self, tick, *images)``. A target with ``*args`` requires
        either ``expand=`` or an explicit ``inputs=``.

        For **factory-style targets** (a function/class that *returns* a ``Node``)
        pass ``inputs=`` and ``outputs=`` explicitly: the signature describes
        construction parameters, not the runtime node's ports.

        ``bound_inputs`` is a ``{port_name: constant_value}`` mapping for inputs
        that should be pre-bound to a literal at materialize time (no edge may
        connect to them); the port name must still appear in ``inputs``.

        Construction kwargs are resolved through Hydra at materialize time
        (identically to :meth:`instance`): kwargs made purely of JSON primitives
        pass through unchanged, while a nested ``{_target_: "module.qualname",
        ...}`` sub-config is built by ``hydra.utils.instantiate`` (honoring
        ``_partial_: true``). This applies only to plain function / factory
        targets; a method-bound node (added via :meth:`instance`) is constructed
        by its shared instance, so its kwargs are the method's, not a ctor's.

        ``node_type=NodeType.CALLBACK`` / ``NodeType.GENERATOR`` declares an **externally driven**
        node: the scheduler manages its
        stages and ``open``/``close`` hooks (so ``warmup`` still publishes and ``post_close``
        still runs in ``close_order``) but gives it no worker thread. Something outside the
        graph spins it via ``Node.callback`` / ``Node.generator`` -- e.g. a viser GUI handler
        publishing straight into the shm queue on its own thread. Or an external stream consumer
        that takes a generator to publish commands to a proprietary api.

        A direct constructor kwarg may be ``handle.ref()`` to reuse an instance
        in the same process. References are injected AFTER Hydra resolves the
        other kwargs, preserving identity even for dataclasses. Nested refs and
        refs on method-bound nodes are not supported.

        Returns the node id for chaining.
        """
        if node_id in self._nodes:
            raise ValueError(f"duplicate node id: {node_id}")
        if process is None and instance is None:
            process = f"{node_id}_process"
        if expand is not None and inputs is not None:
            raise ValueError(f"node {node_id!r}: pass either inputs= or expand=, not both")
        if not callable(target):
            raise TypeError(
                f"node {node_id!r}: target must be a callable reference "
                f"(class, function, or method), got {target!r}"
            )
        target_str = _callable_to_target(target)
        sig = inspect.signature(target)
        kind = _return_kind(sig.return_annotation)
        if kind == "node":
            if (node_type is NodeType.CALLBACK or node_type == NodeType.CALLBACK.value) and outputs is None:
                raise ValueError(
                    f"node {node_id!r}: factory target {target_str!r} returns a callback Node; "
                    f"pass explicit outputs= for this node to be correctly connected and visualized. {node_type}"
                )
            elif (node_type is NodeType.GENERATOR or node_type == NodeType.GENERATOR.value) and inputs is None:
                raise ValueError(
                    f"node {node_id!r}: factory target {target_str!r} returns a generator Node; "
                    f"pass explicit inputs= for this node to be correctly connected and visualized. {node_type}"
                )
            elif (node_type is NodeType.UNLOCKED or node_type == NodeType.UNLOCKED.value) and (
                inputs is None or outputs is None
            ):
                raise ValueError(
                    f"node {node_id!r}: factory target {target_str!r} returns a Node; "
                    f"pass explicit inputs= and outputs= (its signature describes "
                    f"construction parameters, not the runtime node's ports)"
                )
            in_ports = inputs
            out_ports = outputs
            if in_ports is None:
                in_ports = []
            if out_ports is None:
                out_ports = []
        else:
            if node_type is NodeType.CALLBACK or node_type == NodeType.CALLBACK.value:
                in_ports = []
            else:
                in_ports = (
                    inputs if inputs is not None else self._infer_inputs(node_id, sig, instance=instance, expand=expand)
                )
            if node_type is NodeType.GENERATOR or node_type == NodeType.GENERATOR.value:
                out_ports = []
            else:
                out_ports = outputs if outputs is not None else _infer_outputs(target, sig)
        bound = dict(bound_inputs or {})
        unknown = [p for p in bound if p not in in_ports]
        if unknown:
            raise ValueError(
                f"node {node_id!r}: bound_inputs contains unknown port(s) {unknown}; " f"known inputs: {in_ports}"
            )
        # Catch manually spun nodes with incompatible specifications.
        if (node_type is NodeType.CALLBACK or node_type == NodeType.CALLBACK.value) and in_ports != []:
            raise ValueError(f"node {node_id!r}: has input ports but is set as a CALLBACK node.")
        if (node_type is NodeType.GENERATOR or node_type == NodeType.GENERATOR.value) and out_ports != []:
            raise ValueError(f"node {node_id!r}: has output ports but is set as a GENERATOR node.")
        self._nodes[node_id] = _NodeSpec(
            id=node_id,
            target=target_str,
            kwargs=construct_kwargs,
            warmup=warmup or [],
            inputs=in_ports,
            outputs=out_ports,
            pre_open=_callable_to_target(pre_open) if pre_open is not None else None,
            post_close=_callable_to_target(post_close) if post_close is not None else None,
            process=process,
            instance=instance,
            bound_inputs=bound,
            close_order=close_order,
            node_type=NodeType(node_type).value,
        )
        if process is not None:
            self._processes.setdefault(process, _ProcessSpec(name=process)).members.append(node_id)
        return node_id

    def timer(
        self,
        node_id: str,
        *,
        frequency: float | None = None,
        period_s: float | None = None,
        duration_s: float | None = None,
        process: str | None = None,
    ) -> str:
        """Add a periodic timer node that emits a single ``tick`` output port.

        Sugar over :func:`judo.unlocked.nodes.periodic_timer.make_periodic_timer_node`
        that supplies the fixed ``[] -> ["tick"]`` port shape (a factory target's
        signature describes its construction args, not the runtime node's ports,
        so they cannot be inferred). Specify exactly one of ``frequency`` or
        ``period_s``. ``process`` follows the same default as :meth:`node`
        (``f"{node_id}_process"``); pass it to co-schedule timers in one group.

        Returns the node id for chaining.
        """
        from judo.unlocked.nodes.periodic_timer import make_periodic_timer_node

        timer_kwargs: dict[str, Any] = {"name": node_id}
        if frequency is not None:
            timer_kwargs["frequency"] = frequency
        if period_s is not None:
            timer_kwargs["period_s"] = period_s
        if duration_s is not None:
            timer_kwargs["duration_s"] = duration_s
        return self.node(
            node_id,
            make_periodic_timer_node,
            inputs=[],
            outputs=["tick"],
            process=process,
            **timer_kwargs,
        )

    @staticmethod
    def _infer_inputs(
        node_id: str,
        sig: inspect.Signature,
        *,
        instance: str | None,
        expand: dict[str, list[str]] | None,
    ) -> list[str]:
        """Derive input port names from a callable's signature.

        A leading ``self``/``cls`` is dropped (and implies the target is a
        bound method, so it must be added via ``instance.node(...)``). A
        ``*args`` parameter has no fixed arity and must be expanded to concrete
        ports via ``expand={param_name: [ports...]}``.
        """
        params = list(sig.parameters.values())
        if params and params[0].name in ("self", "cls"):
            if instance is None:
                raise ValueError(
                    f"node {node_id!r}: target is a method (first parameter "
                    f"{params[0].name!r}); add it via instance.node(...) so it binds "
                    f"to a shared instance"
                )
            params = params[1:]
        expand = dict(expand or {})
        in_ports: list[str] = []
        for p in params:
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY):
                in_ports.append(p.name)
            elif p.kind == p.VAR_POSITIONAL:
                if p.name not in expand:
                    raise ValueError(
                        f"node {node_id!r}: target has *{p.name}; pass "
                        f"expand={{{p.name!r}: [...port names]}} (or an explicit inputs=) "
                        f"to fan it out to concrete ports"
                    )
                in_ports.extend(expand.pop(p.name))
        if expand:
            raise ValueError(
                f"node {node_id!r}: expand keys {list(expand)} do not match any " f"*args parameter of the target"
            )
        return in_ports

    def instance(
        self,
        instance_id: str,
        factory: Callable[..., Any],
        *,
        process: str | None = None,
        **construct_kwargs: Any,
    ) -> "_InstanceHandle":
        """Register a shared instance and return a handle for adding its nodes.

        ``factory`` is the class or ``make_*`` function that constructs the
        object, passed as a real Python reference (converted to its
        ``"module:qualname"`` form so the spec stays JSON-serializable). Any
        extra keyword arguments are its constructor arguments and are stored as
        the instance's ``kwargs`` so the graph can re-create the object at
        materialization. Pass the *factory*, not an already-built object: the
        spec must stay JSON-serializable. The runner materializes in the parent
        before forking schedulers; live identity is shared only within a process.

        Declarative sub-objects via Hydra: because the object is rebuilt from
        JSON-serializable kwargs, a constructor arg that needs a live sub-object
        (a config dataclass, a bound provider) cannot be passed directly. Pass a
        nested Hydra ``{_target_: "module.qualname", ...}`` sub-config instead:
        at materialize time :meth:`materialize` resolves the kwargs through
        :func:`_instantiate_kwargs`, which hands any kwargs containing a nested
        ``_target_`` to ``hydra.utils.instantiate`` (recursively building each
        sub-object and honoring ``_partial_: true`` for a bound
        ``functools.partial``). Kwargs made purely of JSON primitives skip Hydra
        entirely and pass through unchanged. Authoring code typically builds
        these sub-configs from real references (e.g. app.py's
        ``_hydra_instantiable_cfg`` / ``_hydra_partial_cfg`` helpers) so the spec
        stays serializable while remaining navigable in an IDE.

        Idempotent: re-registering the same ``instance_id`` with the same
        ``factory`` and constructor kwargs returns a fresh handle; a conflicting
        re-registration raises.

        Pass ``other.ref()`` as a direct constructor kwarg to depend on another
        instance in the same process. Dependencies are built first, regardless
        of declaration order; missing references and cycles raise ValueError.
        References are not supported inside containers or Hydra sub-configs.

        ``process`` defaults to ``f"{instance_id}_process"`` (each instance gets
        its own scheduling process, since the shared object lives in one address
        space); pass it explicitly only to co-schedule with another group.

        The returned handle's :meth:`_InstanceHandle.node` tags every node
        with this ``instance_id`` and ``process``, so nodes that share an object
        are written together at one call site and are guaranteed to land in the
        same process (the shared object lives in one address space). Example::

            policy = g.instance("e2e_policy", E2EPolicy, process="policy_process",
                                current_task=task, robot_name=robot)
            policy.node("vision_policy", E2EPolicy.__call__,
                        inputs=["tick"], outputs=["command"], pre_open=E2EPolicy.start)
        """
        if not callable(factory):
            raise TypeError(
                f"instance {instance_id!r}: factory must be a callable reference "
                f"(class or make_* function), got {factory!r}"
            )
        factory_str = _callable_to_target(factory)
        resolved_process = process if process is not None else f"{instance_id}_process"
        existing = self._instances.get(instance_id)
        if existing is not None:
            if existing.factory != factory_str or existing.kwargs != construct_kwargs:
                raise ValueError(f"instance {instance_id!r} already registered with a different factory/kwargs")
            if existing.process is None:
                existing.process = resolved_process
        else:
            self._instances[instance_id] = _InstanceSpec(
                id=instance_id, factory=factory_str, kwargs=construct_kwargs, process=resolved_process
            )
        return _InstanceHandle(self, instance_id, resolved_process)

    def add_process(self, name: str, *, kind: str = "Threaded", members: list[str] | None = None) -> str:
        """Pre-declare a process group (members may also be tagged on nodes via ``process=``)."""
        spec = self._processes.setdefault(name, _ProcessSpec(name=name, kind=kind))
        spec.kind = kind
        for m in members or []:
            if m not in spec.members:
                spec.members.append(m)
        return name

    def process(
        self,
        name: str,
        *,
        kind: str = "Threaded",
        grace_s: float | None = None,
        stop_before: list[str] | None = None,
        role: str | None = None,
        tier: int | None = None,
    ) -> str:
        """Configure a scheduler process's shutdown ordering + grace.

        Shutdown tiers (stop order, consumer-first) are derived from the data flow by default -- a
        process is stopped BEFORE the producers it reads from (see :meth:`_shutdown_tiers`). Use this
        to override that policy:

          * ``role`` picks a default grace: ``"safety_critical"`` (30 s -- runs safe-fallback),
            ``"recorder"`` (120 s -- flushes the mcap footer), else 3 s;
          * ``grace_s`` sets the teardown budget explicitly (overrides ``role``);
          * ``stop_before`` adds explicit stop-before DAG edges to other process names;
          * ``tier`` hard-floors the tier (rarely needed; edges usually suffice).

        Idempotent + additive, so it can be called alongside ``process=`` node tags.
        """
        spec = self._processes.setdefault(name, _ProcessSpec(name=name, kind=kind))
        spec.kind = kind
        if grace_s is not None:
            spec.grace_s = grace_s
        if role is not None:
            spec.role = role
        if tier is not None:
            spec.tier = tier
        if stop_before:
            spec.stop_before = sorted({*spec.stop_before, *stop_before})
        return name

    # -- shutdown ordering -------------------------------------------------

    def _process_of(self, endpoint: str) -> str:
        """Process owning the node an edge endpoint (``"node_id.port"``) refers to."""
        node_id = endpoint.rsplit(".", 1)[0]
        spec = self._nodes.get(node_id)
        return spec.process if spec and spec.process else "main"

    def _process_names(self) -> set[str]:
        """All scheduler process names (from node ``process=`` tags + explicit declarations)."""
        names = {spec.process or "main" for spec in self._nodes.values()}
        names.update(self._processes.keys())
        return names

    def _shutdown_edges(self) -> set[tuple[str, str]]:
        """Stop-before edges ``(a, b)`` meaning "stop process ``a`` before process ``b``".

        These are the *explicit* ordering constraints declared via ``process(stop_before=...)``. Note
        the data flow is deliberately NOT used to auto-derive them: a feedback control graph
        (sense -> compute -> act -> sense) has cross-process cycles, so its reverse is not a DAG --
        which is exactly why ``app.py`` assigns tiers by hand. Ordering here is likewise explicit
        (via ``stop_before`` and/or ``tier`` pins); :meth:`suggest_shutdown_tiers` offers a
        best-effort data-flow layering for the editor to seed from.
        """
        edges: set[tuple[str, str]] = set()
        names = self._process_names()
        for spec in self._processes.values():
            for other in spec.stop_before:
                if spec.name in names and other in names:  # ignore edges to absent processes
                    edges.add((spec.name, other))
        return edges

    def suggest_shutdown_tiers(self) -> list[list[str]]:
        """Best-effort tier suggestion from the data flow (consumer-first), cycles broken.

        For the editor's "suggest" action only -- NOT used at runtime. Builds the reverse
        process-level data-flow graph (consumer -> producer) and layers it, dropping any edge that
        would deepen a cycle so the result is always a DAG. The author then refines + pins tiers.
        """
        names = sorted(self._process_names())
        rev: set[tuple[str, str]] = set()
        for e in self._edges:
            producer, consumer = self._process_of(e.src), self._process_of(e.dst)
            if producer != consumer and producer in names and consumer in names:
                rev.add((consumer, producer))
        tier = {n: 0 for n in names}
        for _ in range(len(names) + 1):
            changed = False
            for a, b in sorted(rev):
                if tier[b] <= tier[a] and tier[a] - tier[b] < len(names):  # skip cycle back-edges
                    tier[b] = tier[a] + 1
                    changed = True
            if not changed:
                break
        by_tier: dict[int, list[str]] = {}
        for n, t in tier.items():
            by_tier.setdefault(t, []).append(n)
        return [sorted(by_tier[t]) for t in sorted(by_tier)]

    def _shutdown_tiers(self) -> list[list[str]]:
        """Order processes into shutdown tiers -- the list index is the tier, tier 0 stops first.

        Longest-path layering of the stop-before DAG (:meth:`_shutdown_edges`): a process sits one
        tier after the latest process that must stop before it. Explicit ``tier`` pins seed the
        layering (a floor). Within a tier processes stop in parallel. Raises on a cycle.
        """
        names = self._process_names()
        edges = self._shutdown_edges()
        tier: dict[str, int] = {n: (self._processes[n].tier if n in self._processes else None) or 0 for n in names}
        for _ in range(len(names) + 1):
            changed = False
            for a, b in edges:  # "stop a before b" => tier[b] must exceed tier[a]
                if tier[b] <= tier[a]:
                    tier[b] = tier[a] + 1
                    changed = True
            if not changed:
                break
        else:
            raise ValueError("shutdown ordering has a cycle (check process stop_before / tier pins)")
        by_tier: dict[int, list[str]] = {}
        for n, t in tier.items():
            by_tier.setdefault(t, []).append(n)
        return [sorted(by_tier[t]) for t in sorted(by_tier)]  # compacted: list index == effective tier

    def _grace_for(self, name: str) -> float:
        """Shutdown budget (seconds) for a process: explicit ``grace_s``, else role default."""
        spec = self._processes.get(name)
        if spec and spec.grace_s is not None:
            return spec.grace_s
        return _DEFAULT_SHUTDOWN_GRACE_S[spec.role if spec else None]

    def connect(
        self,
        src: str,
        dst: str,
        *,
        policy: str = "lossless",
        extra_args: list[Any] | None = None,
        policy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Connect ``src_node.src_port`` to ``dst_node.dst_port`` with a policy.

        Port names must match those registered for each node (inferred from the
        callable's signature, or supplied explicitly via ``inputs=``/``outputs=``
        in :meth:`node`).
        """
        if policy not in _POLICY_REGISTRY:
            raise ValueError(f"unknown policy {policy!r}; known: {list(_POLICY_REGISTRY)}")
        src_id, src_port = _split_port(src)
        dst_id, dst_port = _split_port(dst)
        if src_id not in self._nodes:
            raise ValueError(f"unknown source node {src_id!r}")
        if dst_id not in self._nodes:
            raise ValueError(f"unknown destination node {dst_id!r}")
        if src_port not in self._nodes[src_id].outputs:
            raise ValueError(f"node {src_id!r} has no output port {src_port!r}; known: {self._nodes[src_id].outputs}")
        if dst_port not in self._nodes[dst_id].inputs:
            raise ValueError(f"node {dst_id!r} has no input port {dst_port!r}; known: {self._nodes[dst_id].inputs}")
        if dst_port in self._nodes[dst_id].bound_inputs:
            raise ValueError(
                f"node {dst_id!r} input port {dst_port!r} is bound to a constant; " f"cannot connect an edge to it"
            )
        self._edges.append(
            _EdgeSpec(
                src=src,
                dst=dst,
                policy=policy,
                extra_args=extra_args or [],
                policy_kwargs=policy_kwargs or {},
            )
        )

    # -- JSON --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a plain JSON-able dict (name / nodes / edges / instances / processes)."""
        return {
            "name": self._name,
            "nodes": [
                {
                    "id": n.id,
                    "target": n.target,
                    "kwargs": n.kwargs,
                    "warmup": n.warmup,
                    "inputs": n.inputs,
                    "outputs": n.outputs,
                    "pre_open": n.pre_open,
                    "post_close": n.post_close,
                    "process": n.process,
                    "instance": n.instance,
                    "bound_inputs": n.bound_inputs,
                    "close_order": n.close_order,
                    "node_type": n.node_type,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "from": e.src,
                    "to": e.dst,
                    "policy": e.policy,
                    "extra_args": e.extra_args,
                    "policy_kwargs": e.policy_kwargs,
                }
                for e in self._edges
            ],
            "instances": [
                {"id": i.id, "factory": i.factory, "kwargs": i.kwargs, "process": i.process}
                for i in self._instances.values()
            ],
            "processes": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "members": list(p.members),
                    "stop_before": list(p.stop_before),
                    "tier": p.tier,
                    "grace_s": p.grace_s,
                    "role": p.role,
                }
                for p in self._processes.values()
            ],
            # Computed stop order for the editor to render (list index == tier, 0 stops first). A
            # cyclic override falls back to a single flat tier so the graph still displays.
            "shutdown_tiers": self._shutdown_tiers_safe(),
            # Free-form graph-level metadata (e.g. post_run hooks); round-trips via from_dict.
            "metadata": self._metadata,
        }

    def _shutdown_tiers_safe(self) -> list[list[str]]:
        """Like :meth:`_shutdown_tiers` but degrades to one flat tier on a cycle (for display)."""
        try:
            return self._shutdown_tiers()
        except ValueError:
            return [sorted(self._process_names())]

    def to_json(self, indent: int = 2) -> str:
        """Serialize the graph to a JSON string (see :meth:`to_dict`)."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Serialize the graph to YAML -- same schema as :meth:`to_dict` / :meth:`to_json`.

        YAML is the human-editable interchange format for the whole graph: the emitted document IS
        ``to_dict()`` (name / nodes / edges / instances / processes), so it round-trips through
        :meth:`from_yaml` and materializes identically to the JSON path. Instance ``kwargs`` may hold
        nested Hydra ``_target_`` sub-configs, which read naturally as nested YAML mappings.
        """
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "GraphBuilder":
        """Rebuild a graph from a YAML document produced by :meth:`to_yaml`."""
        import yaml

        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphBuilder":
        """Rebuild a graph from a dict produced by :meth:`to_dict`."""
        g = cls(name=data.get("name", "graph"))
        md = data.get("metadata")
        if isinstance(md, dict):
            g._metadata = md  # noqa: SLF001
        for i in data.get("instances", []):
            g._instances[i["id"]] = _InstanceSpec(  # noqa: SLF001
                id=i["id"], factory=i["factory"], kwargs=i.get("kwargs", {}), process=i.get("process")
            )
        for n in data["nodes"]:
            g._nodes[n["id"]] = _NodeSpec(  # noqa: SLF001
                id=n["id"],
                target=n["target"],
                kwargs=n.get("kwargs", {}),
                warmup=n.get("warmup", []),
                inputs=n.get("inputs", []),
                outputs=n.get("outputs", []),
                pre_open=n.get("pre_open"),
                post_close=n.get("post_close"),
                process=n.get("process"),
                instance=n.get("instance"),
                bound_inputs=n.get("bound_inputs", {}),
                close_order=n.get("close_order", 0),
                node_type=n.get("node_type", NodeType.UNLOCKED.value),
            )
        for e in data["edges"]:
            g._edges.append(  # noqa: SLF001
                _EdgeSpec(
                    src=e["from"],
                    dst=e["to"],
                    policy=e.get("policy", "lossless"),
                    extra_args=e.get("extra_args", []),
                    policy_kwargs=e.get("policy_kwargs", {}),
                )
            )
        for p in data.get("processes", []):
            g._processes[p["name"]] = _ProcessSpec(  # noqa: SLF001
                name=p["name"],
                kind=p.get("kind", "Threaded"),
                members=list(p.get("members", [])),
                stop_before=list(p.get("stop_before", [])),
                tier=p.get("tier"),
                grace_s=p.get("grace_s"),
                role=p.get("role"),
            )
        # Backfill process members from node.process tags if processes section
        # was omitted (older payloads).
        for n_spec in g._nodes.values():  # noqa: SLF001
            if n_spec.process:
                proc = g._processes.setdefault(n_spec.process, _ProcessSpec(name=n_spec.process))  # noqa: SLF001
                if n_spec.id not in proc.members:
                    proc.members.append(n_spec.id)
        return g

    # -- materialization ---------------------------------------------------

    def _instance_order(self) -> list[str]:
        """Validate references without importing targets; return dependency-first instance ids.

        Ownership checks apply to reference-connected instances only, preserving
        legacy no-ref graphs. Node tags, not process.members, drive the scheduler.
        """
        dependencies = {
            inst.id: _instance_refs(inst.kwargs, f"instance {inst.id!r}") for inst in self._instances.values()
        }
        node_refs = {n.id: _instance_refs(n.kwargs, f"node {n.id!r}") for n in self._nodes.values()}
        for node in self._nodes.values():
            if node.instance is not None and node.instance not in self._instances:
                raise ValueError(
                    f"node {node.id!r}: unknown instance {node.instance!r} (known: {list(self._instances)})"
                )
            if node.instance is not None and node_refs[node.id]:
                raise ValueError(f"node {node.id!r}: instance refs require a constructor, not a method-bound node")
        order: list[str] = []
        visiting: list[str] = []
        done: set[str] = set()

        def visit(instance_id: str, source: str) -> None:
            if instance_id not in self._instances:
                raise ValueError(f"{source}: unknown instance reference {instance_id!r}")
            if instance_id in visiting:
                cycle = visiting[visiting.index(instance_id) :] + [instance_id]
                raise ValueError(f"instance reference cycle: {' -> '.join(cycle)}")
            if instance_id in done:
                return
            visiting.append(instance_id)
            for dependency in dependencies[instance_id].values():
                visit(dependency, f"instance {instance_id!r}")
            visiting.pop()
            done.add(instance_id)
            order.append(instance_id)

        for instance_id in self._instances:
            visit(instance_id, "graph")
        for node_id, refs in node_refs.items():
            for dependency in refs.values():
                visit(dependency, f"node {node_id!r}")

        involved = {i for i, refs in dependencies.items() if refs}
        involved.update(ref for refs in [*dependencies.values(), *node_refs.values()] for ref in refs.values())
        owners: dict[str, str] = {}
        for instance_id in sorted(involved):
            inst = self._instances[instance_id]
            processes = {n.process or "main" for n in self._nodes.values() if n.instance == instance_id}
            if inst.process is not None:
                processes.add(inst.process)
            if not processes:
                raise ValueError(f"instance {instance_id!r}: reference has no process owner; set instance process")
            if len(processes) != 1:
                raise ValueError(f"instance {instance_id!r}: cross-process instance reference: {sorted(processes)}")
            owners[instance_id] = next(iter(processes))

        def check(source: str, process: str, refs: dict[str, str]) -> None:
            for kwarg, dependency in refs.items():
                if process != owners[dependency]:
                    raise ValueError(
                        f"{source} kwarg {kwarg!r}: cross-process instance reference to {dependency!r} "
                        f"({process!r} != {owners[dependency]!r}); use the same process"
                    )

        for instance_id, refs in dependencies.items():
            if refs:
                check(f"instance {instance_id!r}", owners[instance_id], refs)
        for node in self._nodes.values():
            check(f"node {node.id!r}", node.process or "main", node_refs[node.id])
        return order

    def materialize(self) -> dict[str, Node]:
        """Instantiate live Node objects and wire them according to the edges.

        Returns a mapping ``{node_id: Node}``. The caller schedules them.

        Shared-instance support: nodes carrying ``instance=<id>`` and a
        ``module:Class.method`` target reuse a single object constructed by the
        instance's ``factory`` (with ``kwargs``). The method is then bound to
        that instance. Plain function targets ignore ``instance=``.

        A target that RETURNS a ``Node`` is adopted as-is rather than wrapped, so the factory
        keeps ownership -- what an externally driven node needs in order to be spun later via
        ``Node.callback``. :func:`_check_factory_node` enforces that such a Node still matches
        the spec that declared it, since none of the constructor-side fields can be applied to
        it after the fact.
        """
        # Pre-construct all shared instances (one object per id, regardless of
        # how many nodes reference it). Instance construction kwargs are resolved
        # through Hydra (:func:`_instantiate_kwargs`): plain-primitive kwargs pass
        # through unchanged, while nested ``_target_`` sub-configs are built by
        # ``hydra.utils.instantiate`` -- so an instance's ctor can take live
        # sub-objects (a config dataclass, a ``_partial_`` provider) declaratively,
        # without a bespoke ``make_*`` factory.
        instances: dict[str, Any] = {}
        for instance_id in self._instance_order():
            inst = self._instances[instance_id]
            factory = _import_target(inst.factory)
            instances[inst.id] = factory(**_resolve_kwargs(inst.kwargs, instances))

        nodes: dict[str, Node] = {}
        for spec in self._nodes.values():
            obj: Any = self._resolve_target(spec, instances)
            if isinstance(obj, Node):
                _check_factory_node(spec, obj)
                nodes[spec.id] = obj
            else:
                hook_kwargs: dict[str, Any] = {}
                if spec.pre_open:
                    hook_kwargs["pre_open"] = self._resolve_hook(spec.pre_open, spec.instance, instances)
                if spec.post_close:
                    hook_kwargs["post_close"] = self._resolve_hook(spec.post_close, spec.instance, instances)
                nodes[spec.id] = Node(
                    spec.id, obj, warmup=spec.warmup, node_type=NodeType(spec.node_type), **hook_kwargs
                )

        # Group incoming edges by destination node, ordered by input port index.
        by_dst: dict[str, list[_EdgeSpec]] = {}
        for e in self._edges:
            dst_node, _ = _split_port(e.dst)
            by_dst.setdefault(dst_node, []).append(e)

        # Wire each destination's input_stage by walking its declared input
        # ports in order. For each port we supply either:
        #   - the literal value from spec.bound_inputs (constant-bound), or
        #   - a (queue, Policy()) tuple from a matching incoming edge.
        # Legacy edge.extra_args (positional constants on the edge itself) are
        # prepended to preserve backward compatibility with older specs.
        for dst_id in self._nodes:
            spec = self._nodes[dst_id]
            dst_inputs = spec.inputs
            incoming = by_dst.get(dst_id, [])
            edge_by_port: dict[str, _EdgeSpec] = {_split_port(e.dst)[1]: e for e in incoming}
            legacy_extra: list[Any] = []
            args: list[Any] = []
            for port in dst_inputs:
                if port in spec.bound_inputs:
                    args.append(spec.bound_inputs[port])
                    continue
                edge = edge_by_port.get(port)
                if edge is None:
                    continue
                legacy_extra.extend(edge.extra_args)
                src_id, src_port = _split_port(edge.src)
                src_index = self._nodes[src_id].outputs.index(src_port)
                src_node = nodes[src_id]
                # ``outputs=`` is the graph's DECLARATION of a producer's ports, but the live
                # ``OutputStage`` is sized from the target's return annotation (one queue per
                # element of a ``tuple[...]`` return, else one). They can disagree -- most easily
                # for an externally driven node whose ``Node`` is built by its instance, where the
                # graph's ``outputs=`` is used only to resolve this index. Say so here: the bare
                # ``IndexError`` from ``OutputStage.__getitem__`` names neither the edge nor the
                # mismatch, and the fix is always "make the target return what the graph declares".
                if src_index >= len(src_node.output_stage):
                    raise ValueError(
                        f"edge {edge.src!r} -> {edge.dst!r}: node {src_id!r} declares outputs "
                        f"{self._nodes[src_id].outputs} but its target publishes {src_node.output_stage.types}, "
                        f"{len(src_node.output_stage)} channel(s). The declared ports must match "
                        f"the target's return annotation (a tuple return -> one channel per "
                        f"element)."
                    )
                args.append((src_node.output_stage[src_index], _POLICY_REGISTRY[edge.policy](**edge.policy_kwargs)))
            if args or legacy_extra:
                nodes[dst_id].input_stage.connect(*legacy_extra, *args)
        return nodes

    # -- shared-instance / method-binding helpers --------------------------

    def _resolve_target(self, spec: _NodeSpec, instances: dict[str, Any]) -> Any:
        """Resolve ``spec.target`` to a callable / Node object.

        Cases:
          * No ``instance=``  → import target; CALL it when it takes construction
            kwargs or when it is a factory (annotated ``-> Node``), otherwise use
            the plain function as-is.
          * ``instance=`` set and target is ``Module:Class.method`` → look up
            ``method`` on the shared instance and bind it; if that method is a
            factory (annotated ``-> Node``) it is CALLED and its Node returned,
            so an instance can build and own the Node it will later drive.
          * ``instance=`` set and target is ``Module:Class`` → just use the
            shared instance (caller expects a callable that's the instance
            itself, e.g. ``__call__``).
        """
        if spec.instance is None:
            target = _import_target(spec.target)
            if spec.kwargs:
                return target(**_resolve_kwargs(spec.kwargs, instances))
            if _returns_node(target):
                return target()
            return target

        if spec.instance not in instances:
            raise ValueError(f"node {spec.id!r}: unknown instance {spec.instance!r} " f"(known: {list(instances)})")
        obj = instances[spec.instance]
        _, qualname = spec.target.split(":", 1)
        parts = qualname.split(".")
        # ``Class.method`` → take the trailing attribute as the method name.
        # ``Class``        → call the instance directly.
        if len(parts) <= 1:
            target = _import_target(spec.target)
            return obj if isinstance(target, type) and isinstance(obj, target) else target
        method_name = parts[-1]
        if not hasattr(obj, method_name):
            raise AttributeError(
                f"node {spec.id!r}: instance {spec.instance!r} of type "
                f"{type(obj).__name__} has no attribute {method_name!r}"
            )
        attr = getattr(obj, method_name)
        if (_return_kind(inspect.signature(attr).return_annotation) == "node") and (
            spec.node_type in {NodeType.CALLBACK, NodeType.GENERATOR}
        ):
            if spec.kwargs != {}:
                raise ValueError(
                    f"""
                    node {spec.id!r}: has kwargs in specification, but is a manually scheduled node.
                    The node is initiated in the instance and the factory function should return the node
                    with no arguments."""
                )
            return attr()
        return attr

    def _resolve_hook(self, hook: str, instance_id: str | None, instances: dict[str, Any]) -> Callable[..., Any]:
        """Resolve a ``pre_open``/``post_close`` hook string to a callable.

        Same rules as :meth:`_resolve_target`: if the node has an instance,
        the trailing attribute is bound to it; otherwise the hook is imported.
        """
        if instance_id is not None and instance_id in instances:
            _, qualname = hook.split(":", 1)
            method_name = qualname.rsplit(".", 1)[-1]
            obj = instances[instance_id]
            if hasattr(obj, method_name):
                return getattr(obj, method_name)
        return _import_target(hook)

    # -- process scheduling ------------------------------------------------

    def build_processes(self) -> tuple[list[Any], dict[str, Node]]:
        """Materialize nodes and group them into scheduler processes.

        Returns ``(processes, nodes)`` where ``processes`` is a list of
        :class:`judo.unlocked.schedule.Threaded` instances ready to be
        ``.start()``ed, and ``nodes`` is the full materialized node map.

        Nodes without a ``process=`` tag are collected into a default
        ``"main"`` process.
        """
        # Local import to avoid a cycle and to keep this module light when
        # used only for description (e.g. by the editor).
        from judo.unlocked.schedule import Threaded

        nodes = self.materialize()
        groups: dict[str, list[_NodeSpec]] = {}
        for spec in self._nodes.values():
            groups.setdefault(spec.process or "main", []).append(spec)
        # Order each process's members by ``close_order`` (stable -> ties keep declaration order).
        # The scheduler opens + closes front-to-back, so this fixes the open/close sequence, e.g. the
        # Spot control node (safe-fallback in post_close) closes before the state / mocap producers
        # it shares a process with, which stay alive to publish the safe-fallback motion.
        processes = [
            Threaded(
                name=pname,
                nodes=[nodes[s.id] for s in sorted(specs, key=lambda s: s.close_order)],
            )
            for pname, specs in groups.items()
        ]
        return processes, nodes

    def run(self) -> None:
        """Materialize, schedule, and block until SIGINT/SIGTERM, then shut down tier by tier.

        Shutdown honors the stop-before DAG (:meth:`_shutdown_tiers`): consumer processes stop before
        the producers they read from, each tier gets its :meth:`_grace_for` budget, and stragglers
        are escalated (``stop`` -> ``join(grace)`` -> ``terminate`` -> ``kill``). This mirrors
        ``app.py``'s tiered teardown so a partial shutdown can never leave dangling subprocesses --
        or, for a ``safety_critical`` process, cut its safe-fallback short.

        ``shutdown_clean`` stays False until every scheduler reports orderly cleanup,
        has exited with code zero, and none required terminate/kill escalation. It does
        not certify error-free node execution; only lifecycle hook errors are tracked.
        """
        import logging as _logging
        import signal as _signal
        import time as _time

        self.shutdown_clean = False
        processes, _nodes = self.build_processes()
        if not processes:
            raise RuntimeError("graph has no nodes to run")
        by_name = {p.name: p for p in processes}
        tiers = self._shutdown_tiers()
        # Defensive: append any built process no tier covers so it is still torn down.
        covered = {n for names in tiers for n in names}
        leftover = [p.name for p in processes if p.name not in covered]
        if leftover:
            tiers = [*tiers, sorted(leftover)]

        stop_requested = False

        def request_stop(_signum: int, _frame: Any) -> None:
            # Python handlers can reenter: Event.set() can deadlock on its own lock.
            # Only latch here; all shutdown work stays in the main loop below.
            nonlocal stop_requested
            stop_requested = True

        _signal.signal(_signal.SIGINT, request_stop)
        _signal.signal(_signal.SIGTERM, request_stop)

        # Collect cyclic garbage BEFORE forking. Otherwise every child inherits it and finalizes it in
        # its own pre-spin ``gc.collect()``; finalizers that talk to a parent-only helper thread (e.g.
        # wandb's ``Api`` cleanup, which awaits its asyncio loop thread) then block that child forever.
        import gc as _gc

        _gc.collect()
        for p in processes:
            p.start()
        # Block until shutdown is requested or a scheduler exits on its own.
        while not stop_requested and all(p.alive for p in processes):
            _time.sleep(0.2)

        escalated = False
        for idx, names in enumerate(tiers):
            tier = [by_name[n] for n in names if n in by_name]
            if not tier:
                continue
            grace = max((self._grace_for(n) for n in names), default=3.0)
            _logging.info("shutdown tier %d (%s) with %.0fs grace", idx, ", ".join(names), grace)
            for p in tier:
                p.stop()
            deadline = _time.monotonic() + grace
            for p in tier:
                p.join(timeout=max(0.0, deadline - _time.monotonic()))
            stragglers = [p for p in tier if p.alive]
            if stragglers:
                escalated = True
            for p in stragglers:
                _logging.warning("scheduler %r did not exit within %.0fs; terminating", p.name, grace)
                p.terminate()
            for p in stragglers:
                p.join(timeout=2.0)
            for p in tier:
                if p.alive:
                    _logging.error("scheduler %r did not respond to terminate; killing", p.name)
                    p.kill()
                    p.join(timeout=2.0)

        self.shutdown_clean = not escalated and all(
            not p.alive and p.exitcode == 0 and p.shutdown_clean for p in processes
        )
        if not self.shutdown_clean:
            _logging.error("graph %r did not shut down cleanly (escalated=%s)", self._name, escalated)


# -- helpers ---------------------------------------------------------------


class _InstanceHandle:
    """Authoring handle returned by :meth:`GraphBuilder.instance`.

    Thin sugar over :meth:`GraphBuilder.node` that pins ``instance`` and
    ``process`` for every node added through it. It holds no state beyond the
    instance id and process name, so it does not affect serialization or
    materialization — those still read the builder's node/instance specs.
    """

    def __init__(self, builder: "GraphBuilder", instance_id: str, process: str | None) -> None:
        self._builder = builder
        self.id = instance_id
        self.process = process

    def ref(self) -> dict[str, str]:
        """JSON-safe reference for a direct instance/node constructor kwarg.

        The consumer must use this instance's process. No live object is stored.
        """
        return {"_instance_ref_": self.id}

    def node(
        self,
        node_id: str,
        target: Callable[..., Any] | Callable[[], Node],
        *,
        warmup: list[Any] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        expand: dict[str, list[str]] | None = None,
        pre_open: str | Callable[..., Any] | None = None,
        post_close: str | Callable[..., Any] | None = None,
        bound_inputs: dict[str, Any] | None = None,
        close_order: int = 0,
        node_type: NodeType | str = NodeType.UNLOCKED,
        **construct_kwargs: Any,
    ) -> str:
        """Add a node bound to this handle's shared instance and process."""
        return self._builder.node(
            node_id,
            target,
            warmup=warmup,
            inputs=inputs,
            outputs=outputs,
            expand=expand,
            pre_open=pre_open,
            post_close=post_close,
            process=self.process,
            instance=self.id,
            bound_inputs=bound_inputs,
            close_order=close_order,
            node_type=node_type,
            **construct_kwargs,
        )


def _split_port(ref: str) -> tuple[str, str]:
    node, port = ref.rsplit(".", 1)
    return node, port


def _returns_node(target: Callable[..., Any]) -> bool:
    """Whether ``target`` is a factory, i.e. annotated as returning a :class:`Node`.

    Tolerates a missing signature (some builtins and C callables have none) by reporting
    "not a factory", which is the behavior-preserving answer: such a target is used directly.
    """
    try:
        annotation = inspect.signature(target).return_annotation
    except (TypeError, ValueError):
        return False
    return _return_kind(annotation) == "node"


def _check_factory_node(spec: _NodeSpec, node: Node) -> None:
    """Assert a factory-built Node agrees with the spec that declared it.

    ``materialize`` adopts a factory's Node as-is, because the factory OWNS it -- that is the
    whole point for externally driven nodes, where the instance keeps the reference so it can
    spin the node later via ``Node.callback``. The cost is that every field ``materialize``
    would otherwise pass to the ``Node`` constructor is dropped on the floor, silently. The
    SPEC is what gets serialized by ``to_dict``, drawn by the editor, and read back by
    ``from_dict``, so a disagreement is not cosmetic:

    * ``node_type`` -- the scheduler keys off ``Node.node_type``, not the spec. A spec saying
      ``callback`` over a default ``UNLOCKED`` Node fails twice at once: it earns a worker
      thread which then BUSY-SPINS (``node()`` forced ``inputs=[]`` for callback nodes, and an
      empty ``InputStage`` reports ready immediately and never parks), while the ``callback()``
      the graph was written around is refused by ``Node._blocked``.
    * ``warmup`` -- load-bearing, not decorative. A ``KeepLatest`` consumer is un-ready until
      something is published and ``InputStage.ready`` is an ``all``, so a dropped warmup hangs
      that consumer for good (see :attr:`Node.warmup`).
    * ``pre_open`` / ``post_close`` -- dropping ``post_close`` silently skips shutdown work that
      can be safety-relevant (Spot's stow + sit + power-off runs there).

    Rejects rather than coerces. The factory already built the stages from its Node's own
    signature, so rewriting ``node_type`` afterwards would quietly move ownership of the loop at
    the last moment; ``warmup`` and the hooks cannot be retrofitted after construction at all.
    This runs once at build time, so failing loudly is cheap and points at the fix -- matching
    how :meth:`GraphBuilder._resolve_target` already rejects ``kwargs`` on a factory node.
    """
    declared = NodeType(spec.node_type)
    if node.node_type is not declared:
        raise ValueError(
            f"node {spec.id!r}: factory target {spec.target!r} returned a {node.node_type} Node "
            f"but the graph declares node_type={declared}; construct the Node with "
            f"node_type={declared} or correct the spec"
        )
    ignored = [name for name in ("warmup", "pre_open", "post_close") if getattr(spec, name)]
    if ignored:
        raise ValueError(
            f"node {spec.id!r}: {', '.join(ignored)} cannot be applied to factory target "
            f"{spec.target!r} -- the factory owns the Node it returns; pass them where that "
            f"Node is constructed"
        )


def _return_kind(annotation: Any) -> str:
    """Classify a callable's return annotation to drive port inference.

    Returns one of: ``"none"`` (``-> None``), ``"node"`` (a factory returning a
    :class:`Node`), ``"variadic_tuple"`` (``tuple`` / ``tuple[X, ...]`` of
    unknown length), ``"fixed_tuple"`` (``tuple[A, B]``), or ``"other"`` (a
    single typed return or a missing annotation). Handles both evaluated
    annotations and the string form produced by
    ``from __future__ import annotations``.
    """
    if annotation is None:
        return "none"
    if annotation is inspect.Signature.empty:
        return "other"
    if isinstance(annotation, str):
        head = annotation.split("|")[0].strip().strip("\"'")
        base = head.split("[", 1)[0].strip().rsplit(".", 1)[-1]
        if base in ("None", "NoneType"):
            return "none"
        if base == "Node":
            return "node"
        if base in ("tuple", "Tuple"):
            if "[" not in head:
                return "variadic_tuple"
            inner = head[head.find("[") + 1 : head.rfind("]")].strip()
            if not inner or inner.replace(" ", "").endswith(",...") or inner.endswith("..."):
                return "variadic_tuple"
            return "fixed_tuple"
        return "other"
    if isinstance(annotation, type) and issubclass(annotation, Node):
        return "node"
    if annotation is tuple:
        return "variadic_tuple"
    if getattr(annotation, "__origin__", None) is tuple:
        args = getattr(annotation, "__args__", ())
        if not args or args[-1] is Ellipsis:
            return "variadic_tuple"
        return "fixed_tuple"
    return "other"


def _infer_outputs(target: Callable[..., Any], sig: inspect.Signature) -> list[str]:
    """Derive output port names for a callable target.

    Priority: a :func:`judo.unlocked.node_outputs` decorator (names live in
    ``__node_outputs__``) always wins. Otherwise the return annotation drives it:

      * ``-> None`` → no outputs;
      * ``-> tuple[A, B, ...]`` (fixed length) → one port per element named
        ``<element_type>_<index>`` (e.g. ``tuple[SpotState, np.ndarray]`` →
        ``spot_state_0``, ``ndarray_1``); the index keeps repeated types unique.
        An element with no resolvable type name falls back to just its index
        (``"0"``, ``"1"`` …). Decorate with :func:`judo.unlocked.node_outputs`
        to give the elements meaningful names instead;
      * ``-> tuple`` / ``-> tuple[A, ...]`` (unknown length) → raises, because
        the number of output ports cannot be inferred; decorate the target with
        ``@node_outputs(...)`` or pass an explicit ``outputs=``;
      * any other typed return → a single port named after the type itself
        (e.g. ``-> SpotState`` → ``spot_state``), so single-output producers are
        self-documenting and need a decorator only to override that name. A
        union (``A | B``) names the port after its first member;
      * a missing annotation → a single ``out0`` port.
    """
    declared = getattr(target, NODE_OUTPUTS_ATTR, None)
    if declared is not None:
        return list(declared)
    return_annotation = sig.return_annotation
    kind = _return_kind(return_annotation)
    if kind == "none":
        return []
    if return_annotation is inspect.Signature.empty:
        return ["out0"]
    if kind == "variadic_tuple":
        raise ValueError(
            f"node target {_callable_to_target(target)!r} returns a "
            f"variable-length tuple ({return_annotation!r}); the number of "
            f"output ports cannot be inferred. Decorate it with "
            f"@node_outputs(...) to name each output, pass an explicit "
            f"outputs=, or annotate a fixed-length tuple[T0, T1, ...]."
        )
    origin = getattr(return_annotation, "__origin__", None)
    if return_annotation is tuple or origin is tuple:
        args = getattr(return_annotation, "__args__", ())
        return [_tuple_element_port_name(a, i) for i, a in enumerate(args)]
    return [_type_port_name(return_annotation) or "out0"]


def _tuple_element_port_name(annotation: Any, index: int) -> str:
    """Name one tuple element ``<element_type>_<index>``, or just the index.

    The trailing index keeps repeated element types (``tuple[ndarray, ndarray]``)
    unique and stable; an element whose type has no resolvable name degrades to
    the bare index.
    """
    name = _type_port_name(annotation)
    return f"{name}_{index}" if name else str(index)


def _type_port_name(annotation: Any) -> str | None:
    """Convert a single (non-tuple) annotation into a port name, else ``None``.

    Uses the type's own name, snake-cased to match port conventions. Handles
    ``X | None`` / ``Optional[X]`` and unions (the first non-``None`` member is
    used) and string annotations from ``from __future__ import annotations``.
    Returns ``None`` when no type name can be resolved (e.g. ``Any``).
    """
    if annotation is Any:
        return None
    args = getattr(annotation, "__args__", None)
    if args:
        # Optional[X] / X | None / a union: use the first non-None member.
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _type_port_name(non_none[0])
    name: str | None
    if isinstance(annotation, str):
        # PEP 563 string annotation: keep the leftmost union member, strip any
        # module/quote decoration down to the bare type name.
        text = annotation.split("|")[0].strip().strip("\"'")
        name = text.split("[")[0].rsplit(".", 1)[-1] or None
    elif isinstance(annotation, type):
        name = annotation.__name__
    else:
        name = getattr(annotation, "__name__", None)
    if not name:
        return None
    return _to_snake_case(name)


def _to_snake_case(name: str) -> str:
    """Convert a ``CamelCase``/``PascalCase`` type name to ``snake_case``."""
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (not name[i - 1].isupper() or (i + 1 < len(name) and name[i + 1].islower())):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _import_target(target: str) -> Callable[..., Any]:
    mod_name, qualname = target.split(":")
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _needs_instantiation(obj: Any) -> bool:
    """True if ``obj`` contains a Hydra ``_target_`` node anywhere (recursively through dict/list)."""
    if isinstance(obj, dict):
        return "_target_" in obj or any(_needs_instantiation(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_needs_instantiation(v) for v in obj)
    return False


def _contains_instance_ref(obj: Any) -> bool:
    """Detect the reserved marker, including unsupported nested placements."""
    if isinstance(obj, dict):
        return "_instance_ref_" in obj or any(_contains_instance_ref(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_instance_ref(v) for v in obj)
    return False


def _instance_refs(kwargs: dict[str, Any], source: str) -> dict[str, str]:
    """Extract direct kwarg references, rejecting malformed or nested markers."""
    refs: dict[str, str] = {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and "_instance_ref_" in value:
            instance_id = value["_instance_ref_"]
            if set(value) != {"_instance_ref_"} or not isinstance(instance_id, str) or not instance_id:
                raise ValueError(f"{source} kwarg {key!r}: expected {{'_instance_ref_': '<nonempty id>'}}")
            refs[key] = instance_id
        elif _contains_instance_ref(value):
            raise ValueError(f"{source} kwarg {key!r}: instance refs must be direct constructor kwargs, not nested")
    return refs


def _resolve_kwargs(kwargs: dict[str, Any], instances: dict[str, Any]) -> dict[str, Any]:
    """Keep live objects out of Hydra/OmegaConf, which can convert/copy dataclasses."""
    refs = _instance_refs(kwargs, "constructor")
    if not refs:
        return _instantiate_kwargs(kwargs)
    resolved = _instantiate_kwargs({k: v for k, v in kwargs.items() if k not in refs})
    resolved.update({key: instances[instance_id] for key, instance_id in refs.items()})
    return resolved


def _instantiate_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve instance construction kwargs, delegating nested ``_target_`` sub-configs to Hydra.

    Backward-compatible by construction: kwargs made purely of JSON primitives (every instance before
    the Hydra migration) contain no ``_target_`` and are returned unchanged -- no Hydra call, no
    OmegaConf interpolation, zero behavior change. Kwargs carrying a nested ``{_target_: ...}``
    sub-config (the declarative instances, e.g. ``spot_control``'s ``config`` + ``robot_provider``)
    are built by ``hydra.utils.instantiate``, which recursively constructs each sub-object and honors
    ``_partial_: true`` (turning the ``robot_provider`` sub-config into the zero-arg
    ``functools.partial`` the node calls). ``_convert_="all"`` returns plain Python containers for any
    non-target leaves so the wrapped factory receives ordinary dict / list values.
    """
    if not _needs_instantiation(kwargs):
        return dict(kwargs)
    from hydra.utils import instantiate  # lazy: only the declarative path pulls Hydra in

    return instantiate(kwargs, _convert_="all")


def _callable_to_target(target: str | Callable[..., Any]) -> str:
    """Normalize a target to its ``"module:qualname"`` string form.

    Strings pass through unchanged; real callables (classes, functions, or
    unbound methods like ``Foxglove.spot_state_subscriber``) are converted via
    their ``__module__``/``__qualname__`` so authoring code can pass real
    references (navigable in an IDE) while the stored spec stays a plain string.
    """
    if isinstance(target, str):
        return target
    try:
        return f"{target.__module__}:{target.__qualname__}"
    except AttributeError as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"target must be a 'module:qualname' string or a callable with " f"__module__/__qualname__, got {target!r}"
        ) from exc
