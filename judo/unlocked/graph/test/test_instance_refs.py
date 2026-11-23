# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Shared constructor dependencies across authoring, serialization, Hydra and forked execution."""

# Tests deliberately inspect the materialized target and declarative validation boundary.
# ruff: noqa: SLF001

import collections
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any
from unittest.mock import patch

import pytest

from judo.unlocked.graph.builder import GraphBuilder
from judo.unlocked.graph.runner import run_builder
from judo.unlocked.graph.ui import server
from judo.unlocked.node import Node, NodeStop

_CREATED: list["Shared"] = []


@dataclass
class Shared:
    """A mutable dataclass: passing this through Hydra would destroy identity."""

    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _CREATED.append(self)


@dataclass
class Consumer:
    simulation: Shared
    alias: Shared
    config: collections.Counter
    status_path: str = ""

    def __post_init__(self) -> None:
        assert self.simulation is self.alias is _CREATED[-1]
        assert isinstance(self.config, collections.Counter)
        assert self.config["count"] == 2

    def step(self) -> None:
        assert self.simulation is self.alias is _CREATED[-1]
        self.simulation.events.append("step")
        if self.status_path:
            Path(self.status_path).write_text(str(os.getpid()))
        raise NodeStop()


def make_node(simulation: Shared, alias: Shared, config: collections.Counter, status_path: str = "") -> Node:
    consumer = Consumer(simulation, alias, config, status_path)
    return Node("reference_factory", consumer.step)


def identity(simulation: Shared) -> Shared:
    return simulation


def _consumer(node: Node) -> Consumer:
    assert isinstance(node._target, MethodType)
    obj = node._target.__self__
    assert isinstance(obj, Consumer)
    return obj


def _graph(tmp_path: Path | None = None) -> GraphBuilder:
    g = GraphBuilder("references")
    sim = g.instance("sim", Shared, process="simulation")
    kwargs: dict[str, Any] = {
        "simulation": sim.ref(),
        "alias": sim.ref(),
        "config": {"_target_": "collections.Counter", "count": 2},
    }
    consumer = g.instance(
        "consumer",
        Consumer,
        process="simulation",
        status_path=str(tmp_path / "instance") if tmp_path else "",
        **kwargs,
    )
    consumer.node("step", Consumer.step)
    g.node(
        "factory",
        make_node,
        process="simulation",
        inputs=[],
        outputs=[],
        status_path=str(tmp_path / "factory") if tmp_path else "",
        **kwargs,
    )
    return g


@pytest.mark.parametrize("encoding", ["dict", "json", "yaml"])
@pytest.mark.parametrize("reverse", [False, True])
def test_round_trip_identity_and_dependency_order(encoding: str, reverse: bool) -> None:
    g = _graph()
    spec = g.to_dict()
    if reverse:
        spec["instances"].reverse()  # JSON forward references are legal.
    g = GraphBuilder.from_dict(spec)
    if encoding == "json":
        g = GraphBuilder.from_dict(json.loads(g.to_json()))
    elif encoding == "yaml":
        g = GraphBuilder.from_yaml(g.to_yaml())
    before = g.to_json()
    _CREATED.clear()
    nodes = g.materialize()
    first = _consumer(nodes["step"])
    second = _consumer(nodes["factory"])
    assert first.simulation is second.simulation is _CREATED[0]
    assert len(_CREATED) == 1
    first.simulation.events.append("shared mutation")
    assert second.simulation.events == ["shared mutation"]
    assert g.to_json() == before  # No live objects leaked into the description.
    assert g.to_dict() == spec


def test_handle_ref_is_fresh_json_data_and_owner_is_persisted() -> None:
    g = GraphBuilder()
    sim = g.instance("sim", Shared)
    assert sim.ref() == {"_instance_ref_": "sim"}
    assert json.loads(json.dumps(sim.ref())) == sim.ref()
    changed = sim.ref()
    changed["_instance_ref_"] = "other"
    assert sim.ref() == {"_instance_ref_": "sim"}
    assert g.to_dict()["instances"][0]["process"] == "sim_process"


def test_transitive_dependency_chain_and_refs_without_hydra() -> None:
    g = _graph()
    g.instance("alias", identity, process="simulation", simulation={"_instance_ref_": "sim"})
    spec = g.to_dict()
    spec["instances"][1]["kwargs"]["simulation"] = {"_instance_ref_": "alias"}
    spec["instances"].reverse()
    _CREATED.clear()
    g = GraphBuilder.from_dict(spec)
    assert g._instance_order() == ["sim", "alias", "consumer"]
    nodes = g.materialize()
    assert _consumer(nodes["step"]).simulation is _consumer(nodes["factory"]).simulation
    assert len(_CREATED) == 1


def test_dependency_only_owner_survives_serialized_node_filtering() -> None:
    spec = _graph().to_dict()
    # A subset that retains only the dependent factory still needs sim even
    # though no node has instance='sim'. There is no core subset runner today.
    spec["nodes"] = [n for n in spec["nodes"] if n["id"] == "factory"]
    spec["instances"] = [i for i in spec["instances"] if i["id"] == "sim"]
    g = GraphBuilder.from_dict(json.loads(json.dumps(spec)))
    assert g.to_dict()["instances"][0]["process"] == "simulation"
    nodes = g.materialize()
    assert _consumer(nodes["factory"]).simulation is _CREATED[-1]
    # Moving the subset must not reinterpret sim's owner and silently clone it.
    spec["nodes"][0]["process"] = "elsewhere"
    with pytest.raises(ValueError, match="cross-process instance reference"):
        GraphBuilder.from_dict(spec).materialize()


@pytest.mark.parametrize("source", ["instance", "node"])
def test_missing_reference_fails_before_any_constructor(source: str) -> None:
    spec = _graph().to_dict()
    target = spec["instances"][1] if source == "instance" else spec["nodes"][1]
    target["kwargs"]["simulation"] = {"_instance_ref_": "missing"}
    with patch("judo.unlocked.graph.builder._import_target", side_effect=AssertionError("must not import")):
        with pytest.raises(ValueError, match=f"{source} .*unknown instance reference 'missing'"):
            GraphBuilder.from_dict(spec).materialize()


@pytest.mark.parametrize("instance_id", ["missing", ""])
@pytest.mark.parametrize("entrypoint", ["materialize", "build_processes", "dry_run"])
def test_missing_method_instance_fails_without_imports(instance_id: str, entrypoint: str) -> None:
    spec = {
        "instances": [{"id": "declared", "factory": "unavailable:Factory"}],
        "nodes": [{"id": "step", "target": "unavailable:Factory.step", "instance": instance_id}],
        "edges": [],
    }
    with patch("judo.unlocked.graph.builder._import_target", side_effect=AssertionError("must not import")):
        g = GraphBuilder.from_dict(spec)
        with pytest.raises(ValueError, match="node 'step': unknown instance"):
            if entrypoint == "dry_run":
                run_builder(g, dry_run=True)
            else:
                getattr(g, entrypoint)()


@pytest.mark.parametrize("self_cycle", [False, True])
def test_cycles_fail_before_any_constructor(self_cycle: bool) -> None:
    spec = _graph().to_dict()
    spec["instances"][0]["kwargs"] = {"dependency": {"_instance_ref_": "sim" if self_cycle else "consumer"}}
    with patch("judo.unlocked.graph.builder._import_target", side_effect=AssertionError("must not import")):
        with pytest.raises(ValueError, match="instance reference cycle: sim -> .*sim"):
            GraphBuilder.from_dict(spec).materialize()


@pytest.mark.parametrize("location", ["dependency", "consumer", "method", "factory"])
@pytest.mark.parametrize("entrypoint", ["materialize", "build_processes", "dry_run"])
def test_cross_process_rejected(location: str, entrypoint: str) -> None:
    spec = _graph().to_dict()
    target = {
        "dependency": spec["instances"][0],
        "consumer": spec["instances"][1],
        "method": spec["nodes"][0],
        "factory": spec["nodes"][1],
    }[location]
    target["process"] = "elsewhere"
    g = GraphBuilder.from_dict(spec)
    with patch("judo.unlocked.graph.builder._import_target", side_effect=AssertionError("must not import")):
        with pytest.raises(ValueError, match="cross-process instance reference"):
            if entrypoint == "dry_run":
                run_builder(g, dry_run=True)
            else:
                getattr(g, entrypoint)()


def test_default_processes_are_not_implicitly_colocated() -> None:
    g = GraphBuilder()
    sim = g.instance("sim", Shared)
    g.instance("consumer", Consumer, simulation=sim.ref())
    with pytest.raises(ValueError, match="cross-process instance reference"):
        g.materialize()


def test_referenced_instance_method_in_another_process_is_rejected() -> None:
    spec = _graph().to_dict()
    spec["nodes"].append({"id": "other", "target": "unused:Shared.step", "instance": "sim", "process": "other"})
    with pytest.raises(ValueError, match="instance 'sim': cross-process"):
        GraphBuilder.from_dict(spec).materialize()


@pytest.mark.parametrize(
    "value, message",
    [
        ({"_instance_ref_": 1}, "expected"),
        ({"_instance_ref_": ""}, "expected"),
        ({"_instance_ref_": "sim", "extra": 1}, "expected"),
        ([{"_instance_ref_": "sim"}], "not nested"),
        ({"_target_": "collections.Counter", "sim": {"_instance_ref_": "sim"}}, "not nested"),
    ],
)
@pytest.mark.parametrize("source", ["instance", "node"])
def test_malformed_and_nested_refs_are_rejected(value: Any, message: str, source: str) -> None:
    spec = _graph().to_dict()
    target = spec["instances"][1] if source == "instance" else spec["nodes"][1]
    target["kwargs"]["simulation"] = value
    with pytest.raises(ValueError, match=message):
        GraphBuilder.from_dict(spec).materialize()


def test_method_kwargs_cannot_silently_ignore_a_reference() -> None:
    spec = _graph().to_dict()
    spec["nodes"][0]["kwargs"] = {"simulation": {"_instance_ref_": "sim"}}
    with pytest.raises(ValueError, match="not a method-bound node"):
        GraphBuilder.from_dict(spec).materialize()


def test_legacy_owner_inferred_only_when_unambiguous() -> None:
    spec = _graph().to_dict()
    del spec["instances"][1]["process"]  # consumer has a method with a process tag
    GraphBuilder.from_dict(spec).materialize()
    del spec["instances"][0]["process"]  # dependency-only sim has no recoverable owner
    with pytest.raises(ValueError, match="no process owner"):
        GraphBuilder.from_dict(spec).materialize()


def test_no_refs_keep_legacy_materialization_order_and_process_behavior() -> None:
    spec = {
        "instances": [
            {"id": "z", "factory": f"{__name__}:Shared"},
            {"id": "a", "factory": f"{__name__}:Shared"},
        ],
        "nodes": [],
        "edges": [],
    }
    g = GraphBuilder.from_dict(spec)
    assert g._instance_order() == ["z", "a"]
    _CREATED.clear()
    assert g.materialize() == {}
    assert len(_CREATED) == 2


def test_serialized_execution_script_runs_real_scheduler(tmp_path: Path) -> None:
    """Generated Python uses the actual JSON runner (there is no separate codegen API).

    The child imports factories from this test module, hydrates configs, injects
    the same dataclass into both consumers, then forks the real scheduler. Both
    runtime targets check identity again and report the scheduler's PID.
    """
    g = _graph(tmp_path)
    spec = g.to_dict()
    spec["instances"].reverse()
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(spec))
    script = tmp_path / "generated.py"
    script.write_text(
        # Bazel's pytest launcher adds import roots for test packages; give the
        # child the same import environment for these importable toy factories.
        f"import sys\nsys.path[:] = {sys.path!r}\n"
        "from judo.unlocked.graph.runner import run_serialized_graph\n"
        f"raise SystemExit(run_serialized_graph(['--graph-json', {str(graph_path)!r}, '--no-shm-cleanup']))\n"
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "instance").read_text() == (tmp_path / "factory").read_text()
    assert int((tmp_path / "instance").read_text()) != os.getpid()


def test_ui_schema_and_rebuild_preserve_builder_owned_references() -> None:
    current = {"simulation": {"_instance_ref_": "new_sim"}, "gain": 2}
    # Even a custom schema that omits the dependency must expose it read-only.
    with patch.object(server, "_resolve_instance_fields", return_value=[{"name": "gain", "kind": "int"}]):
        fields = server._instance_schema("unused:Factory", current)
    ref_field = next(f for f in fields if f["name"] == "simulation")
    assert ref_field["kind"] == "json"
    assert ref_field["app_level"] is True
    assert ref_field["value"] == current["simulation"]
    new_spec = {"instances": [{"id": "consumer", "factory": "unused:Factory", "kwargs": current}]}
    old = {
        "instances": [
            {
                "id": "consumer",
                "factory": "unused:Factory",
                "kwargs": {
                    "simulation": {"_instance_ref_": "old_sim"},
                    "gain": 3,
                },
            }
        ]
    }
    with patch.object(server, "_instance_schema", return_value=[]):
        server._merge_instance_edits(new_spec, json.dumps(old), "{}")
    assert current == {"simulation": {"_instance_ref_": "new_sim"}, "gain": 3}


def test_rebuild_restores_only_user_edited_kwargs() -> None:
    def spec(**kwargs: Any) -> dict[str, Any]:
        return {"instances": [{"id": "sim", "factory": "unused:Factory", "kwargs": kwargs}]}

    new_spec = spec(mocap_id=114, gain=2)
    old, built = spec(mocap_id=112, gain=3), spec(mocap_id=112, gain=2)
    with patch.object(server, "_instance_schema", return_value=[]):
        server._merge_instance_edits(new_spec, json.dumps(old), json.dumps(built))
    assert new_spec["instances"][0]["kwargs"] == {"mocap_id": 114, "gain": 3}
