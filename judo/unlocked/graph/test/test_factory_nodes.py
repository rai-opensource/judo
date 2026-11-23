# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Tests for factory targets -- a graph target that RETURNS a ``Node`` instead of being wrapped in one.

``materialize`` adopts such a Node as-is so the factory keeps ownership of it; that is what lets an
instance hand its Node to viser and spin it later via ``Node.callback``. The trade-off is that every
field ``materialize`` would otherwise pass to the ``Node`` constructor -- ``node_type``, ``warmup``,
``pre_open``, ``post_close`` -- has nowhere to go. These tests pin that the mismatch is reported
instead of silently dropped, because the spec is what ``to_dict`` serializes, the editor draws, and
the scheduler is configured from.

Hermetic: needs only the builder and ``Node``, no runtime node classes and no DVC assets.
"""

from __future__ import annotations

import pytest

from judo.unlocked.graph.builder import GraphBuilder
from judo.unlocked.node import Node, NodeType

_PORTS = {"inputs": [], "outputs": ["payload"]}


def _ports_for(declared: NodeType) -> dict[str, list[str]]:
    """Port declaration a factory node of each kind requires.

    ``node()`` validates these at registration -- a callback node may declare no inputs (it is
    fed by its caller) and a generator node no outputs (it yields to its consumer) -- so the
    spec under test has to be independently valid before ``materialize`` can be reached.
    """
    if declared is NodeType.GENERATOR:
        return {"inputs": ["value"], "outputs": []}
    return {"inputs": [], "outputs": ["payload"]}


def publish(value: int) -> dict:
    """Business logic for the factory-built nodes below."""
    return {"value": value}


def make_callback_node() -> Node:
    """Factory that builds the externally driven node it will later spin itself."""
    return Node("owned", publish, node_type=NodeType.CALLBACK)


def make_unlocked_node() -> Node:
    """Factory that forgets ``node_type``, leaving the scheduler-driven default."""
    return Node("owned", publish)


def make_generator_node() -> Node:
    """Factory for the pull-driven case."""
    return Node("owned", publish, node_type=NodeType.GENERATOR)


def _graph(factory: object, **node_kwargs: object) -> GraphBuilder:
    g = GraphBuilder(name="factory_test")
    g.node("owner", factory, **node_kwargs)  # type: ignore[arg-type]
    return g


def test_factory_node_matching_its_spec_is_adopted_unchanged() -> None:
    """The happy path: the Node the factory built is the Node the graph runs, by identity.

    Adoption (not re-wrapping) is the whole point -- the factory holds the only reference the
    GUI thread will call ``callback`` on, so materializing a copy would publish into a stage
    nobody drives.
    """
    g = _graph(make_callback_node, node_type=NodeType.CALLBACK, **_PORTS)

    node = g.materialize()["owner"]

    assert isinstance(node, Node)
    assert node.node_type is NodeType.CALLBACK
    assert not node.is_scheduled


@pytest.mark.parametrize(
    ("factory", "declared"),
    [
        (make_unlocked_node, NodeType.CALLBACK),
        (make_unlocked_node, NodeType.GENERATOR),
        (make_callback_node, NodeType.GENERATOR),
        (make_generator_node, NodeType.CALLBACK),
    ],
)
def test_factory_node_contradicting_its_declared_node_type_is_rejected(factory: object, declared: NodeType) -> None:
    """A spec that disagrees with the Node it declared must fail at build time, not at runtime.

    The scheduler keys off ``Node.node_type``, not the spec, so an ``UNLOCKED`` Node behind a
    ``callback`` spec fails twice over: it is given a worker thread that busy-spins (``node()``
    forces ``inputs=[]`` for callback nodes, and an empty ``InputStage`` reports ready
    immediately and never parks), while the ``callback()`` the graph was written around is
    refused by ``Node._blocked``. Meanwhile ``to_dict`` keeps serializing the declared value, so
    the saved JSON and the editor both describe a node that does not exist.
    """
    g = _graph(factory, node_type=declared, **_ports_for(declared))

    with pytest.raises(ValueError, match="node_type"):
        g.materialize()


@pytest.mark.parametrize("field", ["warmup", "pre_open", "post_close"])
def test_constructor_only_fields_are_rejected_on_a_factory_node(field: str) -> None:
    """Fields that cannot be retrofitted onto an already-built Node must not be silently dropped.

    ``warmup`` is load-bearing rather than decorative -- a ``KeepLatest`` consumer stays
    un-ready until something is published, so a dropped warmup hangs it for good -- and a
    dropped ``post_close`` skips shutdown work that can be safety-relevant.
    """
    value = [{"value": 0}] if field == "warmup" else f"{__name__}:publish"
    g = _graph(make_callback_node, node_type=NodeType.CALLBACK, **{field: value}, **_PORTS)

    with pytest.raises(ValueError, match=field):
        g.materialize()


def test_wrapped_targets_still_receive_every_spec_field() -> None:
    """The restriction is specific to factories: an ordinary target is still fully configured.

    Guards against "fix" by amputation -- the constructor-side fields must keep working for the
    overwhelmingly common case where ``materialize`` owns the ``Node`` it builds.
    """
    g = GraphBuilder(name="wrapped_test")
    g.node("plain", publish, warmup=[{"value": 7}], node_type=NodeType.CALLBACK, outputs=["payload"])

    node = g.materialize()["plain"]

    assert node.node_type is NodeType.CALLBACK
    assert node.warmup == [{"value": 7}]


def test_declared_node_type_survives_a_serialization_round_trip() -> None:
    """The check is only meaningful if the spec it validates against round-trips intact."""
    g = _graph(make_callback_node, node_type=NodeType.CALLBACK, **_PORTS)

    restored = GraphBuilder.from_dict(g.to_dict())

    assert restored.to_dict()["nodes"][0]["node_type"] == NodeType.CALLBACK.value
    assert restored.materialize()["owner"].node_type is NodeType.CALLBACK
