# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any

import networkx as nx

from judo.unlocked import Node
from judo.unlocked.shmque import SharedMemoryQueueView


def make_node_graph(nodes: list[Node]) -> nx.Graph:
    """Make a networkx graph from a list of nodes."""
    result = nx.DiGraph()
    result.add_node("const", is_topic=True)
    result.add_nodes_from([node.name for node in nodes], is_topic=False)

    for node in nodes:
        for q, t in zip(node.output_stage, node.output_stage.types, strict=False):  # type: ignore[call-overload]
            result.add_node(q.name, is_topic=True)
            result.add_edge(node.name, q.name, message_type=(t.__name__ if hasattr(t, "__name__") else str(t)))

    for node in nodes:
        for qv, p in node.input_stage.all_inputs:
            policy_type = type(p)
            if "Value" in type(qv).__name__:
                result.add_edge("const", node.name, policy_type=qv.queue_name)
            else:
                result.add_edge(qv.queue_name, node.name, policy_type=policy_type.__name__)

    return result


def graph_to_mermaid(graph: nx.DiGraph, left_to_right: bool = True) -> str:
    result = f'graph {"LR" if left_to_right else "TD"}\n'

    result += "\n%% Function nodes\n"
    for node in graph.nodes:
        if node != "const" and graph.nodes[node]["is_topic"]:
            continue
        result += f"{node}\n"

    result += "\n%% Topic nodes\n"
    for node in graph.nodes:
        if node == "const" or not graph.nodes[node]["is_topic"]:
            continue
        result += f"{node}:::hidden\n"

    result += "\n%% Queues\n"
    for node in graph.nodes:
        if graph.nodes[node]["is_topic"]:
            continue
        for published_topic in graph.successors(node):
            message_type = graph.edges[node, published_topic]["message_type"]
            if "|" in message_type:
                message_type = f'"{message_type}"'
            result += f"{node} -.->|{message_type}| {published_topic}\n"

    result += "\n%% Views\n"
    for node in graph.nodes:
        if not graph.nodes[node]["is_topic"]:
            continue
        for subscriber_node in graph.successors(node):
            result += f'{node} -->|{graph.edges[node, subscriber_node]["policy_type"]}| {subscriber_node}\n'

    result += "\n%% Prettifying\n"
    result += "classDef hidden display: none;\n"

    return result


def _queue_with_policy(value: Any) -> bool:
    if not isinstance(value, tuple):
        return False
    if not isinstance(value[0], SharedMemoryQueueView):
        return False
    return True
