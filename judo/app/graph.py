# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""unLocked graphs for the judo app.

Each judo node is a plain object whose ``step`` method is the unLocked node. The three nodes live in separate
processes and communicate through shared-memory queues. Run with ``judo`` / ``benchmark``, or open the graph in the
browser editor with ``judo-graph-ui``.

GUI settings (task, optimizer, reset, pause and configs) are published as one ``GuiState`` snapshot by the
``gui_events`` CALLBACK node, driven directly by the Viser GUI handlers whenever a setting changes.
"""

from typing import Any

from judo.app.benchmark import BenchmarkerNode
from judo.app.nodes.controller_node import ControllerNode
from judo.app.nodes.simulation_node import SimulationNode
from judo.app.nodes.visualization_node import VisualizationNode
from judo.unlocked.graph.builder import GraphBuilder
from judo.unlocked.node import NodeType
from judo.unlocked.nodes.periodic_timer import PeriodicTimer


def _add_simulation_and_controller(
    g: GraphBuilder,
    task: str,
    optimizer: str,
    custom_tasks: dict | None,
    custom_optimizers: dict | None,
    gui_state_source: str,
    controller_states_policy: str = "keep_latest",
) -> None:
    """Add the simulation and controller, driven by the ``GuiState`` published on ``gui_state_source``."""
    sim = g.instance(
        "simulation", SimulationNode, process="simulation", init_task=task, task_registration_cfg=custom_tasks
    )
    sim.node(
        "sim_step",
        SimulationNode.step,
        outputs=["states", "render_pose", "timestep"],
        pre_open=SimulationNode.setup,
    )

    # Clock pacing the sim at its model timestep. The initial period is a placeholder: sim_step publishes the real
    # timestep on its first step (and on every task switch), which retunes the period.
    clock = g.instance("sim_clock", PeriodicTimer, process="simulation", period_s=0.01)
    clock.node("sim_clock_tick", PeriodicTimer.__call__, outputs=["tick"])
    clock.node("sim_clock_period", PeriodicTimer.update_period_s, outputs=[])
    g.connect("sim_clock_tick.tick", "sim_step.tick", policy="lossless")
    g.connect("sim_step.timestep", "sim_clock_period.new_period_s", policy="lossless")

    controller = g.instance(
        "controller",
        ControllerNode,
        process="controller",
        init_task=task,
        init_optimizer=optimizer,
        task_registration_cfg=custom_tasks,
        optimizer_registration_cfg=custom_optimizers,
    )
    controller.node(
        "controller_step",
        ControllerNode.step,
        outputs=["controls", "traces", "plan_time", "period"],
        pre_open=ControllerNode.setup,
    )

    # Clock pacing the controller at control_freq, retuned by controller_step like the sim clock. ``latest`` drops
    # ticks missed while planning overran: replanning back to back to catch up would be pointless.
    ctrl_clock = g.instance("controller_clock", PeriodicTimer, process="controller", period_s=0.01)
    ctrl_clock.node("controller_clock_tick", PeriodicTimer.__call__, outputs=["tick"])
    ctrl_clock.node("controller_clock_period", PeriodicTimer.update_period_s, outputs=[])
    g.connect("controller_clock_tick.tick", "controller_step.tick", policy="latest")
    g.connect("controller_step.period", "controller_clock_period.new_period_s", policy="lossless")

    # Closed loop: the optional/keep_latest policies never block, so the cycle needs no warmup.
    g.connect("sim_step.states", "controller_step.states", policy=controller_states_policy)
    g.connect("controller_step.controls", "sim_step.controls", policy="optional")
    # Never blocks a step (None when the GUI has not changed). ``optional`` keeps only the newest snapshot, which
    # supersedes the older ones.
    g.connect(gui_state_source, "controller_step.gui_state", policy="optional")
    g.connect(gui_state_source, "sim_step.gui_state", policy="optional")


def _set_shutdown_tiers(g: GraphBuilder, events_process: str) -> None:
    """Stop consumers before the producers they read from (tier 0 first).

    A process unlinks its output queues' shared memory when it exits, so a consumer still running could fail to attach
    to a frame its producer just wrote. sim and controller feed each other, so that cycle is broken at the sim, which
    stops first; the events producer (GUI or benchmarker) feeds both and stops last.
    """
    g.process("simulation", tier=0)
    g.process("controller", tier=1)
    g.process(events_process, tier=2)


def build(
    task: str = "cylinder_push",
    optimizer: str = "cem",
    custom_tasks: dict | None = None,
    custom_optimizers: dict | None = None,
    controller_config_overrides: dict | None = None,
    optimizer_config_overrides: dict | None = None,
    sim_pause_button: bool = True,
    **_: Any,
) -> GraphBuilder:
    """The default judo graph: simulation, controller and the Viser GUI."""
    g = GraphBuilder("judo")
    viz = g.instance(
        "visualization",
        VisualizationNode,
        process="visualization",
        init_task=task,
        init_optimizer=optimizer,
        task_registration_cfg=custom_tasks,
        optimizer_registration_cfg=custom_optimizers,
        controller_override_cfg=controller_config_overrides,
        optimizer_override_cfg=optimizer_config_overrides,
        sim_pause_button=sim_pause_button,
    )
    viz.node(
        "gui_step",
        VisualizationNode.step,
        outputs=[],
        pre_open=VisualizationNode.setup,
        post_close=VisualizationNode.cleanup,
    )
    viz.node("gui_events", VisualizationNode.events_node, outputs=["gui_state"], node_type=NodeType.CALLBACK)
    _add_simulation_and_controller(
        g, task, optimizer, custom_tasks, custom_optimizers, gui_state_source="gui_events.gui_state"
    )

    g.connect("sim_step.render_pose", "gui_step.render_pose", policy="latest")
    g.connect("controller_step.traces", "gui_step.traces", policy="optional")
    g.connect("controller_step.plan_time", "gui_step.plan_time", policy="optional")
    _set_shutdown_tiers(g, events_process="visualization")
    return g


def build_benchmark(
    task: str = "cylinder_push",
    optimizer: str = "cem",
    custom_tasks: dict | None = None,
    custom_optimizers: dict | None = None,
    num_samples: int = 100,
    **_: Any,
) -> GraphBuilder:
    """Benchmark graph: a benchmarker node drives the task/optimizer switches and records plan times."""
    g = GraphBuilder("judo_benchmark")
    bench = g.instance(
        "benchmarker",
        BenchmarkerNode,
        process="benchmarker",
        num_samples=num_samples,
    )
    bench.node("benchmarker_step", BenchmarkerNode.step, outputs=["gui_state"], pre_open=BenchmarkerNode.setup)
    _add_simulation_and_controller(
        g,
        task,
        optimizer,
        custom_tasks,
        custom_optimizers,
        gui_state_source="benchmarker_step.gui_state",
    )

    g.connect("controller_step.plan_time", "benchmarker_step.plan_time", policy="lossless")
    _set_shutdown_tiers(g, events_process="benchmarker")
    return g
