# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import dataclasses
import inspect
import typing

from judo.app.benchmark import BenchmarkerNode
from judo.app.graph import build, build_benchmark
from judo.app.nodes.controller_node import ControllerNode
from judo.app.nodes.simulation_node import SimulationNode
from judo.app.nodes.visualization_node import VisualizationNode
from judo.app.structs import GuiState


def _loop(sim: SimulationNode, controller: ControllerNode, steps: int, gui_state: GuiState | None = None) -> tuple:
    """Run the sim <-> controller loop in-process, delivering ``gui_state`` on the first step."""
    controls = None
    out = None
    for i in range(steps):
        state = gui_state if i == 0 else None
        states, _, _ = sim.step(i, controls, gui_state=state)
        controls, traces, plan_time, _ = controller.step(i, states, gui_state=state)
        out = (controls, traces, plan_time)
    assert out is not None
    return out


def test_sim_controller_loop_and_gui_state() -> None:
    """The nodes run closed-loop and apply what changed in each GUI state."""
    sim = SimulationNode(init_task="cylinder_push")
    controller = ControllerNode(init_task="cylinder_push", init_optimizer="cem")
    sim.setup()
    controller.setup()

    _, _, plan_time = _loop(sim, controller, steps=3)
    assert plan_time is not None

    cfg = dataclasses.asdict(controller.controller.controller_cfg)
    cfg["control_freq"] = 50.0
    state = GuiState(task="cylinder_push", optimizer="ps", controller_config=cfg, revisions={"controller_config": 1})
    _loop(sim, controller, steps=1, gui_state=state)
    assert controller.controller.controller_cfg.control_freq == 50.0
    assert isinstance(controller.controller.optimizer, controller.controller.available_optimizers["ps"][0])

    # a config whose revision did not move is not reapplied
    controller.controller.controller_cfg.control_freq = 20.0
    _loop(sim, controller, steps=1, gui_state=state)
    assert controller.controller.controller_cfg.control_freq == 20.0

    # a task switch is applied before the (still old-task) configs, which no longer fit and are skipped
    task_cfg = dataclasses.asdict(controller.controller.task_config)
    state = GuiState(task="cartpole", optimizer="ps", task_config=task_cfg, revisions={"task_config": 1})
    _loop(sim, controller, steps=1, gui_state=state)
    assert sim.sim.task.name == controller.controller.task.name == "cartpole"
    assert isinstance(controller.controller.optimizer, controller.controller.available_optimizers["ps"][0])

    # pausing stops planning but keeps publishing controls
    paused = GuiState(task="cartpole", optimizer="ps", sim_paused=True)
    _, _, plan_time = _loop(sim, controller, steps=1, gui_state=paused)
    assert plan_time is None and sim.sim.paused
    _, _, plan_time = _loop(sim, controller, steps=1, gui_state=GuiState(task="cartpole", optimizer="ps"))
    assert plan_time is not None and not sim.sim.paused


def test_gui_state_reset_applies_once_per_click() -> None:
    """A reset is applied when its revision moves, not on every GUI state carrying it."""
    sim = SimulationNode(init_task="cylinder_push")
    sim.setup()
    resets = []
    sim.sim.task.reset = lambda: resets.append(1)
    sim.step(0, None, gui_state=GuiState(task="cylinder_push", optimizer="cem", revisions={"task_reset": 1}))
    sim.step(1, None, gui_state=GuiState(task="cylinder_push", optimizer="cem", revisions={"task_reset": 1}))
    assert len(resets) == 1
    sim.step(2, None, gui_state=GuiState(task="cylinder_push", optimizer="cem", revisions={"task_reset": 3}))
    assert len(resets) == 2


def test_graph_wiring() -> None:
    """Both graphs build; the GUI state reaches the sim and controller on one port."""
    gui = {(e["from"], e["to"]) for e in build().to_dict()["edges"]}
    bench = {(e["from"], e["to"]) for e in build_benchmark().to_dict()["edges"]}
    for edges, source in ((gui, "gui_events.gui_state"), (bench, "benchmarker_step.gui_state")):
        assert (source, "controller_step.gui_state") in edges
        assert (source, "sim_step.gui_state") in edges
        assert ("sim_step.states", "controller_step.states") in edges
        assert ("controller_step.controls", "sim_step.controls") in edges
        assert ("sim_clock_tick.tick", "sim_step.tick") in edges
        assert ("sim_step.timestep", "sim_clock_period.new_period_s") in edges
        assert ("controller_clock_tick.tick", "controller_step.tick") in edges
        assert ("controller_step.period", "controller_clock_period.new_period_s") in edges


def test_single_output_annotations() -> None:
    """UnLocked derives output channels from the return annotation; both GUI state producers have one."""
    for fn in (VisualizationNode._state_output, BenchmarkerNode.step):
        assert typing.get_origin(inspect.signature(fn).return_annotation) is not tuple


def test_controller_publishes_period_on_change() -> None:
    """The controller reports 1 / control_freq on the first step and after a config change, so its clock can retune."""
    sim = SimulationNode(init_task="cylinder_push")
    controller = ControllerNode(init_task="cylinder_push", init_optimizer="cem")
    sim.setup()
    controller.setup()
    states = sim.step(0, None)[0]
    assert controller.step(0, states)[3] == 1 / controller.controller.controller_cfg.control_freq
    assert controller.step(1, states)[3] is None
    cfg = dataclasses.asdict(controller.controller.controller_cfg)
    cfg["control_freq"] = 50.0
    state = GuiState(task="cylinder_push", optimizer="cem", controller_config=cfg, revisions={"controller_config": 1})
    assert controller.step(2, states, gui_state=state)[3] == 1 / 50.0


def test_sim_publishes_timestep_on_change() -> None:
    """The sim reports its timestep on the first step and after a task switch, so the clock can retune."""
    sim = SimulationNode(init_task="cylinder_push")
    sim.setup()
    assert sim.step(0, None)[2] == sim.sim.timestep
    assert sim.step(1, None)[2] is None
    assert sim.step(2, None, GuiState(task="cartpole", optimizer="cem"))[2] == sim.sim.timestep == 0.04
