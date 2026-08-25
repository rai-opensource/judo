# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Callable

import numpy as np

from judo.controller import Controller, ControllerConfig, make_controller
from judo.optimizers import Optimizer, OptimizerConfig, get_registered_optimizers
from judo.tasks.cylinder_push import CylinderPush

# ##### #
# MOCKS #
# ##### #


class MockOptimizerTrackNominalKnots(Optimizer):
    """A mock optimizer to track the history of nominal_knots."""

    def __init__(self, opt_config: OptimizerConfig, nu: int) -> None:
        """Initializes the mock optimizer."""
        super().__init__(opt_config, nu)
        self.received_knots_history: list[np.ndarray] = []

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Samples control knots and tracks the history."""
        self.received_knots_history.append(nominal_knots.copy())
        num_rollouts = self.num_rollouts
        sampled_knots = nominal_knots + np.random.randn(num_rollouts, self.config.num_nodes, self.nu)
        return sampled_knots

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Selects something."""
        return sampled_knots[0]


# ##### #
# TESTS #
# ##### #


def test_max_opt_iters(temp_np_seed: Callable) -> None:
    """Tests that max_opt_iters correctly applies multiple iterations of optimization to a solution."""

    def _setup_controller(max_opt_iters: int) -> tuple[MockOptimizerTrackNominalKnots, Controller]:
        """Helper function to set up the controller."""
        task = CylinderPush()
        ps_config = OptimizerConfig()
        opt = MockOptimizerTrackNominalKnots(ps_config, task.nu)
        controller = make_controller(
            init_task="cylinder_push",
            init_optimizer="cem",
        )
        controller.controller_cfg = ControllerConfig(max_opt_iters=max_opt_iters)
        controller.optimizer = opt
        return opt, controller

    # generate a solution using max_opt_iters=1
    with temp_np_seed(42):
        opt1, controller1 = _setup_controller(max_opt_iters=1)
        controller1.current_state = np.random.rand(controller1.task.model.nq + controller1.task.model.nv)
        controller1.time = 0.0
        controller1.update_action()

    # generate a solution using max_opt_iters=2
    with temp_np_seed(42):
        opt2, controller2 = _setup_controller(max_opt_iters=2)
        controller2.current_state = np.random.rand(controller2.task.model.nq + controller2.task.model.nv)
        controller2.time = 0.0
        controller2.update_action()

    # check that the initial knots match in the optimization iterations
    assert np.array_equal(opt1.received_knots_history[0], opt2.received_knots_history[0])

    # check that the final knot in opt2 is not the same as the initial knot, matches the nominal knots of controller1
    assert not np.array_equal(opt2.received_knots_history[-1], opt2.received_knots_history[0])
    assert np.array_equal(opt2.received_knots_history[-1], controller1.nominal_knots)


def test_update_action() -> None:
    """Tests the update_action method with different optimizers."""

    def _setup_controller(opt_cls: type[Optimizer], opt_cfg: OptimizerConfig) -> Controller:
        """Helper function to set up the controller."""
        task = CylinderPush()
        opt = opt_cls(opt_cfg, task.nu)
        controller = make_controller(
            init_task="cylinder_push",
            init_optimizer="cem",
        )
        controller.optimizer = opt
        return controller

    # test with all registered optimizers
    for _opt_name, (opt_cls, opt_cfg) in get_registered_optimizers().items():
        controller = _setup_controller(opt_cls, opt_cfg)
        controller.current_state = np.random.rand(controller.task.model.nq + controller.task.model.nv)
        controller.time = 0.0

        # check that update_action runs without error
        controller.update_action()

        # check that the nominal knots have the correct shape
        assert controller.nominal_knots.shape == (controller.optimizer.num_nodes, controller.optimizer.nu)

        # check that the candidate knots have the correct shape
        assert controller.candidate_knots.shape == (
            controller.optimizer.num_rollouts,
            controller.optimizer.num_nodes,
            controller.optimizer.nu,
        )
