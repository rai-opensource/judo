# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import contextlib
import textwrap
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import pytest

import judo.app  # noqa: F401  # registers judo's built-in tasks in the task registry


@pytest.fixture(scope="session")
def temp_np_seed() -> Callable[[int], contextlib._GeneratorContextManager[None]]:
    """Fixture to temporarily set the NumPy random seed for tests."""

    @contextlib.contextmanager
    def _temp_np_seed(seed: int) -> Generator[None, None, None]:
        """Context manager to temporarily set the NumPy random seed."""
        state = np.random.get_state()  # save state before context manager
        try:
            np.random.seed(seed)
            yield
        finally:
            np.random.set_state(state)  # restore state after context manager

    return _temp_np_seed


@pytest.fixture
def task_text_xml_path(tmp_path: Path) -> Path:
    """Write the test MuJoCo model to a temporary directory and return its path."""
    _MODEL_XML = """
    <mujoco model="test">
    <option timestep="0.02" />

    <asset>
        <texture name="blue_grid" type="2d" builtin="checker"
                rgb1=".02 .14 .44" rgb2=".27 .55 1"
                width="300" height="300" mark="edge" markrgb="1 1 1"/>
        <material name="blue_grid" texture="blue_grid" texrepeat="1 1"
                texuniform="true" reflectance=".2"/>
    </asset>

    <worldbody>
        <body>
        <geom mass="0" name="floor" pos="0 0 -0.25"
                condim="3" size="10 10 0.10" rgba="0 1 1 1"
                type="box" material="blue_grid"/>
        </body>

        <body name="body" pos="0 0 0">
        <joint name="body_x" damping="4" type="slide" axis="1 0 0"/>
        <joint name="body_y" damping="4" type="slide" axis="0 1 0"/>
        <joint name="body_z" damping="4" type="slide" axis="0 0 1"/>
        <geom name="body_geom" type="box" size="0.25 0.1 0.1"
                mass="0.1" rgba=".4 .4 .4 1" friction="0"/>
        <site pos="0 0 0.1" name="body_site"/>
        </body>
    </worldbody>

    <sensor>
        <framepos name="body_pos" objtype="site" objname="body_site"/>
    </sensor>
    </mujoco>
    """.strip()
    _MODEL_XML = textwrap.dedent(_MODEL_XML).strip()
    xml_file = tmp_path / "model.xml"
    xml_file.write_text(_MODEL_XML)
    return xml_file
