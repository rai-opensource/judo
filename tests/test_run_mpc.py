# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""End-to-end tests for run_mpc: config loading, MPC execution, data collection, and H5 output."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import h5py
    import warp as wp

    from run_mpc.mpc_batch import run_mpc_batch
    from run_mpc.mpc_config import MPCTimers, PublicMPCConfig, decode_config, load_configs_from_json_data
    from run_mpc.mpc_setup import clamp_for_mjwarp, save_results_to_h5, setup_mpc

    _has_gpu = wp.is_cuda_available()
except ImportError:
    _has_gpu = False

# Skip the entire module at collection time if warp/GPU is unavailable,
# since run_mpc imports warp at module level.
if not _has_gpu:
    pytest.skip("requires CUDA GPU (warp)", allow_module_level=True)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "run_mpc" / "configs"

# Tasks to run end-to-end: (task_name, config_file, expected_nu, is_spot)
#   expected_nu: number of task-level control dimensions (not model actuators)
#   is_spot: whether the task uses a locomotion policy (affects sim backend)
E2E_TASKS = [
    ("cartpole", "cartpole.json", 1, False),
    ("cylinder_push", "cylinder_push.json", 2, False),
    ("spot_navigate", "spot_navigate.json", 3, True),
    ("spot_box_push", "spot_box_push.json", 10, True),
    ("spot_tire_roll", "spot_tire_roll.json", 11, True),
    ("spot_tire_upright", "spot_tire_upright.json", 17, True),
]

MAX_NUM_TASK_STEPS = 10


def _load_json(config_name: str) -> dict:
    """Load and decode a JSON config file."""
    config_path = CONFIGS_DIR / config_name
    with open(config_path) as f:
        return json.load(f, object_hook=decode_config)


def _make_config(config_file: str) -> PublicMPCConfig:
    """Create a lightweight MPC config for testing."""
    return PublicMPCConfig(
        config_path=CONFIGS_DIR / config_file,
        num_trajectories=1,
        num_parallel=1,
        max_num_task_steps=MAX_NUM_TASK_STEPS,
        store_rollouts=True,
        store_viapoints=True,
        require_success=False,
    )


# Module-scoped fixture: runs MPC once per task, shared across all tests.
@pytest.fixture(scope="module", params=E2E_TASKS, ids=[t[0] for t in E2E_TASKS])
def mpc_run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Set up and run MPC for a single task, returning all artifacts for inspection."""
    task_name, config_file, expected_nu, is_spot = request.param

    json_data = _load_json(config_file)
    task, optimizer, controller_cfg = load_configs_from_json_data(json_data)
    clamp_for_mjwarp(task.model)

    config = _make_config(config_file)
    sims, batched_controllers, size_data = setup_mpc(
        config, json_data, task, optimizer, controller_cfg, config.num_parallel
    )

    timers = MPCTimers.create()
    batch_results = run_mpc_batch(sims, batched_controllers, config, size_data, timers)

    # Save to H5
    tmp_dir = tmp_path_factory.mktemp(task_name)
    output_path = tmp_dir / "trajectories.h5"
    size_data.num_trajectories = len(batch_results)
    save_results_to_h5(
        output_path,
        batch_results,
        size_data,
        config,
        CONFIGS_DIR / config_file,
        json_data,
        config.num_parallel,
    )

    return {
        "task_name": task_name,
        "expected_nu": expected_nu,
        "is_spot": is_spot,
        "task": task,
        "config": config,
        "size_data": size_data,
        "results": batch_results,
        "h5_path": output_path,
    }


class TestMPCExecution:
    """Test that MPC runs and produces valid trajectory data."""

    def test_result_keys(self, mpc_run: dict[str, Any]) -> None:
        """Each result dict should contain required trajectory data keys."""
        result = mpc_run["results"][0]
        required_keys = {"task_step", "qpos", "qvel", "control", "sensor", "reward", "success"}
        assert required_keys.issubset(result.keys())

    def test_qpos_shape(self, mpc_run: dict[str, Any]) -> None:
        """Qpos shape should match (max_steps, nq)."""
        result = mpc_run["results"][0]
        nq = mpc_run["task"].model.nq
        assert result["qpos"].shape == (MAX_NUM_TASK_STEPS, nq)

    def test_qvel_shape(self, mpc_run: dict[str, Any]) -> None:
        """Qvel shape should match (max_steps, nv)."""
        result = mpc_run["results"][0]
        nv = mpc_run["task"].model.nv
        assert result["qvel"].shape == (MAX_NUM_TASK_STEPS, nv)

    def test_control_shape(self, mpc_run: dict[str, Any]) -> None:
        """Control shape should match (max_steps, expected_nu)."""
        result = mpc_run["results"][0]
        assert result["control"].shape == (MAX_NUM_TASK_STEPS, mpc_run["expected_nu"])

    def test_reward_shape_and_finite(self, mpc_run: dict[str, Any]) -> None:
        """Rewards should be finite with correct shape."""
        result = mpc_run["results"][0]
        assert result["reward"].shape == (MAX_NUM_TASK_STEPS,)
        assert np.all(np.isfinite(result["reward"]))

    def test_qpos_has_data(self, mpc_run: dict[str, Any]) -> None:
        """Qpos should have real data, not all NaN."""
        result = mpc_run["results"][0]
        assert not np.all(np.isnan(result["qpos"]))

    def test_rollout_and_viapoint_data_present(self, mpc_run: dict[str, Any]) -> None:
        """Rollout and viapoint data should be stored when requested."""
        result = mpc_run["results"][0]
        assert "rollout_states" in result
        assert "rollout_controls" in result
        assert "rollout_rewards" in result
        assert "control_viapoints" in result


class TestH5Output:
    """Test saving results to HDF5 and reading them back."""

    def test_h5_structure(self, mpc_run: dict[str, Any]) -> None:
        """HDF5 file should have correct dataset shapes and metadata."""
        h5_path = mpc_run["h5_path"]
        task = mpc_run["task"]
        num_saved = len(mpc_run["results"])

        with h5py.File(str(h5_path), "r") as f:
            qpos_ds: h5py.Dataset = f["qpos"]  # type: ignore[assignment]
            qvel_ds: h5py.Dataset = f["qvel"]  # type: ignore[assignment]
            control_ds: h5py.Dataset = f["control"]  # type: ignore[assignment]
            sensor_ds: h5py.Dataset = f["sensor"]  # type: ignore[assignment]
            reward_ds: h5py.Dataset = f["reward"]  # type: ignore[assignment]
            traj_len_ds: h5py.Dataset = f["trajectory_length"]  # type: ignore[assignment]

            assert qpos_ds.shape == (num_saved, MAX_NUM_TASK_STEPS, task.model.nq)
            assert qvel_ds.shape == (num_saved, MAX_NUM_TASK_STEPS, task.model.nv)
            assert control_ds.shape == (num_saved, MAX_NUM_TASK_STEPS, mpc_run["expected_nu"])
            assert sensor_ds.shape == (num_saved, MAX_NUM_TASK_STEPS, task.model.nsensordata)
            assert reward_ds.shape == (num_saved, MAX_NUM_TASK_STEPS)
            assert traj_len_ds.shape == (num_saved,)

            assert "rollout_states" in f
            assert "control_viapoints" in f
            assert "task" in f.attrs

            assert np.any(np.isfinite(qpos_ds[0]))
