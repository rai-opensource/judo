# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Entry point for running batched MPC and saving trajectories to HDF5.

Usage: python -m run_mpc --config-path <path_to_config.json>
"""

import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from judo.visualizers.visualizer import Visualizer
from run_mpc.mpc_batch import run_mpc_batch
from run_mpc.mpc_config import MPCTimers, PublicMPCConfig, decode_config, load_configs_from_json_data
from run_mpc.mpc_setup import clamp_for_mjwarp, save_results_to_h5, setup_mpc


def _load_json_config(config_path: Path) -> tuple[dict, Path]:
    """Load and decode a JSON MPC config file, returning (data, resolved_path)."""
    resolved = Path(config_path).resolve()
    logging.info(f"Loading policy config from {resolved}")
    with open(resolved, "r") as f:
        return json.load(f, object_hook=decode_config), resolved


def _create_visualizer(config: PublicMPCConfig, json_configs: dict, num_parallel: int) -> Visualizer | None:
    """Create a visualizer if enabled and in single-trajectory mode."""
    if not config.visualize:
        return None
    if num_parallel > 1:
        logging.warning("Visualization not supported in batched mode. Disabling.")
        return None
    return Visualizer(init_task=json_configs["task"], init_optimizer=json_configs["optimizer"])


def _resolve_output_path(config: PublicMPCConfig, config_path: Path) -> Path:
    """Resolve the HDF5 output path, creating parent directories as needed."""
    if config.dataset_output_path is None:
        output_path = config_path.parent / "trajectories.h5"
    else:
        output_path = Path(config.dataset_output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def run_mpc(config: PublicMPCConfig) -> None:
    """Run batched MPC and write trajectories to an HDF5 file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

    json_configs, config_path = _load_json_config(config.config_path)
    task, optimizer, controller_cfg = load_configs_from_json_data(json_configs)
    clamp_for_mjwarp(task.model)

    # Fail early if require_success is set but the task has no success() implementation
    from judo.tasks.base import Task  # noqa: PLC0415

    if config.require_success and type(task).success is Task.success:
        raise ValueError(
            f"Task '{json_configs['task']}' does not implement success(), "
            f"so all trajectories will be discarded with --require-success (the default). "
            f"Either add a success() method to the task or use --no-require-success."
        )

    num_parallel = min(config.num_parallel, config.num_trajectories)
    sims, batched_controllers, size_data = setup_mpc(
        config, json_configs, task, optimizer, controller_cfg, num_parallel
    )
    vis = _create_visualizer(config, json_configs, num_parallel)
    output_path = _resolve_output_path(config, config_path)

    logging.info(f"Running {config.num_trajectories} trajectories ({num_parallel} parallel) -> {output_path}")

    timers = MPCTimers.create()
    all_results: list[dict[str, np.ndarray | int]] = []
    total_attempted = 0
    max_attempts = config.num_trajectories * 2

    with tqdm(total=config.num_trajectories, desc="Successful trajectories", unit="traj") as pbar:
        while len(all_results) < config.num_trajectories and total_attempted < max_attempts:
            batch_results = run_mpc_batch(sims, batched_controllers, config, size_data, timers, vis=vis)
            for result in batch_results:
                if len(all_results) >= config.num_trajectories:
                    break
                if not config.require_success or result["success"]:
                    all_results.append(result)
                    pbar.update(1)
            total_attempted += len(batch_results)

    if len(all_results) < config.num_trajectories:
        logging.warning(
            f"Only {len(all_results)}/{config.num_trajectories} successful after {total_attempted} attempts"
        )

    logging.info(
        f"Success rate: {len(all_results)}/{total_attempted} ({100 * len(all_results) / max(total_attempted, 1):.0f}%)"
    )

    if len(all_results) == 0:
        logging.error("No successful trajectories. Skipping HDF5 save.")
    else:
        size_data.num_trajectories = len(all_results)
        timers.h5_write.tic()
        save_results_to_h5(output_path, all_results, size_data, config, config_path, json_configs, num_parallel)
        timers.h5_write.toc()
        logging.info(f"Saved {len(all_results)} trajectories to {output_path}")
    timers.print_all()
    logging.info("=== Rollout Backend Statistics ===")
    batched_controllers.print_timer_stats()


def main() -> None:
    """CLI entry point for run-mpc."""
    import tyro  # noqa: PLC0415

    run_mpc(tyro.cli(PublicMPCConfig))
