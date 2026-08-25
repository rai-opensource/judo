# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import atexit
import os
import signal
import subprocess
import warnings
from pathlib import Path

import hydra
from dora_utils.launch.run import run
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

# suppress annoying warning from hydra
warnings.filterwarnings(
    "ignore",
    message=r".*Defaults list is missing `_self_`.*",
    category=UserWarning,
    module="hydra._internal.defaults_list",
)

CONFIG_PATH = (Path(__file__).parent / "configs").resolve()

# ################################################################ #
# Process cleanup for Ctrl+C                                       #
#                                                                  #
# Problems with dora_utils's cleanup (run.py):                     #
# 1. Calls `dora destroy` BEFORE terminating nodes, so nodes hit   #
#    "broken pipe" trying to notify the dead daemon (noisy warns). #
# 2. Catches TimeoutExpired with no SIGKILL fallback → orphans.    #
# 3. `dora up` starts the daemon as a separate background process  #
#    outside any tracked process group — if `dora destroy` is      #
#    interrupted (double Ctrl+C), the daemon survives.             #
#                                                                  #
# Fix: custom SIGINT handler that counts presses.                  #
#  - 1st Ctrl+C: SIGTERM nodes (while daemon alive → no broken     #
#    pipe warnings), then raise KeyboardInterrupt for run.py's     #
#    normal cleanup path.                                          #
#  - 2nd Ctrl+C: uninterruptible force cleanup — `dora destroy`    #
#    in a new session + SIGKILL all pgids + os._exit().            #
#  - atexit: same force cleanup as safety net for single Ctrl+C.   #
# ################################################################ #

_spawned_pgids: list[int] = []
_node_pgids: list[int] = []
_OriginalPopen = subprocess.Popen
_sigint_count = 0


class _TrackingPopen(_OriginalPopen):  # type: ignore[misc]
    """Popen wrapper that records process group IDs of new-session children."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        if kwargs.get("start_new_session"):
            _spawned_pgids.append(self.pid)
            cmd = args[0] if args else kwargs.get("args", "")
            if isinstance(cmd, str) and "dora" not in cmd:
                _node_pgids.append(self.pid)


def _force_cleanup() -> None:
    """Destroy dora daemon and SIGKILL all tracked process groups.

    Safe to call from signal handlers and atexit. Idempotent.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _OriginalPopen(
            ["dora", "destroy"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).wait(timeout=5)
    except Exception:
        pass
    for pgid in _spawned_pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _sigint_handler(signum: int, frame: object) -> None:
    """Handle Ctrl+C: graceful on first press, force-kill on second."""
    global _sigint_count  # noqa: PLW0603
    _sigint_count += 1
    if _sigint_count >= 2:
        # Second Ctrl+C: uninterruptible force cleanup, then exit immediately.
        _force_cleanup()
        os._exit(130)
    # First Ctrl+C: SIGTERM nodes while daemon is still alive so they can
    # disconnect cleanly (avoids broken-pipe warnings), then raise
    # KeyboardInterrupt for run.py's normal cleanup path.
    for pgid in _node_pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    raise KeyboardInterrupt


subprocess.Popen = _TrackingPopen  # type: ignore[misc]
atexit.register(_force_cleanup)
signal.signal(signal.SIGINT, _sigint_handler)


# ### #
# app #
# ### #


# The default configuration file assumes the user is using Dora-RS as a middleware. The Judo app is designed to be
# compatible with different middleware options (e.g. Dora-RS, ZeroMQ, ROS2, etc.).
@hydra.main(config_path=str(CONFIG_PATH), config_name="judo_dora_default", version_base="1.3")
def main_app(cfg: DictConfig) -> None:
    """Main function to run judo via a hydra configuration yaml file."""
    try:
        run(cfg)
    except (KeyboardInterrupt, SystemExit, subprocess.CalledProcessError):
        # On shutdown (e.g. Ctrl+C), dora_utils' cleanup (run.py) calls `dora destroy` with
        # check=True; if the dataflow is already being torn down it exits non-zero and raises
        # CalledProcessError. Our own _force_cleanup() already handles teardown, so all of these
        # are expected shutdown paths rather than real failures.
        _force_cleanup()


def _warm_caches() -> None:
    """Pre-populate caches before Dora spawns subprocesses.

    robot_descriptions clones mujoco_menagerie on first import with no
    cross-process locking, so we must do it here in the single parent
    process to avoid races between Dora nodes.
    """
    from judo import MODEL_PATH  # noqa: PLC0415
    from judo.utils.assets import download_and_extract_meshes  # noqa: PLC0415

    download_and_extract_meshes(extract_root=str(MODEL_PATH), repo="bdaiinstitute/judo", asset_name="meshes.zip")

    try:
        from robot_descriptions import spot_mj_description  # noqa: F401, PLC0415
    except Exception:
        pass  # non-Spot tasks don't need this


def _require_mujoco_extensions() -> None:
    """Fail fast if mujoco_extensions is unavailable in the current environment."""
    try:
        import judo.mujoco_extensions  # noqa: F401, PLC0415
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "mujoco_extensions is required but could not be imported. Build it with: pixi run build"
        ) from e


def app() -> None:
    """Entry point for the judo CLI."""
    _require_mujoco_extensions()
    _warm_caches()
    # we store judo_dora_default in the config store so that custom dora configs outside of judo can inherit from it
    cs = ConfigStore.instance()
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
        default_cfg = compose(config_name="judo_dora_default")
        cs.store("judo_dora", default_cfg)  # don't name this judo_dora_default so it doesn't clash
    main_app()


# ######### #
# benchmark #
# ######### #


@hydra.main(config_path=str(CONFIG_PATH), config_name="benchmark_default", version_base="1.3")
def main_benchmark(cfg: DictConfig) -> None:
    """Benchmarking hydra call."""
    try:
        run(cfg)
    except (KeyboardInterrupt, SystemExit):
        _force_cleanup()


def benchmark() -> None:
    """Entry point for benchmarking."""
    _require_mujoco_extensions()
    # we store benchmark_default in the config store so that custom configs located outside of judo can inherit from it
    cs = ConfigStore.instance()
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
        default_cfg = compose(config_name="benchmark_default")
        cs.store("benchmark", default_cfg)  # don't name this benchmark_default so it doesn't clash
    main_benchmark()
