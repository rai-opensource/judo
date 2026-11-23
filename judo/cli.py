# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from judo.unlocked.graph.runner import run_builder
from judo.unlocked.logging_util import setup_logger

# suppress annoying warning from hydra
warnings.filterwarnings(
    "ignore",
    message=r".*Defaults list is missing `_self_`.*",
    category=UserWarning,
    module="hydra._internal.defaults_list",
)

CONFIG_PATH = (Path(__file__).parent / "configs").resolve()


def _run(cfg: DictConfig, benchmark: bool = False) -> None:
    """Build the unLocked graph from the hydra config and run it until it stops."""
    from judo.app.graph import build, build_benchmark  # noqa: PLC0415

    setup_logger()
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(kwargs, dict)
    run_builder((build_benchmark if benchmark else build)(**{str(k): v for k, v in kwargs.items()}))


# ### #
# app #
# ### #


@hydra.main(config_path=str(CONFIG_PATH), config_name="judo_default", version_base="1.3")
def main_app(cfg: DictConfig) -> None:
    """Main function to run judo via a hydra configuration yaml file."""
    _run(cfg)


def _warm_caches() -> None:
    """Pre-populate caches before the graph forks its processes.

    robot_descriptions clones mujoco_menagerie on first import with no
    cross-process locking, so we must do it here in the single parent
    process to avoid races between nodes.
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
        import mujoco_extensions  # noqa: F401, PLC0415
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "mujoco_extensions is required but could not be imported. Build it with: pixi run build"
        ) from e


def app() -> None:
    """Entry point for the judo CLI."""
    _require_mujoco_extensions()
    _warm_caches()
    # we store judo_default in the config store so that custom configs outside of judo can inherit from it
    cs = ConfigStore.instance()
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
        default_cfg = compose(config_name="judo_default")
        cs.store("judo", default_cfg)  # don't name this judo_default so it doesn't clash
    main_app()


# ######### #
# benchmark #
# ######### #


@hydra.main(config_path=str(CONFIG_PATH), config_name="benchmark_default", version_base="1.3")
def main_benchmark(cfg: DictConfig) -> None:
    """Benchmarking hydra call."""
    _run(cfg, benchmark=True)


def benchmark() -> None:
    """Entry point for benchmarking."""
    _require_mujoco_extensions()
    # we store benchmark_default in the config store so that custom configs located outside of judo can inherit from it
    cs = ConfigStore.instance()
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
        default_cfg = compose(config_name="benchmark_default")
        cs.store("benchmark", default_cfg)  # don't name this benchmark_default so it doesn't clash
    main_benchmark()
