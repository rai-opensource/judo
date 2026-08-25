# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo.simulation.base import Simulation
from judo.simulation.mj_simulation import MJSimulation

# ``hierarchical_mj_simulation`` imports the ``judo.mujoco_extensions`` C++ pybind extension, which
# is only available once it has been built (``pixi run build``). Make the import optional so that
# third-party consumers using only the plain MuJoCo backend can ``import judo.simulation`` without
# building the extension. The hierarchical backend is registered only when it imports successfully;
# requesting it otherwise fails at registry lookup with a clear ``KeyError``.
try:
    from judo.simulation.hierarchical_mj_simulation import HierarchicalMJSimulation

    _HIERARCHICAL_BACKEND: dict[str, type[Simulation]] = {"mujoco_hierarchical": HierarchicalMJSimulation}
    _HIERARCHICAL_AVAILABLE = True
except ImportError:
    _HIERARCHICAL_BACKEND = {}
    _HIERARCHICAL_AVAILABLE = False

DEFAULT_SIMULATION_BACKEND_REGISTRY: dict[str, type[Simulation]] = {
    "mujoco": MJSimulation,
    **_HIERARCHICAL_BACKEND,
}


def get_simulation_backend(simulation_backend: str) -> type:
    """Get the simulation class for a given backend.

    Args:
        simulation_backend: Name of the simulation backend to get.

    Returns:
        The simulation class for the given backend.
    """
    if simulation_backend not in DEFAULT_SIMULATION_BACKEND_REGISTRY:
        raise KeyError(f"Unknown simulation backend: {simulation_backend!r}")
    return DEFAULT_SIMULATION_BACKEND_REGISTRY[simulation_backend]


__all__ = [
    "Simulation",
    "MJSimulation",
    "DEFAULT_SIMULATION_BACKEND_REGISTRY",
    "get_simulation_backend",
]

if _HIERARCHICAL_AVAILABLE:
    __all__.append("HierarchicalMJSimulation")
