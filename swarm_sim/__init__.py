"""Schwarm-Simulation – GPU-beschleunigte Multi-Agent-Arena.

Phase 1: vektorisierte 2D-Physik + regelbasiertes Schwarmverhalten + Viewer.
Alle Zustands-Tensoren haben die Form (E, A, ...) mit
E = Anzahl paralleler Arenen (Environments), A = Agenten pro Arena.
"""

from .config import SimConfig, TrainConfig
from .physics import PhysicsWorld
from .swarm import BoidsController
from .recorder import ReplayRecorder
from .mesh import MeshNetwork, MeshState
from .metrics import MetricsLogger
from .env import ZoneControlEnv, NUM_ACTIONS
from .mappo import MAPPO, ActorCritic, compute_gae

__all__ = [
    "SimConfig",
    "TrainConfig",
    "PhysicsWorld",
    "BoidsController",
    "ReplayRecorder",
    "MeshNetwork",
    "MeshState",
    "MetricsLogger",
    "ZoneControlEnv",
    "NUM_ACTIONS",
    "MAPPO",
    "ActorCritic",
    "compute_gae",
]
