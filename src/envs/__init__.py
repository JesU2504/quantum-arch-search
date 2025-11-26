"""
Environments package for Quantum Architecture Search.

See ExpPlan.md for the full experimental plan. This package contains:
- ArchitectEnv: Base environment for the architect agent.
- AdversarialArchitectEnv: Architect environment with adversarial evaluation.
- Saboteur: Noise injection agent for robustness testing.
- VQEArchitectEnv: Environment for VQE-based architecture search.

TODO: Expose all environment classes via this __init__.py as the package develops.
"""

from .architect_env import ArchitectEnv
from .adversarial_architect_env import AdversarialArchitectEnv
from .saboteur import Saboteur
from .vqe_architect_env import VQEArchitectEnv

__all__ = [
    "ArchitectEnv",
    "AdversarialArchitectEnv",
    "Saboteur",
    "VQEArchitectEnv",
]
