"""
Environments package for Quantum Architecture Search.

DEPRECATED: This module is deprecated. All environments have been unified under
src.qas_gym.envs. Please update your imports to use:

    from src.qas_gym.envs import (
        ArchitectEnv,
        AdversarialArchitectEnv,
        Saboteur,
        VQEArchitectEnv,
    )

This module re-exports from src.qas_gym.envs for backward compatibility.
"""

# Re-export from canonical location for backward compatibility
from src.qas_gym.envs import (
    ArchitectEnv,
    AdversarialArchitectEnv,
    Saboteur,
    VQEArchitectEnv,
)

__all__ = [
    "ArchitectEnv",
    "AdversarialArchitectEnv",
    "Saboteur",
    "VQEArchitectEnv",
]
