"""
Utilities package for Quantum Architecture Search.

See ExpPlan.md, Implementation notes section. This package contains:
- metrics: Fidelity computation, CNOT counting, evaluation metrics
- parallel: Vectorized environment utilities (DummyVecEnv, SubprocVecEnv)
- plotting: Visualization utilities for training curves and Pareto plots

TODO: Expose utility functions as the package develops.
"""

from .metrics import compute_fidelity, count_cnots, evaluate_circuit
from .parallel import create_vec_env, benchmark_parallelism
from .plotting import plot_training_curve, plot_pareto_frontier

__all__ = [
    # Metrics
    "compute_fidelity",
    "count_cnots",
    "evaluate_circuit",
    # Parallelism
    "create_vec_env",
    "benchmark_parallelism",
    # Plotting
    "plot_training_curve",
    "plot_pareto_frontier",
]
