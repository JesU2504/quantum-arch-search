"""
Utilities package for Quantum Architecture Search.

See ExpPlan.md, Implementation notes section. This package contains:
- metrics: Fidelity computation, CNOT counting, evaluation metrics
- parallel: Vectorized environment utilities (DummyVecEnv, SubprocVecEnv)
- plotting: Visualization utilities for training curves and Pareto plots

TODO: Expose utility functions as the package develops.
"""

from .metrics import (
    compute_fidelity,
    count_cnots,
    count_gates,
    get_circuit_depth,
    evaluate_circuit,
    state_energy,
    ghz_circuit,
    ideal_ghz_state,
    state_fidelity,
    simulate_circuit,
    fidelity_retention_ratio,
)
from .parallel import create_vec_env, benchmark_parallelism
from .plotting import (
    plot_training_curve,
    plot_pareto_frontier,
    plot_lambda_sweep,
    plot_qubit_overhead,
    plot_noise_resilience,
)

__all__ = [
    # Metrics
    "compute_fidelity",
    "count_cnots",
    "count_gates",
    "get_circuit_depth",
    "evaluate_circuit",
    "state_energy",
    "ghz_circuit",
    "ideal_ghz_state",
    "state_fidelity",
    "simulate_circuit",
    "fidelity_retention_ratio",
    # Parallelism
    "create_vec_env",
    "benchmark_parallelism",
    # Plotting
    "plot_training_curve",
    "plot_pareto_frontier",
    "plot_lambda_sweep",
    "plot_qubit_overhead",
    "plot_noise_resilience",
]
