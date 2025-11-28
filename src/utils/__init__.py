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
    # Full basis-sweep fidelity for gate synthesis verification
    computational_basis_state,
    toffoli_truth_table,
    full_basis_fidelity,
    full_basis_fidelity_toffoli,
)
from .parallel import create_vec_env, benchmark_parallelism
from .plotting import (
    plot_training_curve,
    plot_pareto_frontier,
    plot_lambda_sweep,
    plot_qubit_overhead,
    plot_noise_resilience,
)
from .stats import (
    aggregate_metrics,
    aggregate_seed_results,
    create_experiment_summary,
    save_experiment_summary,
    write_summary_txt,
    get_git_commit_hash,
    format_metric_with_error,
    compute_success_rate,
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
    # Full basis-sweep fidelity for gate synthesis
    "computational_basis_state",
    "toffoli_truth_table",
    "full_basis_fidelity",
    "full_basis_fidelity_toffoli",
    # Parallelism
    "create_vec_env",
    "benchmark_parallelism",
    # Plotting
    "plot_training_curve",
    "plot_pareto_frontier",
    "plot_lambda_sweep",
    "plot_qubit_overhead",
    "plot_noise_resilience",
    # Stats
    "aggregate_metrics",
    "aggregate_seed_results",
    "create_experiment_summary",
    "save_experiment_summary",
    "write_summary_txt",
    "get_git_commit_hash",
    "format_metric_with_error",
    "compute_success_rate",
]
