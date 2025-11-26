"""
Plotting utilities for Quantum Architecture Search.

See ExpPlan.md for visualization requirements:
  - Experiment 1.1: Dual Y-axis plot (Lambda sweep)
  - Experiment 3.1: Pareto scatter (CNOT count vs fidelity)
  - Experiment 6.1: Bar chart (Qubit overhead comparison)

TODO: Implement the following:
  - Training curve plots with confidence intervals
  - Pareto frontier visualization
  - Lambda sweep dual-axis plots
  - Qubit overhead bar charts
"""

from typing import Dict, List, Optional


def plot_training_curve(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Progress",
    xlabel: str = "Timestep",
    ylabel: str = "Metric Value",
):
    """
    Plot training curves with optional confidence intervals.

    Args:
        metrics: Dict mapping metric names to lists of values.
        save_path: Path to save the figure (displays if None).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    # TODO: Import matplotlib
    # TODO: Create figure and axes
    # TODO: Plot each metric series
    # TODO: Add legend, labels, title
    # TODO: Save or display
    pass


def plot_pareto_frontier(
    cnot_counts: List[int],
    fidelities: List[float],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Pareto Frontier: CNOT Count vs Fidelity",
):
    """
    Create Pareto scatter plot for circuit optimization.

    See ExpPlan.md, Part 3 (Pareto scatter):
      - Adversarial points should dominate top-left (high F, low CNOT)

    Args:
        cnot_counts: List of CNOT counts per circuit.
        fidelities: List of fidelities per circuit.
        labels: Optional labels for different methods.
        save_path: Path to save the figure.
        title: Plot title.
    """
    # TODO: Create scatter plot
    # TODO: Identify and highlight Pareto-optimal points
    # TODO: Add labels for different methods
    # TODO: Save or display
    pass


def plot_lambda_sweep(
    lambdas: List[float],
    success_rates: List[float],
    avg_depths: List[float],
    adversarial_success: float,
    adversarial_depth: float,
    save_path: Optional[str] = None,
):
    """
    Create dual Y-axis plot for lambda sweep experiment.

    See ExpPlan.md, Part 1 (Lambda sweep):
      - X = lambda
      - Left Y = Success rate
      - Right Y = Avg. depth
      - Horizontal line for Adversarial performance

    Args:
        lambdas: List of lambda values tested.
        success_rates: Success rates for each lambda.
        avg_depths: Average depths for each lambda.
        adversarial_success: Adversarial method success rate.
        adversarial_depth: Adversarial method average depth.
        save_path: Path to save the figure.
    """
    # TODO: Create figure with dual y-axes
    # TODO: Plot success rate on left axis
    # TODO: Plot depth on right axis
    # TODO: Add horizontal lines for adversarial baseline
    # TODO: Save or display
    pass


def plot_qubit_overhead(
    method_names: List[str],
    qubit_counts: List[int],
    save_path: Optional[str] = None,
    title: str = "Qubit Overhead Comparison",
):
    """
    Create bar chart for qubit overhead comparison.

    See ExpPlan.md, Part 6 (NISQ vs QEC):
      - Bar A: Our method — 4 qubits
      - Bar B: Surface code QEC — 68 qubits

    Args:
        method_names: Names of methods being compared.
        qubit_counts: Number of qubits for each method.
        save_path: Path to save the figure.
        title: Plot title.
    """
    # TODO: Create bar chart
    # TODO: Add value labels on bars
    # TODO: Save or display
    pass


def plot_noise_resilience(
    noise_levels: List[float],
    baseline_fidelities: List[float],
    robust_fidelities: List[float],
    save_path: Optional[str] = None,
    title: str = "Noise Resilience Comparison",
):
    """
    Plot fidelity degradation under increasing noise.

    See ExpPlan.md, Part 2 (Cross-noise evaluation).

    Args:
        noise_levels: List of noise levels (e.g., error probabilities).
        baseline_fidelities: Fidelities for baseline circuit.
        robust_fidelities: Fidelities for robust (adversarial) circuit.
        save_path: Path to save the figure.
        title: Plot title.
    """
    # TODO: Create line plot
    # TODO: Add markers for each method
    # TODO: Add legend
    # TODO: Save or display
    pass
