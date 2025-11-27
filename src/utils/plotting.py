"""
Plotting utilities for Quantum Architecture Search.

See ExpPlan.md for visualization requirements:
  - Experiment 1.1: Dual Y-axis plot (Lambda sweep)
  - Experiment 3.1: Pareto scatter (CNOT count vs fidelity)
  - Experiment 6.1: Bar chart (Qubit overhead comparison)

This module provides standardized plotting functions for:
  - Training curves with confidence intervals
  - Pareto frontier visualization
  - Lambda sweep dual-axis plots
  - Qubit overhead bar charts
  - Noise resilience curves
"""

from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt


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
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric_name, values in metrics.items():
        x = np.arange(len(values))
        ax.plot(x, values, label=metric_name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


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
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to arrays
    cnots = np.array(cnot_counts)
    fidels = np.array(fidelities)

    # Identify Pareto-optimal points
    pareto_mask = _compute_pareto_mask(cnots, fidels)

    if labels is not None:
        # Group by labels and plot with different colors
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = np.array([l == label for l in labels])
            ax.scatter(
                cnots[mask], fidels[mask],
                c=[colors[i]], label=label, alpha=0.7, s=100
            )
    else:
        # Single color scatter
        ax.scatter(cnots, fidels, c="blue", alpha=0.7, s=100)

    # Highlight Pareto-optimal points
    pareto_cnots = cnots[pareto_mask]
    pareto_fidels = fidels[pareto_mask]
    # Sort for line plot
    sort_idx = np.argsort(pareto_cnots)
    ax.plot(
        pareto_cnots[sort_idx], pareto_fidels[sort_idx],
        "r--", linewidth=2, label="Pareto Frontier"
    )
    ax.scatter(
        pareto_cnots, pareto_fidels,
        c="red", marker="*", s=200, zorder=5, label="Pareto Optimal"
    )

    ax.set_xlabel("CNOT Count")
    ax.set_ylabel("Fidelity")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _compute_pareto_mask(cnots: np.ndarray, fidelities: np.ndarray) -> np.ndarray:
    """
    Compute mask for Pareto-optimal points.

    A point is Pareto-optimal if no other point has both lower CNOT count
    AND higher fidelity.

    Args:
        cnots: Array of CNOT counts.
        fidelities: Array of fidelities.

    Returns:
        Boolean mask array indicating Pareto-optimal points.
    """
    n = len(cnots)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Check if j dominates i (lower CNOTs AND higher fidelity)
                if cnots[j] <= cnots[i] and fidelities[j] >= fidelities[i]:
                    if cnots[j] < cnots[i] or fidelities[j] > fidelities[i]:
                        pareto_mask[i] = False
                        break

    return pareto_mask


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
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot success rate on left Y-axis
    color1 = "tab:blue"
    ax1.set_xlabel("Lambda (λ)")
    ax1.set_ylabel("Success Rate (%)", color=color1)
    ax1.plot(lambdas, success_rates, "o-", color=color1, linewidth=2, markersize=8)
    ax1.axhline(
        y=adversarial_success, color=color1, linestyle="--",
        linewidth=2, label=f"Adversarial Success: {adversarial_success:.1f}%"
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim([0, 105])

    # Create second Y-axis for depth
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Average Depth", color=color2)
    ax2.plot(lambdas, avg_depths, "s-", color=color2, linewidth=2, markersize=8)
    ax2.axhline(
        y=adversarial_depth, color=color2, linestyle="--",
        linewidth=2, label=f"Adversarial Depth: {adversarial_depth:.1f}"
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Set x-axis to log scale for lambda
    ax1.set_xscale("log")

    # Title and legend
    fig.suptitle("Lambda Sweep: Success Rate vs Circuit Depth", fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


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
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(method_names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(method_names)))

    bars = ax.bar(x, qubit_counts, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, count in zip(bars, qubit_counts):
        height = bar.get_height()
        ax.annotate(
            f"{count}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold"
        )

    ax.set_xlabel("Method")
    ax.set_ylabel("Number of Qubits")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.set_ylim([0, max(qubit_counts) * 1.2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


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
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        noise_levels, baseline_fidelities, "o-",
        color="tab:red", linewidth=2, markersize=8, label="Baseline (Static Penalty)"
    )
    ax.plot(
        noise_levels, robust_fidelities, "s-",
        color="tab:green", linewidth=2, markersize=8, label="Robust (Adversarial)"
    )

    ax.set_xlabel("Noise Level (Error Probability)")
    ax.set_ylabel("Fidelity")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
