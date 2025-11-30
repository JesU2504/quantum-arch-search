#!/usr/bin/env python3
"""
Generate classification comparison plots.

This module reads classification metrics from compute_classif_metrics output
and generates plots for comparing DRL vs EA methods:
- Validation/test accuracy vs evaluations (learning curves)
- ECDF of final accuracies
- Pareto plot: accuracy vs circuit depth/gate count
- Boxplots of final accuracies by method

Usage:
    python -m comparison.analysis.generate_classif_plots \\
        --input comparison/logs/classif_analysis/classif_metrics_summary.json \\
        --out comparison/logs/plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_metrics(path: str) -> dict:
    """Load classification metrics from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_accuracy_vs_evals(
    metrics: dict,
    out_dir: Path,
    title: str = "Accuracy vs Evaluations",
    filename: str = "accuracy_vs_evals.png"
) -> Optional[Path]:
    """
    Plot accuracy learning curves for each method.

    Args:
        metrics: Aggregated metrics dictionary
        out_dir: Output directory for plots
        title: Plot title
        filename: Output filename

    Returns:
        Path to saved plot, or None if plotting unavailable
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'drl': 'blue', 'ea': 'orange'}

    by_method = metrics.get('by_method', {})
    for method, stats in by_method.items():
        color = colors.get(method, 'gray')
        n_runs = stats.get('n_runs', 0)
        mean_acc = stats.get('mean_max_val_accuracy')
        std_acc = stats.get('std_max_val_accuracy', 0) or 0

        if mean_acc is not None:
            # Plot a bar for mean accuracy (simplified visualization)
            ax.bar(method.upper(), mean_acc, yerr=std_acc, color=color,
                   alpha=0.7, capsize=5, label=f'{method.upper()} (n={n_runs})')

    ax.set_xlabel('Method')
    ax.set_ylabel('Max Validation Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add threshold lines
    for thresh in [0.70, 0.80, 0.90]:
        ax.axhline(y=thresh, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ecdf(
    metrics: dict,
    out_dir: Path,
    title: str = "ECDF of Final Accuracies",
    filename: str = "ecdf_accuracies.png"
) -> Optional[Path]:
    """
    Plot empirical CDF of final accuracies for each method.

    Args:
        metrics: Aggregated metrics dictionary
        out_dir: Output directory for plots
        title: Plot title
        filename: Output filename

    Returns:
        Path to saved plot, or None if plotting unavailable
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'drl': 'blue', 'ea': 'orange'}

    per_run = metrics.get('per_run', {})

    # Group accuracies by method
    method_accs: dict[str, list] = {}
    for run_key, run_data in per_run.items():
        method = run_data.get('method', 'unknown')
        acc = run_data.get('max_val_accuracy') or run_data.get('final_val_accuracy')
        if acc is not None:
            if method not in method_accs:
                method_accs[method] = []
            method_accs[method].append(acc)

    for method, accs in method_accs.items():
        if not accs:
            continue
        sorted_accs = sorted(accs)
        n = len(sorted_accs)
        ecdf_y = [(i + 1) / n for i in range(n)]
        color = colors.get(method, 'gray')
        ax.step(sorted_accs, ecdf_y, where='post', color=color,
                linewidth=2, label=f'{method.upper()} (n={n})')

    ax.set_xlabel('Final Accuracy')
    ax.set_ylabel('ECDF')
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pareto(
    metrics: dict,
    out_dir: Path,
    title: str = "Pareto: Accuracy vs Circuit Depth",
    filename: str = "pareto_acc_depth.png"
) -> Optional[Path]:
    """
    Plot Pareto front of accuracy vs circuit depth/gate count.

    Args:
        metrics: Aggregated metrics dictionary
        out_dir: Output directory for plots
        title: Plot title
        filename: Output filename

    Returns:
        Path to saved plot, or None if plotting unavailable
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'drl': 'blue', 'ea': 'orange'}
    markers = {'drl': 'o', 'ea': 's'}

    per_run = metrics.get('per_run', {})

    for run_key, run_data in per_run.items():
        method = run_data.get('method', 'unknown')
        acc = run_data.get('max_val_accuracy') or run_data.get('final_val_accuracy')
        depth = run_data.get('final_depth') or run_data.get('min_depth')
        gate_count = run_data.get('final_gate_count') or run_data.get('min_gate_count')

        # Use gate_count if depth not available
        x_val = depth if depth is not None else gate_count
        if acc is not None and x_val is not None:
            color = colors.get(method, 'gray')
            marker = markers.get(method, 'x')
            ax.scatter(x_val, acc, c=color, marker=marker, s=100, alpha=0.7,
                      label=method.upper() if run_key.endswith('_seed0') else '')

    ax.set_xlabel('Circuit Depth / Gate Count')
    ax.set_ylabel('Best Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_boxplots(
    metrics: dict,
    out_dir: Path,
    title: str = "Final Accuracy Distribution",
    filename: str = "boxplot_accuracies.png"
) -> Optional[Path]:
    """
    Plot boxplots of final accuracies by method.

    Args:
        metrics: Aggregated metrics dictionary
        out_dir: Output directory for plots
        title: Plot title
        filename: Output filename

    Returns:
        Path to saved plot, or None if plotting unavailable
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'orange']

    per_run = metrics.get('per_run', {})

    # Group accuracies by method
    method_accs: dict[str, list] = {}
    for run_key, run_data in per_run.items():
        method = run_data.get('method', 'unknown')
        acc = run_data.get('max_val_accuracy') or run_data.get('final_val_accuracy')
        if acc is not None:
            if method not in method_accs:
                method_accs[method] = []
            method_accs[method].append(acc)

    if not method_accs:
        print("No data for boxplot")
        plt.close(fig)
        return None

    methods = sorted(method_accs.keys())
    data = [method_accs[m] for m in methods]
    labels = [m.upper() for m in methods]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors[:len(methods)])):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('Final Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Add threshold lines
    for thresh in [0.70, 0.80, 0.90]:
        ax.axhline(y=thresh, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_plots(metrics: dict, out_dir: Path) -> list:
    """
    Generate all classification comparison plots.

    Args:
        metrics: Aggregated metrics dictionary
        out_dir: Output directory for plots

    Returns:
        List of paths to generated plots
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    plot_funcs = [
        (plot_accuracy_vs_evals, "Accuracy vs Evaluations"),
        (plot_ecdf, "ECDF of Final Accuracies"),
        (plot_pareto, "Pareto: Accuracy vs Depth"),
        (plot_boxplots, "Boxplot of Final Accuracies"),
    ]

    for func, name in plot_funcs:
        print(f"Generating: {name}...")
        path = func(metrics, out_dir)
        if path:
            plots.append(path)
            print(f"  Saved: {path}")
        else:
            print(f"  Skipped (no data or matplotlib unavailable)")

    return plots


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate classification comparison plots from metrics.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to classification metrics JSON file'
    )
    parser.add_argument(
        '--out', '-o',
        default='comparison/logs/plots',
        help='Output directory for plots (default: comparison/logs/plots)'
    )
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plot generation.", file=sys.stderr)
        print("Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    # Load metrics
    metrics_path = Path(args.input)
    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading metrics from: {metrics_path}")
    metrics = load_metrics(str(metrics_path))

    # Generate plots
    out_dir = Path(args.out)
    print(f"Generating plots to: {out_dir}")
    plots = generate_all_plots(metrics, out_dir)

    print(f"\nGenerated {len(plots)} plots:")
    for p in plots:
        print(f"  - {p}")


if __name__ == '__main__':
    main()
