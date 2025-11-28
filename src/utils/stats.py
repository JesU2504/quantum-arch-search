"""
Statistical aggregation utilities for multi-seed quantum architecture experiments.

This module provides functions for:
- Aggregating results across multiple random seeds
- Computing mean, standard deviation, and confidence intervals
- Generating summary statistics for experiments
- Creating summary files with metadata

Statistical Protocol:
    - All experiments should run with at least 5 seeds (configurable, recommended 10)
    - Each seed's results are saved separately for reproducibility
    - Aggregated metrics include mean ± std for all key quantities
    - Plots display error bars (mean ± std) with sample size annotations

See README.md for full statistical reporting guidelines.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np


def aggregate_metrics(
    values: List[float],
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute aggregate statistics for a list of values.

    Args:
        values: List of metric values from multiple seeds.
        confidence_level: Confidence level for interval (default 0.95).

    Returns:
        Dict with 'mean', 'std', 'median', 'min', 'max', 'n',
        'sem' (standard error), and 'ci_lower', 'ci_upper'.
    """
    if not values:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'median': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'n': 0,
            'sem': float('nan'),
            'ci_lower': float('nan'),
            'ci_upper': float('nan'),
        }

    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0

    # Compute confidence interval using t-distribution
    from scipy import stats
    if n > 1:
        t_val = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)
        ci_half = t_val * sem
    else:
        ci_half = 0.0

    return {
        'mean': mean,
        'std': std,
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'n': n,
        'sem': float(sem),
        'ci_lower': mean - ci_half,
        'ci_upper': mean + ci_half,
    }


def aggregate_seed_results(
    seed_results: List[Dict[str, Any]],
    metric_keys: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results from multiple seeds for specified metrics.

    Args:
        seed_results: List of result dicts, one per seed.
        metric_keys: List of metric names to aggregate.

    Returns:
        Dict mapping metric_name -> aggregate_stats dict.
    """
    aggregated = {}
    for key in metric_keys:
        values = []
        for result in seed_results:
            if key in result:
                val = result[key]
                if isinstance(val, (int, float)):
                    values.append(float(val))
                elif isinstance(val, list) and len(val) > 0:
                    # Take the final value if it's a list (e.g., training curve)
                    values.append(float(val[-1]))
        aggregated[key] = aggregate_metrics(values)
    return aggregated


def create_experiment_summary(
    experiment_name: str,
    n_seeds: int,
    seeds_used: List[int],
    hyperparameters: Dict[str, Any],
    aggregated_results: Dict[str, Dict[str, float]],
    noise_seeds: Optional[List[int]] = None,
    noise_parameters: Optional[Dict[str, Any]] = None,
    script_version: Optional[str] = None,
    commit_hash: Optional[str] = None,
    additional_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a comprehensive summary dictionary for an experiment.

    Args:
        experiment_name: Name of the experiment (e.g., 'lambda_sweep', 'adversarial').
        n_seeds: Total number of seeds used.
        seeds_used: List of actual seed values.
        hyperparameters: Dict of hyperparameters used.
        aggregated_results: Dict of aggregated metrics.
        noise_seeds: Optional list of noise seeds used for stochastic elements.
        noise_parameters: Optional dict of noise parameters.
        script_version: Optional version string for the experiment script.
        commit_hash: Optional git commit hash.
        additional_notes: Optional additional notes.

    Returns:
        Summary dict ready to be saved as JSON.
    """
    summary = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'statistical_protocol': {
            'n_seeds': n_seeds,
            'seeds_used': seeds_used,
            'aggregation_method': 'mean ± std (sample standard deviation with ddof=1)',
            'confidence_level': 0.95,
        },
        'hyperparameters': hyperparameters,
        'aggregated_results': aggregated_results,
    }

    if noise_seeds is not None:
        summary['noise_seeds'] = noise_seeds
    if noise_parameters is not None:
        summary['noise_parameters'] = noise_parameters
    if script_version is not None:
        summary['script_version'] = script_version
    if commit_hash is not None:
        summary['commit_hash'] = commit_hash
    if additional_notes is not None:
        summary['additional_notes'] = additional_notes

    return summary


def save_experiment_summary(
    summary: Dict[str, Any],
    output_dir: str,
    filename: str = 'experiment_summary.json'
) -> str:
    """
    Save experiment summary to a JSON file.

    Args:
        summary: Summary dict from create_experiment_summary().
        output_dir: Directory to save the summary.
        filename: Name of the summary file.

    Returns:
        Path to the saved summary file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    return filepath


def write_summary_txt(
    output_dir: str,
    experiment_name: str,
    n_seeds: int,
    seeds_used: List[int],
    hyperparameters: Dict[str, Any],
    aggregated_results: Dict[str, Dict[str, float]],
    noise_seeds: Optional[List[int]] = None,
    noise_parameters: Optional[Dict[str, Any]] = None,
    filename: str = 'experiment_summary.txt'
) -> str:
    """
    Write a human-readable summary text file.

    Args:
        output_dir: Directory to save the summary.
        experiment_name: Name of the experiment.
        n_seeds: Number of seeds used.
        seeds_used: List of seed values.
        hyperparameters: Hyperparameters dict.
        aggregated_results: Aggregated results dict.
        noise_seeds: Optional noise seeds.
        noise_parameters: Optional noise parameters.
        filename: Output filename.

    Returns:
        Path to the saved summary file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"Experiment Summary: {experiment_name}\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        f.write("Statistical Protocol:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Number of seeds (n): {n_seeds}\n")
        f.write(f"  Seeds used: {seeds_used}\n")
        f.write("  Aggregation: mean ± std (sample std, ddof=1)\n")
        f.write("  Confidence level: 95%\n\n")

        if noise_seeds is not None:
            f.write(f"  Noise seeds: {noise_seeds}\n")
        if noise_parameters is not None:
            f.write(f"  Noise parameters: {noise_parameters}\n")
        f.write("\n")

        f.write("Hyperparameters:\n")
        f.write("-" * 40 + "\n")
        for key, val in hyperparameters.items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")

        f.write("Aggregated Results:\n")
        f.write("-" * 40 + "\n")
        for metric_name, stats in aggregated_results.items():
            if isinstance(stats, dict) and 'mean' in stats:
                f.write(f"\n  {metric_name}:\n")
                f.write(f"    Mean ± Std: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
                f.write(f"    Median: {stats['median']:.6f}\n")
                f.write(f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"    n = {stats['n']}\n")
                f.write(f"    95% CI: [{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}]\n")
            else:
                f.write(f"  {metric_name}: {stats}\n")

        f.write("\n" + "=" * 60 + "\n")

    return filepath


def get_git_commit_hash() -> Optional[str]:
    """
    Get the current git commit hash, if available.

    Returns:
        Git commit hash string, or None if not in a git repo.
    """
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
    except Exception:
        pass
    return None


def format_metric_with_error(
    mean: float,
    std: float,
    n: int,
    precision: int = 4
) -> str:
    """
    Format a metric value with error bar and sample size annotation.

    Args:
        mean: Mean value.
        std: Standard deviation.
        n: Sample size.
        precision: Number of decimal places.

    Returns:
        Formatted string like "0.9523 ± 0.0125 (n=10)".
    """
    return f"{mean:.{precision}f} ± {std:.{precision}f} (n={n})"


def compute_success_rate(
    successes: List[bool],
) -> Dict[str, float]:
    """
    Compute success rate with confidence interval.

    Uses Wilson score interval for binomial proportion.

    Args:
        successes: List of boolean success indicators.

    Returns:
        Dict with 'rate', 'n_success', 'n_total', 'ci_lower', 'ci_upper'.
    """
    n = len(successes)
    if n == 0:
        return {
            'rate': float('nan'),
            'n_success': 0,
            'n_total': 0,
            'ci_lower': float('nan'),
            'ci_upper': float('nan'),
        }

    n_success = sum(successes)
    rate = n_success / n

    # Wilson score interval
    z = 1.96  # 95% confidence
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denominator

    return {
        'rate': rate,
        'n_success': n_success,
        'n_total': n,
        'ci_lower': max(0, center - spread),
        'ci_upper': min(1, center + spread),
    }
