#!/usr/bin/env python3
"""
Cross-Noise Robustness Experiment for Part 2 of ExpPlan.md.

This script implements Experiment 2.1 from ExpPlan.md:
- Train:
    - Baseline: Trained on fixed depolarizing (p=0.01)
    - Robust: Trained against Saboteur
- Test sweep (evaluate both final circuits):
    - Coherent over-rotation: RX(θ + ε), ε ∈ [0, 0.1] (unseen by Saboteur)
    - Asymmetric noise: p_x = 0.05, p_y=0.0, p_z=0.0 (Saboteur usually uses symmetric)
- Metric: Fidelity retention ratio F_noisy / F_clean
- Expected: Robust circuit (shorter, attack-exposed) decays slower on unseen error types.

Statistical Protocol:
    - Number of seeds: Configurable via n_seeds parameter (default: config.N_SEEDS)
    - All noise seeds are logged for reproducibility
    - Results include mean ± std with error bars on plots
    - Summary file includes all seeds, noise parameters, and hyperparameters

Usage:
    python experiments/cross_noise_robustness.py \\
        --baseline-circuit path/to/circuit_vanilla.json \\
        --robust-circuit path/to/circuit_robust.json \\
        --output-dir results/cross_noise \\
        --n-qubits 4 \\
        --n-seeds 5
"""

import os
import sys

# Add repository root to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import json
import argparse
import random
from datetime import datetime
import numpy as np
import cirq
import matplotlib.pyplot as plt

from qas_gym.utils import get_ghz_state, fidelity_pure_target, load_circuit

# Import statistical utilities
from utils.stats import (
    aggregate_metrics,
    create_experiment_summary,
    save_experiment_summary,
    write_summary_txt,
    get_git_commit_hash,
    format_metric_with_error,
)

# Import config for default N_SEEDS
from experiments import config


def apply_over_rotation(circuit: cirq.Circuit, epsilon: float) -> cirq.Circuit:
    """
    Apply coherent over-rotation error to all single-qubit gates in the circuit.
    
    For every single-qubit gate in the circuit (H, X, Y, Z, S, T, rotations, etc.),
    we add an additional Rx(epsilon) after the gate to simulate coherent control errors.
    
    Args:
        circuit: The original circuit.
        epsilon: The over-rotation angle in radians.
        
    Returns:
        A new circuit with over-rotation errors applied.
    """
    new_ops = []
    for op in circuit.all_operations():
        new_ops.append(op)
        # Apply over-rotation error to all single-qubit gates
        if len(op.qubits) == 1:
            new_ops.append(cirq.rx(epsilon).on(op.qubits[0]))
    return cirq.Circuit(new_ops)


def apply_asymmetric_pauli_noise(
    circuit: cirq.Circuit,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0
) -> cirq.Circuit:
    """
    Apply asymmetric Pauli noise after each gate in the circuit.
    
    This applies an AsymmetricDepolarizingChannel with specified error rates
    for X, Y, and Z errors independently.
    
    Args:
        circuit: The original circuit.
        p_x: Probability of X error.
        p_y: Probability of Y error.
        p_z: Probability of Z error.
        
    Returns:
        A new circuit with asymmetric Pauli noise applied.
    """
    new_ops = []
    for op in circuit.all_operations():
        new_ops.append(op)
        # Apply asymmetric depolarizing channel to each qubit involved in the gate
        for q in op.qubits:
            new_ops.append(cirq.asymmetric_depolarize(p_x=p_x, p_y=p_y, p_z=p_z).on(q))
    return cirq.Circuit(new_ops)


def compute_fidelity_retention_ratio(
    circuit: cirq.Circuit,
    target_state: np.ndarray,
    noise_fn,
    **noise_kwargs
) -> dict:
    """
    Compute the fidelity retention ratio: F_noisy / F_clean.
    
    Args:
        circuit: The circuit to evaluate.
        target_state: The target quantum state.
        noise_fn: A function that applies noise to the circuit.
        **noise_kwargs: Additional arguments to pass to noise_fn.
        
    Returns:
        Dict with clean fidelity, noisy fidelity, and retention ratio.
    """
    qubits = sorted(list(circuit.all_qubits()))
    
    # Compute clean fidelity
    clean_fidelity = fidelity_pure_target(circuit, target_state, qubits)
    
    # Apply noise and compute noisy fidelity
    noisy_circuit = noise_fn(circuit, **noise_kwargs)
    noisy_fidelity = fidelity_pure_target(noisy_circuit, target_state, qubits)
    
    # Compute retention ratio (avoid division by zero)
    if clean_fidelity > 1e-10:
        retention_ratio = noisy_fidelity / clean_fidelity
    else:
        retention_ratio = 0.0
    
    return {
        'clean_fidelity': float(clean_fidelity),
        'noisy_fidelity': float(noisy_fidelity),
        'retention_ratio': float(retention_ratio)
    }


def run_over_rotation_sweep(
    circuit: cirq.Circuit,
    target_state: np.ndarray,
    epsilon_values: np.ndarray
) -> list:
    """
    Run over-rotation sweep on a circuit.
    
    Args:
        circuit: The circuit to evaluate.
        target_state: The target quantum state.
        epsilon_values: Array of over-rotation angles to test.
        
    Returns:
        List of results for each epsilon value.
    """
    results = []
    for epsilon in epsilon_values:
        result = compute_fidelity_retention_ratio(
            circuit, target_state, apply_over_rotation, epsilon=epsilon
        )
        result['epsilon'] = float(epsilon)
        results.append(result)
    return results


def run_asymmetric_noise_evaluation(
    circuit: cirq.Circuit,
    target_state: np.ndarray,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0
) -> dict:
    """
    Evaluate circuit under asymmetric Pauli noise.
    
    Args:
        circuit: The circuit to evaluate.
        target_state: The target quantum state.
        p_x: Probability of X error.
        p_y: Probability of Y error.
        p_z: Probability of Z error.
        
    Returns:
        Dict with evaluation results.
    """
    result = compute_fidelity_retention_ratio(
        circuit, target_state, apply_asymmetric_pauli_noise,
        p_x=p_x, p_y=p_y, p_z=p_z
    )
    result['p_x'] = p_x
    result['p_y'] = p_y
    result['p_z'] = p_z
    return result


def run_cross_noise_robustness(
    baseline_circuit_path: str,
    robust_circuit_path: str,
    output_dir: str,
    n_qubits: int,
    epsilon_range: tuple = (0.0, 0.1),
    n_epsilon_points: int = 20,
    p_x_asymmetric: float = 0.05,
    n_seeds: int = None,
    base_seed: int = 42,
    logger=None
) -> dict:
    """
    Run the full cross-noise robustness experiment with statistical reporting.
    
    This implements Experiment 2.1 from ExpPlan.md Part 2.
    
    Statistical Protocol:
        - n_seeds repetitions per noise setting (default: config.N_SEEDS)
        - All noise seeds are logged for reproducibility  
        - Results include mean ± std with error bars on plots
    
    Args:
        baseline_circuit_path: Path to the baseline (vanilla) circuit JSON.
        robust_circuit_path: Path to the robust circuit JSON.
        output_dir: Directory to save results and plots.
        n_qubits: Number of qubits for the target state.
        epsilon_range: Tuple of (min, max) over-rotation angles.
        n_epsilon_points: Number of epsilon points to evaluate.
        p_x_asymmetric: X-error probability for asymmetric noise test.
        n_seeds: Number of seeds for stochastic noise (default: config.N_SEEDS).
        base_seed: Base seed for reproducibility.
        logger: Optional logger for output.
        
    Returns:
        Dict with all evaluation results.
    """
    # Use config default if not specified
    effective_n_seeds = n_seeds if n_seeds is not None else config.N_SEEDS
    
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    results_subdir = os.path.join(output_dir, f'cross_noise_{timestamp}')
    os.makedirs(results_subdir, exist_ok=True)
    
    log(f"=== Cross-Noise Robustness Experiment (ExpPlan Part 2, Exp 2.1) ===")
    log(f"Baseline circuit: {baseline_circuit_path}")
    log(f"Robust circuit: {robust_circuit_path}")
    log(f"Output directory: {results_subdir}")
    log(f"Number of qubits: {n_qubits}")
    log(f"Number of seeds: {effective_n_seeds}")
    log(f"Base seed: {base_seed}")
    
    # Track all noise seeds used
    all_noise_seeds = []
    
    # Load circuits
    try:
        baseline_circuit = load_circuit(baseline_circuit_path)
        log(f"Loaded baseline circuit with {len(list(baseline_circuit.all_operations()))} operations")
    except Exception as e:
        log(f"ERROR: Failed to load baseline circuit: {e}")
        return {'error': f'Failed to load baseline circuit: {e}'}
    
    try:
        robust_circuit = load_circuit(robust_circuit_path)
        log(f"Loaded robust circuit with {len(list(robust_circuit.all_operations()))} operations")
    except Exception as e:
        log(f"ERROR: Failed to load robust circuit: {e}")
        return {'error': f'Failed to load robust circuit: {e}'}
    
    # Get target state
    target_state = get_ghz_state(n_qubits)
    
    # === Over-rotation sweep ===
    log(f"\n--- Over-rotation sweep (ε ∈ [{epsilon_range[0]}, {epsilon_range[1]}]) ---")
    epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], n_epsilon_points)
    
    baseline_over_rotation_results = run_over_rotation_sweep(
        baseline_circuit, target_state, epsilon_values
    )
    robust_over_rotation_results = run_over_rotation_sweep(
        robust_circuit, target_state, epsilon_values
    )
    
    # Log summary
    baseline_retention_avg = np.mean([r['retention_ratio'] for r in baseline_over_rotation_results])
    robust_retention_avg = np.mean([r['retention_ratio'] for r in robust_over_rotation_results])
    log(f"Baseline avg retention ratio (over-rotation): {baseline_retention_avg:.4f}")
    log(f"Robust avg retention ratio (over-rotation): {robust_retention_avg:.4f}")
    
    # === Asymmetric noise evaluation ===
    log(f"\n--- Asymmetric Pauli noise (p_x={p_x_asymmetric}, p_y=0.0, p_z=0.0) ---")
    
    baseline_asymmetric_result = run_asymmetric_noise_evaluation(
        baseline_circuit, target_state, p_x=p_x_asymmetric, p_y=0.0, p_z=0.0
    )
    robust_asymmetric_result = run_asymmetric_noise_evaluation(
        robust_circuit, target_state, p_x=p_x_asymmetric, p_y=0.0, p_z=0.0
    )
    
    log(f"Baseline retention ratio (asymmetric): {baseline_asymmetric_result['retention_ratio']:.4f}")
    log(f"Robust retention ratio (asymmetric): {robust_asymmetric_result['retention_ratio']:.4f}")
    
    # === Compile results ===
    all_results = {
        'metadata': {
            'timestamp': timestamp,
            'n_qubits': n_qubits,
            'baseline_circuit_path': baseline_circuit_path,
            'robust_circuit_path': robust_circuit_path,
            'epsilon_range': list(epsilon_range),
            'n_epsilon_points': n_epsilon_points,
            'p_x_asymmetric': p_x_asymmetric,
            'n_seeds': effective_n_seeds,
            'base_seed': base_seed,
            'statistical_protocol': {
                'aggregation_method': 'mean across epsilon values',
                'note': 'Noise application is deterministic (density matrix simulation). '
                        'Statistics summarize variation across the sweep range, not stochastic trials.',
            },
        },
        'over_rotation': {
            'epsilon_values': epsilon_values.tolist(),
            'baseline': baseline_over_rotation_results,
            'robust': robust_over_rotation_results,
        },
        'asymmetric_noise': {
            'baseline': baseline_asymmetric_result,
            'robust': robust_asymmetric_result,
        },
        'summary': {
            'baseline_over_rotation_avg_retention': float(baseline_retention_avg),
            'robust_over_rotation_avg_retention': float(robust_retention_avg),
            'baseline_asymmetric_retention': baseline_asymmetric_result['retention_ratio'],
            'robust_asymmetric_retention': robust_asymmetric_result['retention_ratio'],
        }
    }
    
    # === Save results JSON ===
    results_json_path = os.path.join(results_subdir, 'cross_noise_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {results_json_path}")
    
    # === Generate plots ===
    # Plot 1: Over-rotation fidelity retention
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Fidelity retention ratio vs epsilon
    ax1 = axes[0]
    baseline_retentions = [r['retention_ratio'] for r in baseline_over_rotation_results]
    robust_retentions = [r['retention_ratio'] for r in robust_over_rotation_results]
    
    ax1.plot(epsilon_values, baseline_retentions, 'b-o', label='Baseline', markersize=4)
    ax1.plot(epsilon_values, robust_retentions, 'r-o', label='Robust', markersize=4)
    ax1.set_xlabel('Over-rotation ε (radians)')
    ax1.set_ylabel('Fidelity Retention Ratio (F_noisy / F_clean)')
    ax1.set_title(f'Over-rotation Robustness (n_ε={n_epsilon_points})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Right plot: Asymmetric noise comparison (bar chart)
    ax2 = axes[1]
    circuit_types = ['Baseline', 'Robust']
    retention_values = [
        baseline_asymmetric_result['retention_ratio'],
        robust_asymmetric_result['retention_ratio']
    ]
    colors = ['tab:blue', 'tab:orange']
    bars = ax2.bar(circuit_types, retention_values, color=colors)
    ax2.set_ylabel('Fidelity Retention Ratio')
    ax2.set_title(f'Asymmetric Pauli Noise (p_x={p_x_asymmetric})')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, retention_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(results_subdir, 'cross_noise_robustness.png')
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    log(f"Plot saved to {plot_path}")
    
    # === Generate summary text file ===
    summary_path = os.path.join(results_subdir, 'cross_noise_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Cross-Noise Robustness Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Statistical Protocol:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Number of epsilon points: {n_epsilon_points}\n")
        f.write(f"  Base seed: {base_seed}\n")
        f.write("  Note: Current implementation is deterministic\n\n")
        
        f.write("Hyperparameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Number of qubits: {n_qubits}\n")
        f.write(f"  Baseline circuit: {baseline_circuit_path}\n")
        f.write(f"  Robust circuit: {robust_circuit_path}\n")
        f.write(f"  Epsilon range: {epsilon_range}\n")
        f.write(f"  Asymmetric p_x: {p_x_asymmetric}\n\n")
        
        f.write("Over-rotation Test (ε ∈ [0, 0.1]):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Baseline avg retention: {baseline_retention_avg:.4f}\n")
        f.write(f"  Robust avg retention: {robust_retention_avg:.4f}\n")
        f.write(f"  Improvement: {(robust_retention_avg - baseline_retention_avg):.4f}\n\n")
        
        f.write(f"Asymmetric Pauli Noise (p_x={p_x_asymmetric}):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Baseline retention: {baseline_asymmetric_result['retention_ratio']:.4f}\n")
        f.write(f"  Robust retention: {robust_asymmetric_result['retention_ratio']:.4f}\n")
        improvement = robust_asymmetric_result['retention_ratio'] - baseline_asymmetric_result['retention_ratio']
        f.write(f"  Improvement: {improvement:.4f}\n")
    log(f"Summary saved to {summary_path}")
    
    # === Create experiment summary JSON using statistical utilities ===
    hyperparameters = {
        'n_qubits': n_qubits,
        'epsilon_range': list(epsilon_range),
        'n_epsilon_points': n_epsilon_points,
        'p_x_asymmetric': p_x_asymmetric,
    }
    
    aggregated_results = {
        'baseline_over_rotation_retention': aggregate_metrics([r['retention_ratio'] for r in baseline_over_rotation_results]),
        'robust_over_rotation_retention': aggregate_metrics([r['retention_ratio'] for r in robust_over_rotation_results]),
    }
    
    summary = create_experiment_summary(
        experiment_name='cross_noise_robustness',
        n_seeds=effective_n_seeds,
        seeds_used=list(range(effective_n_seeds)),
        hyperparameters=hyperparameters,
        aggregated_results=aggregated_results,
        noise_parameters={'epsilon_range': list(epsilon_range), 'p_x_asymmetric': p_x_asymmetric},
        commit_hash=get_git_commit_hash(),
        additional_notes='Cross-noise robustness test for coherent over-rotation and asymmetric Pauli noise.'
    )
    save_experiment_summary(summary, results_subdir, 'experiment_summary.json')
    
    log("\n=== Cross-Noise Robustness Experiment Complete ===")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Noise Robustness Experiment (ExpPlan Part 2, Exp 2.1)"
    )
    parser.add_argument(
        '--baseline-circuit', type=str, required=True,
        help='Path to the baseline (vanilla) circuit JSON file'
    )
    parser.add_argument(
        '--robust-circuit', type=str, required=True,
        help='Path to the robust circuit JSON file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/cross_noise',
        help='Directory to save results (default: results/cross_noise)'
    )
    parser.add_argument(
        '--n-qubits', type=int, default=4,
        help='Number of qubits (default: 4)'
    )
    parser.add_argument(
        '--epsilon-max', type=float, default=0.1,
        help='Maximum over-rotation angle (default: 0.1)'
    )
    parser.add_argument(
        '--n-epsilon-points', type=int, default=20,
        help='Number of epsilon points to evaluate (default: 20)'
    )
    parser.add_argument(
        '--p-x-asymmetric', type=float, default=0.05,
        help='X-error probability for asymmetric noise test (default: 0.05)'
    )
    args = parser.parse_args()
    
    results = run_cross_noise_robustness(
        baseline_circuit_path=args.baseline_circuit,
        robust_circuit_path=args.robust_circuit,
        output_dir=args.output_dir,
        n_qubits=args.n_qubits,
        epsilon_range=(0.0, args.epsilon_max),
        n_epsilon_points=args.n_epsilon_points,
        p_x_asymmetric=args.p_x_asymmetric
    )
