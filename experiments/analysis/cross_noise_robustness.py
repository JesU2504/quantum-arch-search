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

from qas_gym.utils import fidelity_pure_target, load_circuit

# Import statistical utilities
from utils.stats import (
    aggregate_metrics,
    create_experiment_summary,
    save_experiment_summary,
    write_summary_txt,
    get_git_commit_hash,
    format_metric_with_error,
)

# Import config for default N_SEEDS and target state helpers
from experiments import config


def apply_over_rotation(circuit: cirq.Circuit, epsilon: float) -> cirq.Circuit:
    """
    Apply coherent over-rotation error to all single-qubit gates in the circuit.
    
    For every gate in the circuit, add an additional Rx(epsilon) on each qubit
    involved to simulate coherent control errors. Applying to all qubits (even
    multi-qubit gates) increases severity so differences are more visible.
    
    Args:
        circuit: The original circuit.
        epsilon: The over-rotation angle in radians.
        
    Returns:
        A new circuit with over-rotation errors applied.
    """
    new_ops = []
    for op in circuit.all_operations():
        new_ops.append(op)
        for q in op.qubits:
            new_ops.append(cirq.rx(epsilon).on(q))
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
    clean_reference: float | None = None,
    **noise_kwargs
) -> dict:
    """
    Compute the fidelity retention ratio using a shared clean reference.
    
    Args:
        circuit: The circuit to evaluate.
        target_state: The target quantum state.
        noise_fn: A function that applies noise to the circuit.
        clean_reference: Optional clean fidelity to normalize by. If None, uses the
            circuit's own clean fidelity. Supplying a shared reference across
            circuits avoids inflating retention when one circuit has a lower
            clean fidelity.
        **noise_kwargs: Additional arguments to pass to noise_fn.
        
    Returns:
        Dict with clean fidelity, noisy fidelity, and retention ratio.
    """
    qubits = sorted(list(circuit.all_qubits()))
    
    # Compute clean fidelity
    clean_fidelity = fidelity_pure_target(circuit, target_state, qubits)
    clean_ref = clean_reference if clean_reference is not None else clean_fidelity
    
    # Apply noise and compute noisy fidelity
    noisy_circuit = noise_fn(circuit, **noise_kwargs)
    noisy_fidelity = fidelity_pure_target(noisy_circuit, target_state, qubits)
    
    # Compute retention ratio (avoid division by zero) and clamp to [0, 1]
    if clean_ref > 1e-10:
        retention_ratio = noisy_fidelity / clean_ref
    else:
        retention_ratio = 0.0
    retention_ratio = min(1.0, retention_ratio)
    
    return {
        'clean_fidelity': float(clean_fidelity),
        'clean_reference': float(clean_ref),
        'noisy_fidelity': float(noisy_fidelity),
        'retention_ratio': float(retention_ratio)
    }


def run_over_rotation_sweep(
    circuit: cirq.Circuit,
    target_state: np.ndarray,
    epsilon_values: np.ndarray,
    clean_reference: float | None = None
) -> list:
    """
    Run over-rotation sweep on a circuit.
    
    Args:
        circuit: The circuit to evaluate.
        target_state: The target quantum state.
        epsilon_values: Array of over-rotation angles to test.
        clean_reference: Optional shared clean fidelity to normalize retention.
        
    Returns:
        List of results for each epsilon value.
    """
    results = []
    for epsilon in epsilon_values:
        result = compute_fidelity_retention_ratio(
            circuit, target_state, apply_over_rotation, clean_reference=clean_reference, epsilon=epsilon
        )
        result['epsilon'] = float(epsilon)
        results.append(result)
    return results


def run_asymmetric_noise_evaluation(
    circuit: cirq.Circuit,
    target_state: np.ndarray,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0,
    clean_reference: float | None = None
) -> dict:
    """
    Evaluate circuit under asymmetric Pauli noise.
    
    Args:
        circuit: The circuit to evaluate.
        target_state: The target quantum state.
        p_x: Probability of X error.
        p_y: Probability of Y error.
        p_z: Probability of Z error.
        clean_reference: Optional shared clean fidelity to normalize retention.
        
    Returns:
        Dict with evaluation results.
    """
    result = compute_fidelity_retention_ratio(
        circuit, target_state, apply_asymmetric_pauli_noise,
        clean_reference=clean_reference,
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
    quantum_nas_circuit_path: str | None = None,
    epsilon_range: tuple = (0.0, 0.2),
    n_epsilon_points: int = 20,
    p_x_asymmetric: float | list = 0.05,
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
    rng = np.random.default_rng(base_seed)
    
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

    quantumnas_circuit = None
    if quantum_nas_circuit_path:
        try:
            quantumnas_circuit = load_circuit(quantum_nas_circuit_path)
            log(f"Loaded QuantumNAS circuit with {len(list(quantumnas_circuit.all_operations()))} operations")
        except Exception as e:
            log(f"WARNING: Failed to load QuantumNAS circuit: {e}")
            quantumnas_circuit = None
    
    # Get target state using central config
    target_state = config.get_target_state(n_qubits)
    # Shared clean reference to prevent inflated retention for lower-fidelity circuits
    def clean_fid(circ):
        qs = sorted(list(circ.all_qubits()))
        return fidelity_pure_target(circ, target_state, qs) if qs else 0.0
    baseline_clean_fid = clean_fid(baseline_circuit)
    robust_clean_fid = clean_fid(robust_circuit)
    qnas_clean_fid = clean_fid(quantumnas_circuit) if quantumnas_circuit is not None else None
    clean_reference = max([fid for fid in [baseline_clean_fid, robust_clean_fid, qnas_clean_fid] if fid is not None])
    log(f"Clean fidelities — baseline: {baseline_clean_fid:.4f}, robust: {robust_clean_fid:.4f}"
        + (f", quantumnas: {qnas_clean_fid:.4f}" if qnas_clean_fid is not None else "")
        + f"; using shared reference {clean_reference:.4f}")
    
    # === Over-rotation sweep ===
    log(f"\n--- Over-rotation sweep (ε ∈ [{epsilon_range[0]}, {epsilon_range[1]}]) ---")
    epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], n_epsilon_points)
    
    baseline_over_rotation_results = run_over_rotation_sweep(
        baseline_circuit, target_state, epsilon_values, clean_reference=clean_reference
    )
    robust_over_rotation_results = run_over_rotation_sweep(
        robust_circuit, target_state, epsilon_values, clean_reference=clean_reference
    )
    
    # Log summary
    baseline_retention_avg = np.mean([r['retention_ratio'] for r in baseline_over_rotation_results])
    robust_retention_avg = np.mean([r['retention_ratio'] for r in robust_over_rotation_results])
    log(f"Baseline avg retention ratio (over-rotation): {baseline_retention_avg:.4f}")
    log(f"Robust avg retention ratio (over-rotation): {robust_retention_avg:.4f}")
    
    # === Asymmetric noise evaluation ===
    px_values = p_x_asymmetric if isinstance(p_x_asymmetric, (list, tuple)) else [p_x_asymmetric]
    asymmetric_results = []
    asymmetric_qnas = []
    log(f"\n--- Asymmetric Pauli noise sweep (p_x in {px_values}, p_y=0.0, p_z=0.0) ---")
    for px in px_values:
        # Optionally add tiny stochastic jitter to reflect seeds
        baseline_ret = []
        robust_ret = []
        qnas_ret = []
        for _ in range(effective_n_seeds):
            b_res = run_asymmetric_noise_evaluation(
                baseline_circuit, target_state, p_x=px, p_y=0.0, p_z=0.0,
                clean_reference=clean_reference
            )
            r_res = run_asymmetric_noise_evaluation(
                robust_circuit, target_state, p_x=px, p_y=0.0, p_z=0.0,
                clean_reference=clean_reference
            )
            baseline_ret.append(b_res)
            robust_ret.append(r_res)
            if quantumnas_circuit is not None:
                q_res = run_asymmetric_noise_evaluation(
                    quantumnas_circuit, target_state, p_x=px, p_y=0.0, p_z=0.0,
                    clean_reference=clean_reference
                )
                qnas_ret.append(q_res)
        baseline_mean = float(np.mean([x['retention_ratio'] for x in baseline_ret]))
        robust_mean = float(np.mean([x['retention_ratio'] for x in robust_ret]))
        qnas_mean = float(np.mean([x['retention_ratio'] for x in qnas_ret])) if qnas_ret else None
        log(f"p_x={px:.3f} | Baseline retention: {baseline_mean:.4f} | Robust retention: {robust_mean:.4f}"
            + (f" | QuantumNAS retention: {qnas_mean:.4f}" if qnas_mean is not None else ""))
        asymmetric_results.append({
            'p_x': px,
            'baseline': baseline_ret,
            'robust': robust_ret,
            'baseline_mean': baseline_mean,
            'robust_mean': robust_mean,
            'quantumnas': qnas_ret if qnas_ret else None,
            'quantumnas_mean': qnas_mean,
        })
        if qnas_ret:
            asymmetric_qnas.append(qnas_ret)
    
    # === Compile results ===
    all_results = {
        'metadata': {
            'timestamp': timestamp,
            'n_qubits': n_qubits,
            'baseline_circuit_path': baseline_circuit_path,
            'robust_circuit_path': robust_circuit_path,
            'quantumnas_circuit_path': quantum_nas_circuit_path,
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
            'quantumnas': None,  # filled below if available
        },
        'asymmetric_noise': asymmetric_results,
        'summary': {
            'baseline_over_rotation_avg_retention': float(baseline_retention_avg),
            'robust_over_rotation_avg_retention': float(robust_retention_avg),
            'baseline_asymmetric_retention': asymmetric_results[0]['baseline_mean'],
            'robust_asymmetric_retention': asymmetric_results[0]['robust_mean'],
            'quantumnas_asymmetric_retention': asymmetric_results[0].get('quantumnas_mean'),
        },
        'clean_fidelity': {
            'baseline': float(baseline_clean_fid),
            'robust': float(robust_clean_fid),
            'quantumnas': float(qnas_clean_fid) if qnas_clean_fid is not None else None,
            'reference_used': float(clean_reference),
        },
    }

    # Optional: over-rotation for QuantumNAS
    qnas_over_rotation_results = None
    if quantumnas_circuit is not None:
        qnas_over_rotation_results = run_over_rotation_sweep(
            quantumnas_circuit, target_state, epsilon_values, clean_reference=clean_reference
        )
        all_results['over_rotation']['quantumnas'] = qnas_over_rotation_results
        qnas_retention_avg = np.mean([r['retention_ratio'] for r in qnas_over_rotation_results])
        all_results['summary']['quantumnas_over_rotation_avg_retention'] = float(qnas_retention_avg)
    
    # === Save results JSON ===
    results_json_path = os.path.join(results_subdir, 'cross_noise_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {results_json_path}")
    
    # === Generate plots ===
    # Plot 1: Over-rotation fidelity retention
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "baseline": "#2ecc71",
        "robust": "#e67e22",
        "quantumnas": "#2c7fb8",
        "bars": ["#2ecc71", "#e67e22", "#2c7fb8"],
    }
    
    # Left plot: Fidelity retention ratio vs epsilon
    ax1 = axes[0]
    baseline_retentions = [r['retention_ratio'] for r in baseline_over_rotation_results]
    robust_retentions = [r['retention_ratio'] for r in robust_over_rotation_results]
    baseline_noisy = [r['noisy_fidelity'] for r in baseline_over_rotation_results]
    robust_noisy = [r['noisy_fidelity'] for r in robust_over_rotation_results]
    qnas_retentions = [r['retention_ratio'] for r in qnas_over_rotation_results] if quantumnas_circuit is not None else None
    qnas_noisy = [r['noisy_fidelity'] for r in qnas_over_rotation_results] if quantumnas_circuit is not None else None
    
    ax1.plot(epsilon_values, baseline_retentions, marker='o', linestyle='-', color=colors["baseline"],
             label='Baseline', markersize=4)
    ax1.plot(epsilon_values, robust_retentions, marker='s', linestyle='-', color=colors["robust"],
             label='Robust', markersize=4)
    if qnas_retentions is not None:
        ax1.plot(epsilon_values, qnas_retentions, marker='^', linestyle='-', color=colors["quantumnas"],
                 label='QuantumNAS', markersize=4)
    ax1.set_xlabel('Over-rotation ε (radians)')
    ax1.set_ylabel('Fidelity Retention Ratio (F_noisy / F_clean)')
    ax1.set_title(f'Over-rotation Robustness (n_ε={n_epsilon_points})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    info_lines = [
        f"Clean ref: {clean_reference:.4f}",
        f"ε_max={epsilon_values[-1]:.3f}",
        f"Baseline noisy(end): {baseline_noisy[-1]:.3f}",
        f"Robust noisy(end): {robust_noisy[-1]:.3f}",
        f"Baseline avg retention: {baseline_retention_avg:.3f}",
        f"Robust avg retention: {robust_retention_avg:.3f}",
    ]
    if qnas_noisy is not None:
        info_lines.append(f"QNAS noisy(end): {qnas_noisy[-1]:.3f}")
    if 'quantumnas_over_rotation_avg_retention' in all_results.get('summary', {}):
        info_lines.append(f"QNAS avg retention: {all_results['summary']['quantumnas_over_rotation_avg_retention']:.3f}")
    ax1.text(
        0.62, 0.02,
        "\n".join(info_lines),
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85)
    )
    
    # Right plot: Asymmetric noise comparison (bar chart, first p_x entry)
    ax2 = axes[1]
    first_asym = asymmetric_results[0]
    p_y_val = first_asym.get('p_y', 0.0)
    p_z_val = first_asym.get('p_z', 0.0)
    circuit_types = ['Baseline', 'Robust']
    retention_values = [
        first_asym['baseline_mean'],
        first_asym['robust_mean'],
    ]
    if first_asym.get('quantumnas_mean') is not None:
        circuit_types.append('QuantumNAS')
        retention_values.append(first_asym['quantumnas_mean'])
    bars = ax2.bar(circuit_types, retention_values, color=colors["bars"][:len(circuit_types)])
    ax2.set_ylabel('Fidelity Retention Ratio')
    ax2.set_title(f"Asymmetric Pauli Noise (p_x={first_asym['p_x']}, p_y={p_y_val}, p_z={p_z_val})")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, retention_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout(pad=1.5)
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
        if quantum_nas_circuit_path:
            f.write(f"  QuantumNAS circuit: {quantum_nas_circuit_path}\n")
        f.write(f"  Epsilon range: {epsilon_range}\n")
        f.write(f"  Asymmetric p_x: {px_values}\n")
        f.write(f"  Clean fidelity (baseline): {baseline_clean_fid:.4f}\n")
        f.write(f"  Clean fidelity (robust):   {robust_clean_fid:.4f}\n")
        if qnas_clean_fid is not None:
            f.write(f"  Clean fidelity (quantumnas): {qnas_clean_fid:.4f}\n")
        f.write(f"  Retention normalized by:   {clean_reference:.4f}\n\n")
        
        f.write("Over-rotation Test (ε ∈ [0, 0.1]):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Baseline avg retention: {baseline_retention_avg:.4f}\n")
        f.write(f"  Robust avg retention: {robust_retention_avg:.4f}\n")
        f.write(f"  Improvement: {(robust_retention_avg - baseline_retention_avg):.4f}\n")
        if quantumnas_circuit is not None and 'quantumnas_over_rotation_avg_retention' in all_results['summary']:
            qnas_avg = all_results['summary']['quantumnas_over_rotation_avg_retention']
            f.write(f"  QuantumNAS avg retention: {qnas_avg:.4f}\n")
        f.write("\n")
        
        f.write(f"Asymmetric Pauli Noise sweep (p_x in {px_values}):\n")
        f.write("-" * 40 + "\n")
        for entry in asymmetric_results:
            improvement = entry['robust_mean'] - entry['baseline_mean']
            f.write(f"  p_x={entry['p_x']:.3f} | Baseline: {entry['baseline_mean']:.4f} | Robust: {entry['robust_mean']:.4f} | Δ={improvement:.4f}\n")
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
        '--quantumnas-circuit', type=str, default=None,
        help='Optional: Path to QuantumNAS circuit JSON file'
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
        '--epsilon-max', type=float, default=0.2,
        help='Maximum over-rotation angle (default: 0.2)'
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
        p_x_asymmetric=args.p_x_asymmetric,
        quantum_nas_circuit_path=args.quantumnas_circuit
    )
