#!/usr/bin/env python3
"""
Parameter Recovery Test Experiment.

This script automates parameter recovery for both baseline and adversarial (robust) circuits.
It tests how well we can recover the true noise parameter from measurement statistics
under different depolarizing noise rates.

Statistical Protocol:
    - Number of seeds: Configurable via n_seeds parameter (default: config.N_SEEDS)
    - Stochastic elements: All noise seeds are logged for reproducibility
    - Results include mean ± std with error bars on plots
    - Summary file includes all seeds, noise parameters, and hyperparameters

Experiment workflow:
1. Load baseline and robust circuits
2. For each circuit, simulate noisy measurement statistics across noise rates
3. For each p, use MLE to recover the noise parameter (with multiple seeds)
4. Plot recovered vs true p with error bars (diagonal = perfect recovery)
5. Save JSON results, summary files, and plots

This experiment helps validate the practical utility of circuits in quantum metrology.
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
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize_scalar

import cirq

from qas_gym.utils import get_ghz_state, load_circuit, fidelity_pure_target

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


# Default noise rates to test (per problem statement)
DEFAULT_P_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1]

# Number of measurement shots for statistical estimation
DEFAULT_N_SHOTS = 1000

# Number of repetitions for each p value to get mean/std estimates
DEFAULT_N_REPETITIONS = 10

# Small epsilon to avoid division by zero in variance calculations
NOISE_VARIANCE_EPSILON = 1e-8


def apply_depolarizing_noise_to_circuit(circuit: cirq.Circuit, p: float) -> cirq.Circuit:
    """
    Apply uniform depolarizing noise after each gate in the circuit.
    
    Args:
        circuit: The input circuit.
        p: Depolarizing noise probability (0 <= p <= 1).
    
    Returns:
        A new circuit with depolarizing channels inserted after each gate.
    """
    if p <= 0:
        return circuit
    
    noisy_ops = []
    for op in circuit.all_operations():
        noisy_ops.append(op)
        # Add depolarizing noise to each qubit involved in the gate
        for q in op.qubits:
            noisy_ops.append(cirq.DepolarizingChannel(p).on(q))
    
    return cirq.Circuit(noisy_ops)


def simulate_noisy_fidelity(circuit: cirq.Circuit, target_state: np.ndarray, 
                            p: float, n_qubits: int) -> float:
    """
    Simulate the fidelity of a circuit under depolarizing noise.
    
    Args:
        circuit: The circuit to simulate.
        target_state: The target quantum state.
        p: Depolarizing noise probability.
        n_qubits: Number of qubits.
    
    Returns:
        Fidelity of the noisy circuit output with respect to the target state.
    """
    noisy_circuit = apply_depolarizing_noise_to_circuit(circuit, p)
    qubits = cirq.LineQubit.range(n_qubits)
    return fidelity_pure_target(noisy_circuit, target_state, qubits)


def generate_measurement_samples(circuit: cirq.Circuit, target_state: np.ndarray,
                                 true_p: float, n_qubits: int, n_shots: int) -> list:
    """
    Generate measurement fidelity samples under noise.
    
    For simplicity, we simulate the expected fidelity for multiple "shots" 
    with small noise perturbations to model measurement statistics.
    
    Args:
        circuit: The circuit to simulate.
        target_state: The target quantum state.
        true_p: The true depolarizing noise parameter.
        n_qubits: Number of qubits.
        n_shots: Number of measurement shots/samples.
    
    Returns:
        List of fidelity samples.
    """
    # Get the expected fidelity under noise
    expected_fid = simulate_noisy_fidelity(circuit, target_state, true_p, n_qubits)
    
    # Model measurement noise as binomial variance in fidelity estimation
    # The variance scales as fid * (1 - fid) / n_shots for a single measurement
    # We add small Gaussian noise to model shot noise
    std_dev = np.sqrt(expected_fid * (1 - expected_fid) / n_shots + NOISE_VARIANCE_EPSILON)
    
    # Generate samples centered around expected fidelity
    samples = np.random.normal(expected_fid, std_dev, n_shots)
    # Clip to valid fidelity range
    samples = np.clip(samples, 0.0, 1.0)
    
    return samples.tolist()


def fidelity_model(p: float, circuit: cirq.Circuit, target_state: np.ndarray, 
                   n_qubits: int) -> float:
    """
    Model function that computes expected fidelity for a given noise parameter p.
    
    Args:
        p: Depolarizing noise parameter to test.
        circuit: The circuit.
        target_state: Target state.
        n_qubits: Number of qubits.
    
    Returns:
        Expected fidelity.
    """
    return simulate_noisy_fidelity(circuit, target_state, p, n_qubits)


def mle_recover_noise_parameter(observed_fidelity: float, circuit: cirq.Circuit,
                                target_state: np.ndarray, n_qubits: int,
                                p_min: float = 0.0, p_max: float = 0.15) -> float:
    """
    Recover the noise parameter using Maximum Likelihood Estimation.
    
    Given an observed fidelity, find the noise parameter p that would produce
    this fidelity according to our model. This is essentially inverting the
    fidelity-vs-noise curve.
    
    Args:
        observed_fidelity: The observed/measured fidelity.
        circuit: The circuit used.
        target_state: Target quantum state.
        n_qubits: Number of qubits.
        p_min: Minimum p to search.
        p_max: Maximum p to search.
    
    Returns:
        Estimated noise parameter p.
    """
    def objective(p):
        model_fid = fidelity_model(p, circuit, target_state, n_qubits)
        # Minimize squared error between model and observed fidelity
        return (model_fid - observed_fidelity) ** 2
    
    # Use bounded scalar optimization
    result = minimize_scalar(objective, bounds=(p_min, p_max), method='bounded')
    return result.x


def run_parameter_recovery_for_circuit(circuit: cirq.Circuit, circuit_name: str,
                                       target_state: np.ndarray, n_qubits: int,
                                       p_values: list, n_shots: int,
                                       n_repetitions: int, logger=None,
                                       base_seed: int = 0,
                                       save_dir: str = None) -> dict:
    """
    Run parameter recovery experiment for a single circuit with multi-seed support.
    
    Args:
        circuit: The circuit to test.
        circuit_name: Name for logging (e.g., 'baseline', 'robust').
        target_state: Target quantum state.
        n_qubits: Number of qubits.
        p_values: List of true noise parameters to test.
        n_shots: Number of measurement shots per trial.
        n_repetitions: Number of repetitions per p value (seeds).
        logger: Optional logger.
        base_seed: Base seed for reproducibility.
        save_dir: Optional directory to save per-seed results.
    
    Returns:
        Dict with results for each p value, including per-seed data.
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # Track all noise seeds used
    noise_seeds_used = []
    
    results = {
        'true_p': [],
        'recovered_p_mean': [],
        'recovered_p_std': [],
        'observed_fidelity_mean': [],
        'observed_fidelity_std': [],
        'recovery_error_mean': [],
        'recovery_error_std': [],
        'per_seed_data': [],  # New: detailed per-seed results
        'noise_seeds': [],  # New: log all noise seeds
    }
    
    log(f"\n  Testing {circuit_name} circuit (n_repetitions={n_repetitions}):")
    
    for true_p in p_values:
        recovered_p_list = []
        observed_fid_list = []
        seed_results = []
        p_noise_seeds = []
        
        # Get index of this p value for seed offset
        p_idx = p_values.index(true_p) if true_p in p_values else 0
        
        for rep in range(n_repetitions):
            # Use hash-based seed generation to avoid collisions
            # Format: base_seed * 1000000 + p_index * 1000 + rep
            noise_seed = base_seed * 1000000 + p_idx * 1000 + rep
            np.random.seed(noise_seed)
            p_noise_seeds.append(noise_seed)
            
            # Generate measurement samples
            samples = generate_measurement_samples(circuit, target_state, true_p, 
                                                   n_qubits, n_shots)
            observed_fid = np.mean(samples)
            observed_fid_list.append(observed_fid)
            
            # Recover noise parameter using MLE
            recovered_p = mle_recover_noise_parameter(observed_fid, circuit, 
                                                      target_state, n_qubits)
            recovered_p_list.append(recovered_p)
            
            # Store per-seed result
            seed_results.append({
                'seed': rep,
                'noise_seed': noise_seed,
                'recovered_p': float(recovered_p),
                'observed_fidelity': float(observed_fid),
                'recovery_error': float(recovered_p - true_p),
            })
        
        noise_seeds_used.extend(p_noise_seeds)
        
        # Compute aggregated statistics using proper functions
        recovered_stats = aggregate_metrics(recovered_p_list)
        fidelity_stats = aggregate_metrics(observed_fid_list)
        recovery_errors = [r - true_p for r in recovered_p_list]
        error_stats = aggregate_metrics(recovery_errors)
        
        results['true_p'].append(float(true_p))
        results['recovered_p_mean'].append(recovered_stats['mean'])
        results['recovered_p_std'].append(recovered_stats['std'])
        results['observed_fidelity_mean'].append(fidelity_stats['mean'])
        results['observed_fidelity_std'].append(fidelity_stats['std'])
        results['recovery_error_mean'].append(error_stats['mean'])
        results['recovery_error_std'].append(error_stats['std'])
        results['per_seed_data'].append(seed_results)
        results['noise_seeds'].append(p_noise_seeds)
        
        log(f"    p={true_p:.3f}: recovered={recovered_stats['mean']:.4f}±{recovered_stats['std']:.4f} (n={n_repetitions}), "
            f"fidelity={fidelity_stats['mean']:.4f}±{fidelity_stats['std']:.4f}")
    
    # Save per-circuit results if directory provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        circuit_file = os.path.join(save_dir, f'{circuit_name}_recovery_results.json')
        with open(circuit_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def create_recovery_plot(baseline_results: dict, robust_results: dict, 
                         output_path: str, n_qubits: int, n_repetitions: int = None):
    """
    Create and save the parameter recovery plot with error bars and annotations.
    
    Plots recovered p vs true p for both baseline and robust circuits,
    with a diagonal line representing perfect recovery.
    Shows error bars (mean ± std) and sample size annotation.
    Overlays individual seed results as faint points.
    
    Args:
        baseline_results: Results dict for baseline circuit.
        robust_results: Results dict for robust circuit.
        output_path: Path to save the plot.
        n_qubits: Number of qubits (for title).
        n_repetitions: Number of seeds (for annotation).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get data
    true_p = baseline_results['true_p']
    n_seeds = n_repetitions or len(baseline_results.get('per_seed_data', [[]])[0])
    
    # Plot individual seed data as faint points (if available)
    if 'per_seed_data' in baseline_results:
        for i, (p, seed_data) in enumerate(zip(true_p, baseline_results['per_seed_data'])):
            rec_ps = [sd['recovered_p'] for sd in seed_data]
            ax.scatter([p] * len(rec_ps), rec_ps, alpha=0.2, s=20, color='tab:blue')
    
    if 'per_seed_data' in robust_results:
        for i, (p, seed_data) in enumerate(zip(true_p, robust_results['per_seed_data'])):
            rec_ps = [sd['recovered_p'] for sd in seed_data]
            ax.scatter([p] * len(rec_ps), rec_ps, alpha=0.2, s=20, color='tab:orange')
    
    # Plot baseline results with error bars
    ax.errorbar(true_p, baseline_results['recovered_p_mean'],
                yerr=baseline_results['recovered_p_std'],
                fmt='o-', capsize=5, capthick=2, label='Baseline Circuit', 
                color='tab:blue', markersize=8, linewidth=2)
    
    # Plot robust results with error bars
    ax.errorbar(true_p, robust_results['recovered_p_mean'],
                yerr=robust_results['recovered_p_std'],
                fmt='s-', capsize=5, capthick=2, label='Robust Circuit',
                color='tab:orange', markersize=8, linewidth=2)
    
    # Plot perfect recovery line (diagonal)
    p_range = [0, max(true_p) * 1.1]
    ax.plot(p_range, p_range, 'k--', alpha=0.5, label='Perfect Recovery', linewidth=2)
    
    ax.set_xlabel('True Noise Parameter (p)', fontsize=12)
    ax.set_ylabel('Recovered Noise Parameter (p)', fontsize=12)
    ax.set_title(f'Parameter Recovery: {n_qubits}-Qubit Circuits (n={n_seeds} seeds)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(p_range)
    ax.set_ylim(p_range)
    
    # Make plot square for better visualization of diagonal
    ax.set_aspect('equal', adjustable='box')
    
    # Add annotation about error bars
    ax.text(0.02, 0.98, f'Error bars: mean ± std', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_parameter_recovery(results_dir: str = None, n_qubits: int = 4,
                           baseline_circuit_path: str = None,
                           robust_circuit_path: str = None,
                           p_values: list = None, n_shots: int = DEFAULT_N_SHOTS,
                           n_repetitions: int = None,
                           base_seed: int = 42,
                           logger=None) -> dict:
    """
    Run the full parameter recovery experiment with statistical reporting.
    
    Statistical Protocol:
        - n_repetitions seeds per noise parameter (default: config.N_SEEDS)
        - All noise seeds are logged for reproducibility
        - Results include mean ± std with error bars on plots
        - Summary file includes all seeds, noise parameters, and hyperparameters
    
    Args:
        results_dir: Directory to save results. Creates timestamped subdir if None.
        n_qubits: Number of qubits.
        baseline_circuit_path: Path to baseline circuit JSON.
        robust_circuit_path: Path to robust circuit JSON.
        p_values: List of noise parameters to test.
        n_shots: Number of measurement shots per trial.
        n_repetitions: Number of repetitions/seeds per p value.
        base_seed: Base seed for reproducibility.
        logger: Optional logger.
    
    Returns:
        Dict with results for both circuits and summary statistics.
    """
    # Use config default if not specified
    effective_n_repetitions = n_repetitions if n_repetitions is not None else config.N_SEEDS
    
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    if p_values is None:
        p_values = DEFAULT_P_VALUES
    
    # Set up results directory
    if results_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        results_dir = f"results/parameter_recovery_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    log(f"=== Parameter Recovery Experiment ===")
    log(f"Qubits: {n_qubits}")
    log(f"Noise parameters to test: {p_values}")
    log(f"Shots per trial: {n_shots}")
    log(f"Repetitions per p (seeds): {effective_n_repetitions}")
    log(f"Base seed: {base_seed}")
    log(f"Results directory: {results_dir}")
    
    # Get target state
    target_state = get_ghz_state(n_qubits)
    
    # Load circuits
    # Try to find circuits in expected locations
    if baseline_circuit_path is None:
        # Look in parent directory structure
        parent_dir = os.path.dirname(results_dir.rstrip('/'))
        baseline_circuit_path = os.path.join(parent_dir, 'baseline', 'circuit_vanilla.json')
    
    if robust_circuit_path is None:
        import re
        parent_dir = os.path.dirname(results_dir.rstrip('/'))
        robust_circuit_path = os.path.join(parent_dir, 'adversarial', 'circuit_robust.json')
        # Also try adversarial_training_* subdirs
        if not os.path.exists(robust_circuit_path):
            from glob import glob
            adv_dir = os.path.join(parent_dir, 'adversarial')
            if os.path.isdir(adv_dir):
                adv_subdirs = glob(os.path.join(adv_dir, 'adversarial_training_*'))
                if adv_subdirs:
                    # Sort by timestamp in folder name (matching run_experiments.py)
                    adv_subdirs.sort(key=lambda x: re.findall(r'adversarial_training_(\d+)-(\d+)', x)[0] if re.findall(r'adversarial_training_(\d+)-(\d+)', x) else ('', ''))
                    robust_circuit_path = os.path.join(adv_subdirs[-1], 'circuit_robust.json')
    
    # Load baseline circuit
    baseline_circuit = None
    if os.path.exists(baseline_circuit_path):
        try:
            baseline_circuit = load_circuit(baseline_circuit_path)
            log(f"Loaded baseline circuit from {baseline_circuit_path}")
        except Exception as e:
            log(f"Warning: Could not load baseline circuit: {e}")
    else:
        log(f"Warning: Baseline circuit not found at {baseline_circuit_path}")
    
    # Load robust circuit
    robust_circuit = None
    if os.path.exists(robust_circuit_path):
        try:
            robust_circuit = load_circuit(robust_circuit_path)
            log(f"Loaded robust circuit from {robust_circuit_path}")
        except Exception as e:
            log(f"Warning: Could not load robust circuit: {e}")
    else:
        log(f"Warning: Robust circuit not found at {robust_circuit_path}")
    
    # Check if we have at least one circuit to test
    if baseline_circuit is None and robust_circuit is None:
        log("Error: No circuits available for testing. Run baseline and adversarial training first.")
        return None
    
    all_results = {
        'n_qubits': n_qubits,
        'p_values': p_values,
        'n_shots': n_shots,
        'n_repetitions': effective_n_repetitions,
        'base_seed': base_seed,
        'baseline': None,
        'robust': None,
    }
    
    # Collect all noise seeds used
    all_noise_seeds = []
    
    # Run parameter recovery for baseline circuit
    if baseline_circuit is not None:
        baseline_results = run_parameter_recovery_for_circuit(
            baseline_circuit, 'baseline', target_state, n_qubits,
            p_values, n_shots, effective_n_repetitions, logger,
            base_seed=base_seed,
            save_dir=results_dir
        )
        all_results['baseline'] = baseline_results
        if 'noise_seeds' in baseline_results:
            all_noise_seeds.extend([s for seeds in baseline_results['noise_seeds'] for s in seeds])
    else:
        log("Skipping baseline circuit (not available)")
    
    # Run parameter recovery for robust circuit
    if robust_circuit is not None:
        robust_results = run_parameter_recovery_for_circuit(
            robust_circuit, 'robust', target_state, n_qubits,
            p_values, n_shots, effective_n_repetitions, logger,
            base_seed=base_seed + 10000,  # Offset for robust circuit
            save_dir=results_dir
        )
        all_results['robust'] = robust_results
        if 'noise_seeds' in robust_results:
            all_noise_seeds.extend([s for seeds in robust_results['noise_seeds'] for s in seeds])
    else:
        log("Skipping robust circuit (not available)")
    
    # Save JSON results
    results_file = os.path.join(results_dir, 'parameter_recovery_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {results_file}")
    
    # Create plot if both circuits available
    if all_results['baseline'] is not None and all_results['robust'] is not None:
        plot_path = os.path.join(results_dir, 'parameter_recovery_plot.png')
        create_recovery_plot(all_results['baseline'], all_results['robust'], 
                            plot_path, n_qubits, n_repetitions=effective_n_repetitions)
        log(f"Plot saved to {plot_path}")
    elif all_results['baseline'] is not None or all_results['robust'] is not None:
        # Create single-circuit plot with error bars
        plot_path = os.path.join(results_dir, 'parameter_recovery_plot.png')
        available_results = all_results['baseline'] if all_results['baseline'] else all_results['robust']
        circuit_name = 'baseline' if all_results['baseline'] else 'robust'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        true_p = available_results['true_p']
        
        # Plot individual seed data as faint points (if available)
        if 'per_seed_data' in available_results:
            for i, (p, seed_data) in enumerate(zip(true_p, available_results['per_seed_data'])):
                rec_ps = [sd['recovered_p'] for sd in seed_data]
                ax.scatter([p] * len(rec_ps), rec_ps, alpha=0.2, s=20, color='tab:blue')
        
        ax.errorbar(true_p, available_results['recovered_p_mean'],
                    yerr=available_results['recovered_p_std'],
                    fmt='o-', capsize=5, capthick=2, label=f'{circuit_name.capitalize()} Circuit',
                    markersize=8, linewidth=2)
        
        p_range = [0, max(true_p) * 1.1]
        ax.plot(p_range, p_range, 'k--', alpha=0.5, label='Perfect Recovery', linewidth=2)
        
        ax.set_xlabel('True Noise Parameter (p)', fontsize=12)
        ax.set_ylabel('Recovered Noise Parameter (p)', fontsize=12)
        ax.set_title(f'Parameter Recovery: {n_qubits}-Qubit {circuit_name.capitalize()} Circuit (n={effective_n_repetitions})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(p_range)
        ax.set_ylim(p_range)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        log(f"Plot saved to {plot_path}")
    
    # Create experiment summary using statistical utilities
    hyperparameters = {
        'n_qubits': n_qubits,
        'p_values': p_values,
        'n_shots': n_shots,
        'n_repetitions': effective_n_repetitions,
        'base_seed': base_seed,
    }
    
    # Aggregate recovery error stats
    aggregated_results = {}
    for circuit_type in ['baseline', 'robust']:
        if all_results[circuit_type] is not None:
            r = all_results[circuit_type]
            recovery_errors = r['recovery_error_mean']
            aggregated_results[f'{circuit_type}_mean_abs_error'] = aggregate_metrics([abs(e) for e in recovery_errors])
    
    summary = create_experiment_summary(
        experiment_name='parameter_recovery',
        n_seeds=effective_n_repetitions,
        seeds_used=list(range(effective_n_repetitions)),
        hyperparameters=hyperparameters,
        aggregated_results=aggregated_results,
        noise_seeds=all_noise_seeds if all_noise_seeds else None,
        noise_parameters={'p_values': p_values, 'depolarizing_noise': True},
        commit_hash=get_git_commit_hash(),
        additional_notes='Parameter recovery experiment testing noise estimation accuracy.'
    )
    
    save_experiment_summary(summary, results_dir, 'experiment_summary.json')
    
    # Create summary statistics text file
    summary_file = os.path.join(results_dir, 'parameter_recovery_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Parameter Recovery Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write("Statistical Protocol:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Number of seeds per p value: {effective_n_repetitions}\n")
        f.write(f"  Base seed: {base_seed}\n")
        f.write(f"  Aggregation: mean ± std (sample std, ddof=1)\n\n")
        f.write("Hyperparameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Qubits: {n_qubits}\n")
        f.write(f"  Noise parameters tested: {p_values}\n")
        f.write(f"  Shots per trial: {n_shots}\n\n")
        
        for circuit_type in ['baseline', 'robust']:
            if all_results[circuit_type] is not None:
                r = all_results[circuit_type]
                f.write(f"\n{circuit_type.upper()} Circuit Results:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'True p':<10} {'Recovered p (mean±std)':<25} {'Error':<15} n\n")
                for i, p in enumerate(r['true_p']):
                    rec_p = r['recovered_p_mean'][i]
                    rec_std = r['recovered_p_std'][i]
                    error = r['recovery_error_mean'][i]
                    f.write(f"{p:<10.4f} {rec_p:.4f}±{rec_std:.4f}              {error:+.4f}         {effective_n_repetitions}\n")
                
                # Overall statistics
                mean_abs_error = np.mean(np.abs(r['recovery_error_mean']))
                f.write(f"\nMean Absolute Recovery Error: {mean_abs_error:.5f}\n")
    
    log(f"Summary saved to {summary_file}")
    log("\n=== Parameter Recovery Experiment Complete ===")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameter Recovery Test: Recover noise parameter from measurement statistics"
    )
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory to save results (default: results/parameter_recovery_<timestamp>)')
    parser.add_argument('--n-qubits', type=int, default=4,
                        help='Number of qubits (default: 4)')
    parser.add_argument('--baseline-circuit', type=str, default=None,
                        help='Path to baseline circuit JSON')
    parser.add_argument('--robust-circuit', type=str, default=None,
                        help='Path to robust circuit JSON')
    parser.add_argument('--n-shots', type=int, default=DEFAULT_N_SHOTS,
                        help=f'Number of measurement shots (default: {DEFAULT_N_SHOTS})')
    parser.add_argument('--n-repetitions', type=int, default=DEFAULT_N_REPETITIONS,
                        help=f'Number of repetitions per p value (default: {DEFAULT_N_REPETITIONS})')
    
    args = parser.parse_args()
    
    run_parameter_recovery(
        results_dir=args.results_dir,
        n_qubits=args.n_qubits,
        baseline_circuit_path=args.baseline_circuit,
        robust_circuit_path=args.robust_circuit,
        n_shots=args.n_shots,
        n_repetitions=args.n_repetitions,
    )
