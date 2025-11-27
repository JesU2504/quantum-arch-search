#!/usr/bin/env python3
"""
Parameter Recovery Test Experiment.

This script automates parameter recovery for both baseline and adversarial (robust) circuits.
It tests how well we can recover the true noise parameter from measurement statistics
under different depolarizing noise rates.

Experiment workflow:
1. Load baseline (trained on depolarizing noise) and robust (trained with adversarial saboteur) circuits
2. For each circuit, simulate noisy measurement statistics across a range of depolarizing noise rates
3. For each p, use MLE (Maximum Likelihood Estimation) to recover the noise parameter
4. Plot recovered vs true p for both circuits (diagonal = perfect recovery)
5. Save JSON results and PNG plot to results directory

This experiment helps validate the practical utility of circuits in quantum metrology applications,
where estimating physical noise parameters from measurement data is important.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize_scalar

import cirq

from qas_gym.utils import get_ghz_state, load_circuit, fidelity_pure_target


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
                                       n_repetitions: int, logger=None) -> dict:
    """
    Run parameter recovery experiment for a single circuit.
    
    Args:
        circuit: The circuit to test.
        circuit_name: Name for logging (e.g., 'baseline', 'robust').
        target_state: Target quantum state.
        n_qubits: Number of qubits.
        p_values: List of true noise parameters to test.
        n_shots: Number of measurement shots per trial.
        n_repetitions: Number of repetitions per p value.
        logger: Optional logger.
    
    Returns:
        Dict with results for each p value.
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    results = {
        'true_p': [],
        'recovered_p_mean': [],
        'recovered_p_std': [],
        'observed_fidelity_mean': [],
        'observed_fidelity_std': [],
        'recovery_error_mean': [],
        'recovery_error_std': [],
    }
    
    log(f"\n  Testing {circuit_name} circuit:")
    
    for true_p in p_values:
        recovered_p_list = []
        observed_fid_list = []
        
        for _ in range(n_repetitions):
            # Generate measurement samples
            samples = generate_measurement_samples(circuit, target_state, true_p, 
                                                   n_qubits, n_shots)
            observed_fid = np.mean(samples)
            observed_fid_list.append(observed_fid)
            
            # Recover noise parameter using MLE
            recovered_p = mle_recover_noise_parameter(observed_fid, circuit, 
                                                      target_state, n_qubits)
            recovered_p_list.append(recovered_p)
        
        recovered_p_mean = np.mean(recovered_p_list)
        recovered_p_std = np.std(recovered_p_list)
        observed_fid_mean = np.mean(observed_fid_list)
        observed_fid_std = np.std(observed_fid_list)
        recovery_error = np.array(recovered_p_list) - true_p
        
        results['true_p'].append(true_p)
        results['recovered_p_mean'].append(recovered_p_mean)
        results['recovered_p_std'].append(recovered_p_std)
        results['observed_fidelity_mean'].append(observed_fid_mean)
        results['observed_fidelity_std'].append(observed_fid_std)
        results['recovery_error_mean'].append(np.mean(recovery_error))
        results['recovery_error_std'].append(np.std(recovery_error))
        
        log(f"    p={true_p:.3f}: recovered={recovered_p_mean:.4f}±{recovered_p_std:.4f}, "
            f"fidelity={observed_fid_mean:.4f}±{observed_fid_std:.4f}")
    
    return results


def create_recovery_plot(baseline_results: dict, robust_results: dict, 
                         output_path: str, n_qubits: int):
    """
    Create and save the parameter recovery plot.
    
    Plots recovered p vs true p for both baseline and robust circuits,
    with a diagonal line representing perfect recovery.
    
    Args:
        baseline_results: Results dict for baseline circuit.
        robust_results: Results dict for robust circuit.
        output_path: Path to save the plot.
        n_qubits: Number of qubits (for title).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get data
    true_p = baseline_results['true_p']
    
    # Plot baseline results
    ax.errorbar(true_p, baseline_results['recovered_p_mean'],
                yerr=baseline_results['recovered_p_std'],
                fmt='o-', capsize=4, label='Baseline Circuit', 
                color='tab:blue', markersize=8)
    
    # Plot robust results
    ax.errorbar(true_p, robust_results['recovered_p_mean'],
                yerr=robust_results['recovered_p_std'],
                fmt='s-', capsize=4, label='Robust Circuit',
                color='tab:orange', markersize=8)
    
    # Plot perfect recovery line (diagonal)
    p_range = [0, max(true_p) * 1.1]
    ax.plot(p_range, p_range, 'k--', alpha=0.5, label='Perfect Recovery', linewidth=2)
    
    ax.set_xlabel('True Noise Parameter (p)', fontsize=12)
    ax.set_ylabel('Recovered Noise Parameter (p)', fontsize=12)
    ax.set_title(f'Parameter Recovery: {n_qubits}-Qubit Circuits', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(p_range)
    ax.set_ylim(p_range)
    
    # Make plot square for better visualization of diagonal
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_parameter_recovery(results_dir: str = None, n_qubits: int = 4,
                           baseline_circuit_path: str = None,
                           robust_circuit_path: str = None,
                           p_values: list = None, n_shots: int = DEFAULT_N_SHOTS,
                           n_repetitions: int = DEFAULT_N_REPETITIONS,
                           logger=None) -> dict:
    """
    Run the full parameter recovery experiment.
    
    Args:
        results_dir: Directory to save results. Creates timestamped subdir if None.
        n_qubits: Number of qubits. Default is 4 (per problem statement), but when called
                  from run_experiments.py, it uses args.n_qubits (default 3 there).
        baseline_circuit_path: Path to baseline circuit JSON. If None, looks in results_dir parent.
        robust_circuit_path: Path to robust circuit JSON. If None, looks in results_dir parent.
        p_values: List of noise parameters to test. Uses default if None.
        n_shots: Number of measurement shots per trial.
        n_repetitions: Number of repetitions per p value.
        logger: Optional logger.
    
    Returns:
        Dict with results for both circuits and summary statistics.
    """
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
    log(f"Repetitions per p: {n_repetitions}")
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
        'n_repetitions': n_repetitions,
        'baseline': None,
        'robust': None,
    }
    
    # Run parameter recovery for baseline circuit
    if baseline_circuit is not None:
        baseline_results = run_parameter_recovery_for_circuit(
            baseline_circuit, 'baseline', target_state, n_qubits,
            p_values, n_shots, n_repetitions, logger
        )
        all_results['baseline'] = baseline_results
    else:
        log("Skipping baseline circuit (not available)")
    
    # Run parameter recovery for robust circuit
    if robust_circuit is not None:
        robust_results = run_parameter_recovery_for_circuit(
            robust_circuit, 'robust', target_state, n_qubits,
            p_values, n_shots, n_repetitions, logger
        )
        all_results['robust'] = robust_results
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
                            plot_path, n_qubits)
        log(f"Plot saved to {plot_path}")
    elif all_results['baseline'] is not None or all_results['robust'] is not None:
        # Create single-circuit plot
        plot_path = os.path.join(results_dir, 'parameter_recovery_plot.png')
        available_results = all_results['baseline'] if all_results['baseline'] else all_results['robust']
        circuit_name = 'baseline' if all_results['baseline'] else 'robust'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        true_p = available_results['true_p']
        
        ax.errorbar(true_p, available_results['recovered_p_mean'],
                    yerr=available_results['recovered_p_std'],
                    fmt='o-', capsize=4, label=f'{circuit_name.capitalize()} Circuit',
                    markersize=8)
        
        p_range = [0, max(true_p) * 1.1]
        ax.plot(p_range, p_range, 'k--', alpha=0.5, label='Perfect Recovery', linewidth=2)
        
        ax.set_xlabel('True Noise Parameter (p)', fontsize=12)
        ax.set_ylabel('Recovered Noise Parameter (p)', fontsize=12)
        ax.set_title(f'Parameter Recovery: {n_qubits}-Qubit {circuit_name.capitalize()} Circuit', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(p_range)
        ax.set_ylim(p_range)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        log(f"Plot saved to {plot_path}")
    
    # Create summary statistics
    summary_file = os.path.join(results_dir, 'parameter_recovery_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Parameter Recovery Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Qubits: {n_qubits}\n")
        f.write(f"Noise parameters tested: {p_values}\n")
        f.write(f"Shots per trial: {n_shots}\n")
        f.write(f"Repetitions per p: {n_repetitions}\n\n")
        
        for circuit_type in ['baseline', 'robust']:
            if all_results[circuit_type] is not None:
                r = all_results[circuit_type]
                f.write(f"\n{circuit_type.upper()} Circuit Results:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'True p':<10} {'Recovered p':<20} {'Error':<15}\n")
                for i, p in enumerate(r['true_p']):
                    rec_p = r['recovered_p_mean'][i]
                    rec_std = r['recovered_p_std'][i]
                    error = r['recovery_error_mean'][i]
                    f.write(f"{p:<10.4f} {rec_p:.4f}±{rec_std:.4f}     {error:+.4f}\n")
                
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
