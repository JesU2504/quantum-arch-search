#!/usr/bin/env python3
"""
Lambda Sweep Experiment for Part 1 (Brittleness) of ExpPlan.md.

This script implements Experiment 1.1 from ExpPlan.md:
- Task: n-qubit Toffoli gate compilation (configurable, default n=4)
- Baseline: Train ArchitectEnv with static penalty λ ∈ [0.001, 0.005, 0.01, 0.05, 0.1]
- Metrics:
    - Success rate: % of seeds reaching fidelity > 0.99
    - Convergence variance: std. dev. of final CNOT counts
    - Mean/median CNOT count for each lambda

Note: This experiment uses n-controlled Toffoli gates as the default compilation
target (CCNOT for 3 qubits, CCCNOT for 4 qubits, etc.). GHZ state preparation
is available as a legacy option via get_ghz_state().

Statistical Protocol:
    - Number of seeds: Configurable via n_seeds parameter (default: config.N_SEEDS)
    - Recommended: At least 5 seeds, ideally 10 for publication-quality results
    - Each seed's results are saved separately in per-seed JSON files
    - Aggregated metrics include mean ± std with error bars on plots
    - Summary file includes all seeds used, hyperparameters, and statistical info

Results are saved to results/lambda_sweep_<timestamp>/.
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
import random
import argparse
import cirq
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from experiments import config
from qas_gym.envs import ArchitectEnv
from qas_gym.utils import get_ghz_state, get_toffoli_state

# Import statistical utilities
from utils.stats import (
    aggregate_metrics,
    create_experiment_summary,
    save_experiment_summary,
    write_summary_txt,
    get_git_commit_hash,
    format_metric_with_error,
    compute_success_rate,
)

# ============================================================================
# Configuration (per ExpPlan.md - Part 1, Experiment 1.1)
# ============================================================================
LAMBDA_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
FIDELITY_THRESHOLD = 0.99  # Success threshold per ExpPlan.md
MAX_CIRCUIT_TIMESTEPS = config.MAX_CIRCUIT_TIMESTEPS
# Default training steps for PPO when not specified by the pipeline
DEFAULT_TRAINING_STEPS = 50000  # Enough for convergence but manageable runtime
N_STEPS_PER_UPDATE = 1024


class LambdaSweepCallback(BaseCallback):
    """
    Callback to track training progress and final circuit metrics.
    
    Records final fidelity and CNOT count at the end of each episode.
    Stores full training history for analysis.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.final_fidelity = 0.0
        self.final_cnot_count = 0
        self.best_fidelity = 0.0
        self.best_cnot_count = 0
        # Training history for detailed analysis
        self.fidelity_history = []
        self.step_history = []

    def _on_step(self) -> bool:
        # Check for episode end and record metrics
        if 'infos' in self.locals and self.locals['infos'] and self.locals['dones'][0]:
            info = self.locals['infos'][0]
            fidelity = info.get('fidelity', 0.0)
            
            # Record training history
            self.fidelity_history.append(fidelity)
            self.step_history.append(self.num_timesteps)
            
            # Track best fidelity achieved
            if fidelity > self.best_fidelity:
                self.best_fidelity = fidelity
                # Count CNOTs in the circuit
                if 'circuit' in info:
                    circuit = info['circuit']
                    self.best_cnot_count = _count_cnots(circuit)
            
            # Update final metrics (last episode's metrics)
            self.final_fidelity = fidelity
            if 'circuit' in info:
                self.final_cnot_count = _count_cnots(info['circuit'])
        return True


def _count_cnots(circuit) -> int:
    """Count the number of CNOT gates in a circuit."""
    count = 0
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.CNotPowGate):
            count += 1
    return count


def run_single_trial(
    lambda_penalty: float,
    seed: int,
    target_state: np.ndarray,
    training_steps: int,
    n_qubits: int,
    save_dir: str = None,
) -> dict:
    """
    Run a single training trial with given lambda penalty and seed.
    
    Args:
        lambda_penalty: The complexity penalty weight (λ) for R = F - λC.
        seed: Random seed for reproducibility.
        target_state: The target quantum state.
        training_steps: Total number of timesteps for PPO training.
        n_qubits: Number of qubits.
        save_dir: Optional directory to save per-seed results.
        
    Returns:
        Dict with trial results including training history.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    
    # Create environment with the specified lambda penalty
    env = ArchitectEnv(
        target=target_state,
        fidelity_threshold=1.1,  # > 1.0 to ensure episodes run to max_timesteps
        reward_penalty=0.01,  # Fixed penalty for reward shaping
        max_timesteps=MAX_CIRCUIT_TIMESTEPS,
        complexity_penalty_weight=lambda_penalty,
    )
    
    # Create agent with fixed hyperparameters
    agent_params = config.AGENT_PARAMS.copy()
    agent_params['n_steps'] = N_STEPS_PER_UPDATE
    agent_params['seed'] = seed
    
    model = PPO("MlpPolicy", env=env, **agent_params)
    
    # Train with callback to track metrics
    callback = LambdaSweepCallback()
    model.learn(total_timesteps=training_steps, callback=callback)
    
    # Determine success: best fidelity > threshold
    success = callback.best_fidelity > FIDELITY_THRESHOLD
    
    result = {
        'seed': seed,
        'lambda': lambda_penalty,
        'final_fidelity': float(callback.final_fidelity),
        'best_fidelity': float(callback.best_fidelity),
        'cnot_count': int(callback.best_cnot_count),
        'success': bool(success),
        'training_steps': training_steps,
        'fidelity_history': callback.fidelity_history,
        'step_history': callback.step_history,
    }
    
    # Save per-seed results if directory provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        seed_file = os.path.join(save_dir, f'seed_{seed}_results.json')
        with open(seed_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_result = {k: (v if not isinstance(v, np.ndarray) else v.tolist()) 
                          for k, v in result.items()}
            json.dump(json_result, f, indent=2)
    
    return result


def create_lambda_sweep_plot(
    all_results: dict,
    lambda_values: list,
    output_path: str,
    n_seeds: int,
):
    """
    Create error bar plot for lambda sweep results.
    
    Shows mean ± std for fidelity and CNOT count, with sample size annotation.
    Also overlays individual seed results as faint points.
    
    Args:
        all_results: Dict with per-lambda aggregated results.
        lambda_values: List of lambda values.
        output_path: Path to save the plot.
        n_seeds: Number of seeds (for annotation).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    lambdas = np.array(lambda_values)
    
    # --- Left plot: Fidelity ---
    ax1 = axes[0]
    fidelity_means = []
    fidelity_stds = []
    all_fidelities = []
    
    for lv in lambda_values:
        r = all_results[str(lv)]
        fidelity_means.append(r['fidelity_mean'])
        fidelity_stds.append(r['fidelity_std'])
        all_fidelities.append([t['best_fidelity'] for t in r['trials']])
    
    # Plot individual points (faint)
    for i, (lv, fids) in enumerate(zip(lambda_values, all_fidelities)):
        ax1.scatter([lv] * len(fids), fids, alpha=0.3, s=30, color='tab:blue')
    
    # Plot mean with error bars
    ax1.errorbar(lambdas, fidelity_means, yerr=fidelity_stds, fmt='o-',
                 capsize=5, capthick=2, linewidth=2, markersize=10,
                 color='tab:blue', label='Mean ± Std')
    
    ax1.axhline(y=FIDELITY_THRESHOLD, color='red', linestyle='--', 
                alpha=0.7, label=f'Success threshold ({FIDELITY_THRESHOLD})')
    ax1.set_xscale('log')
    ax1.set_xlabel('Lambda (λ)', fontsize=12)
    ax1.set_ylabel('Best Fidelity', fontsize=12)
    ax1.set_title(f'Fidelity vs Lambda (n={n_seeds} seeds)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # --- Right plot: CNOT Count ---
    ax2 = axes[1]
    cnot_means = []
    cnot_stds = []
    all_cnots = []
    
    for lv in lambda_values:
        r = all_results[str(lv)]
        cnot_means.append(r['cnot_mean'])
        cnot_stds.append(r['cnot_std'])
        all_cnots.append([t['cnot_count'] for t in r['trials']])
    
    # Plot individual points (faint)
    for i, (lv, cnots) in enumerate(zip(lambda_values, all_cnots)):
        ax2.scatter([lv] * len(cnots), cnots, alpha=0.3, s=30, color='tab:orange')
    
    # Plot mean with error bars
    ax2.errorbar(lambdas, cnot_means, yerr=cnot_stds, fmt='s-',
                 capsize=5, capthick=2, linewidth=2, markersize=10,
                 color='tab:orange', label='Mean ± Std')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Lambda (λ)', fontsize=12)
    ax2.set_ylabel('CNOT Count', fontsize=12)
    ax2.set_title(f'CNOT Count vs Lambda (n={n_seeds} seeds)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_lambda_sweep(
    results_dir: str = None,
    logger=None,
    training_steps: int = None,
    n_seeds: int = None,
    n_qubits: int = 4,
    lambda_values: list = None,
) -> dict:
    """
    Run the full lambda sweep experiment with statistical reporting.
    
    Trains the architect agent for each lambda value with multiple seeds,
    then computes aggregate metrics with error bars.
    
    Args:
        results_dir: Directory to save results. If None, creates a timestamped dir.
        logger: Optional logger for output. If None, uses print.
        training_steps: Total timesteps for PPO training per trial. 
        n_seeds: Number of seeds per lambda. Defaults to config.N_SEEDS.
        n_qubits: Number of qubits for the target state.
        lambda_values: List of lambda values to sweep. Defaults to LAMBDA_VALUES.
        
    Returns:
        Dict with per-lambda and aggregate metrics.
    """
    # Use defaults if not specified
    effective_training_steps = training_steps if training_steps is not None else DEFAULT_TRAINING_STEPS
    effective_n_seeds = n_seeds if n_seeds is not None else config.N_SEEDS
    effective_lambda_values = lambda_values if lambda_values is not None else LAMBDA_VALUES
    
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # Set up results directory
    if results_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        results_dir = f"results/lambda_sweep_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    log(f"=== Lambda Sweep Experiment (ExpPlan Part 1, Exp 1.1) ===")
    log(f"Lambda values: {effective_lambda_values}")
    log(f"Seeds per lambda: {effective_n_seeds}")
    log(f"Qubits: {n_qubits}")
    log(f"Training steps per trial: {effective_training_steps}")
    log(f"Results directory: {results_dir}")
    
    # Get target state - use n-controlled Toffoli as default target
    # For n >= 2 qubits: CNOT (n=2), CCNOT/Toffoli (n=3), CCCNOT (n=4), etc.
    target_state = get_toffoli_state(n_qubits)
    
    # Track all seeds used for summary
    all_seeds_used = list(range(effective_n_seeds))
    
    # Store all results
    all_results = {}
    
    for lambda_val in effective_lambda_values:
        log(f"\n--- Lambda = {lambda_val} ---")
        lambda_results = []
        
        # Create per-lambda directory for seed results
        lambda_dir = os.path.join(results_dir, f'lambda_{lambda_val}')
        os.makedirs(lambda_dir, exist_ok=True)
        
        for seed in range(effective_n_seeds):
            log(f"  Seed {seed + 1}/{effective_n_seeds}...")
            trial_result = run_single_trial(
                lambda_penalty=lambda_val,
                seed=seed,
                target_state=target_state,
                training_steps=effective_training_steps,
                n_qubits=n_qubits,
                save_dir=lambda_dir,
            )
            lambda_results.append(trial_result)
            log(f"    Best fidelity: {trial_result['best_fidelity']:.4f}, "
                f"CNOT count: {trial_result['cnot_count']}, "
                f"Success: {trial_result['success']}")
        
        # Compute aggregate metrics using statistical utilities
        best_fidelities = [r['best_fidelity'] for r in lambda_results]
        cnot_counts = [r['cnot_count'] for r in lambda_results]
        successes = [r['success'] for r in lambda_results]
        
        fidelity_stats = aggregate_metrics(best_fidelities)
        cnot_stats = aggregate_metrics(cnot_counts)
        success_stats = compute_success_rate(successes)
        
        all_results[str(lambda_val)] = {
            'trials': lambda_results,
            'seeds_used': list(range(effective_n_seeds)),
            'n_seeds': effective_n_seeds,
            # Legacy format for backwards compatibility
            'success_rate': success_stats['rate'],
            'cnot_mean': cnot_stats['mean'],
            'cnot_median': cnot_stats['median'],
            'cnot_std': cnot_stats['std'],
            'fidelity_mean': fidelity_stats['mean'],
            'fidelity_std': fidelity_stats['std'],
            # Full statistical info
            'fidelity_stats': fidelity_stats,
            'cnot_stats': cnot_stats,
            'success_stats': success_stats,
        }
        
        log(f"  Summary for λ={lambda_val}:")
        log(f"    Success rate: {success_stats['rate']:.2%} (n={success_stats['n_total']})")
        log(f"    Fidelity: {format_metric_with_error(fidelity_stats['mean'], fidelity_stats['std'], fidelity_stats['n'])}")
        log(f"    CNOT count: {format_metric_with_error(cnot_stats['mean'], cnot_stats['std'], cnot_stats['n'])}")
    
    # Save results to JSON
    results_file = os.path.join(results_dir, 'lambda_sweep_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        json.dump(convert_numpy(all_results), f, indent=2)
    log(f"\nResults saved to {results_file}")
    
    # Create error bar plot
    plot_path = os.path.join(results_dir, 'lambda_sweep_plot.png')
    create_lambda_sweep_plot(all_results, effective_lambda_values, plot_path, effective_n_seeds)
    log(f"Plot with error bars saved to {plot_path}")
    
    # Create experiment summary using statistical utilities
    hyperparameters = {
        'lambda_values': effective_lambda_values,
        'n_qubits': n_qubits,
        'training_steps': effective_training_steps,
        'fidelity_threshold': FIDELITY_THRESHOLD,
        'max_circuit_timesteps': MAX_CIRCUIT_TIMESTEPS,
        'n_steps_per_update': N_STEPS_PER_UPDATE,
    }
    
    aggregated_results = {
        f'lambda_{lv}': all_results[str(lv)]['fidelity_stats']
        for lv in effective_lambda_values
    }
    
    summary = create_experiment_summary(
        experiment_name='lambda_sweep',
        n_seeds=effective_n_seeds,
        seeds_used=all_seeds_used,
        hyperparameters=hyperparameters,
        aggregated_results=aggregated_results,
        commit_hash=get_git_commit_hash(),
        additional_notes='Lambda sweep experiment for hyperparameter sensitivity analysis.'
    )
    
    save_experiment_summary(summary, results_dir, 'experiment_summary.json')
    write_summary_txt(
        output_dir=results_dir,
        experiment_name='Lambda Sweep (ExpPlan Part 1, Exp 1.1)',
        n_seeds=effective_n_seeds,
        seeds_used=all_seeds_used,
        hyperparameters=hyperparameters,
        aggregated_results=aggregated_results,
    )
    log(f"Summary files saved to {results_dir}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lambda Sweep Experiment for hyperparameter sensitivity analysis"
    )
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--n-seeds', type=int, default=None,
                        help=f'Number of seeds per lambda (default: {config.N_SEEDS})')
    parser.add_argument('--n-qubits', type=int, default=4,
                        help='Number of qubits (default: 4)')
    parser.add_argument('--training-steps', type=int, default=None,
                        help=f'Training steps per trial (default: {DEFAULT_TRAINING_STEPS})')
    args = parser.parse_args()
    
    results = run_lambda_sweep(
        results_dir=args.results_dir,
        n_seeds=args.n_seeds,
        n_qubits=args.n_qubits,
        training_steps=args.training_steps,
    )
    print("\n=== Final Results ===")
    # Use keys from results dict to handle custom lambda_values
    for lambda_key in sorted(results.keys(), key=lambda x: float(x)):
        r = results[lambda_key]
        print(f"λ={lambda_key}: Success={r['success_rate']:.2%}, "
              f"Fidelity={r['fidelity_mean']:.4f}±{r['fidelity_std']:.4f}, "
              f"CNOT={r['cnot_mean']:.2f}±{r['cnot_std']:.2f}")
