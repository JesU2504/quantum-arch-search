#!/usr/bin/env python3
"""
Lambda Sweep Experiment for Part 1 (Brittleness) of ExpPlan.md.

This script implements Experiment 1.1 from ExpPlan.md:
- Task: 4-qubit GHZ state preparation
- Baseline: Train ArchitectEnv with static penalty λ ∈ [0.001, 0.005, 0.01, 0.05, 0.1]
- Metrics:
    - Success rate: % of seeds (out of 5) reaching fidelity > 0.99
    - Convergence variance: std. dev. of final CNOT counts
    - Mean/median CNOT count for each lambda

Hard-coded configuration (per ExpPlan.md):
    - Lambda values: [0.001, 0.005, 0.01, 0.05, 0.1]
    - Seeds: 5 random seeds for each lambda
    - Target: 4-qubit GHZ state

Results are saved to results/lambda_sweep_<timestamp>/.
"""

import os
import json
import random
import cirq
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from experiments import config
from qas_gym.envs import ArchitectEnv
from qas_gym.utils import get_ghz_state

# ============================================================================
# Hard-coded Configuration (per ExpPlan.md - Part 1, Experiment 1.1)
# ============================================================================
LAMBDA_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
N_SEEDS = 5
N_QUBITS = 4
FIDELITY_THRESHOLD = 0.99  # Success threshold per ExpPlan.md
MAX_CIRCUIT_TIMESTEPS = config.MAX_CIRCUIT_TIMESTEPS
# Default training steps for PPO when not specified by the pipeline
# This value is used for standalone CLI invocations
DEFAULT_TRAINING_STEPS = 50000  # Enough for convergence but manageable runtime
N_STEPS_PER_UPDATE = 1024


class LambdaSweepCallback(BaseCallback):
    """
    Callback to track training progress and final circuit metrics.
    
    Records final fidelity and CNOT count at the end of each episode.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.final_fidelity = 0.0
        self.final_cnot_count = 0
        self.best_fidelity = 0.0
        self.best_cnot_count = 0

    def _on_step(self) -> bool:
        # Check for episode end and record metrics
        if 'infos' in self.locals and self.locals['infos'] and self.locals['dones'][0]:
            info = self.locals['infos'][0]
            fidelity = info.get('fidelity', 0.0)
            
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


def run_single_trial(lambda_penalty: float, seed: int, target_state: np.ndarray, training_steps: int) -> dict:
    """
    Run a single training trial with given lambda penalty and seed.
    
    Args:
        lambda_penalty: The complexity penalty weight (λ) for R = F - λC.
        seed: Random seed for reproducibility.
        target_state: The target quantum state (4-qubit GHZ).
        training_steps: Total number of timesteps for PPO training.
                        Controlled by the main experiment pipeline.
        
    Returns:
        Dict with 'final_fidelity', 'best_fidelity', 'cnot_count', 'success'.
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
    # The complexity_penalty_weight is the λ in R = F - λC (per ExpPlan.md)
    # reward_penalty is kept at baseline value (0.01) for consistency
    env = ArchitectEnv(
        target=target_state,
        fidelity_threshold=1.1,  # > 1.0 to ensure episodes run to max_timesteps
        reward_penalty=0.01,  # Fixed penalty for reward shaping (same as baseline)
        max_timesteps=MAX_CIRCUIT_TIMESTEPS,
        complexity_penalty_weight=lambda_penalty,  # The λ parameter we're sweeping
    )
    
    # Create agent with fixed hyperparameters
    agent_params = config.AGENT_PARAMS.copy()
    agent_params['n_steps'] = N_STEPS_PER_UPDATE
    agent_params['seed'] = seed
    
    model = PPO("MlpPolicy", env=env, **agent_params)
    
    # Train with callback to track metrics
    # Use the training_steps parameter passed from the pipeline (or default)
    callback = LambdaSweepCallback()
    model.learn(total_timesteps=training_steps, callback=callback)
    
    # Determine success: best fidelity > 0.99
    success = callback.best_fidelity > FIDELITY_THRESHOLD
    
    return {
        'final_fidelity': callback.final_fidelity,
        'best_fidelity': callback.best_fidelity,
        'cnot_count': callback.best_cnot_count,
        'success': success,
    }


def run_lambda_sweep(results_dir: str = None, logger=None, training_steps: int = None) -> dict:
    """
    Run the full lambda sweep experiment.
    
    Trains the architect agent for each lambda value with multiple seeds,
    then computes aggregate metrics.
    
    Args:
        results_dir: Directory to save results. If None, creates a timestamped dir.
        logger: Optional logger for output. If None, uses print.
        training_steps: Total timesteps for PPO training per trial. 
                        If None, defaults to DEFAULT_TRAINING_STEPS (50000).
                        When called from run_experiments.py, this is typically set
                        to baseline_steps (or architect_steps) to match the preset.
        
    Returns:
        Dict with per-lambda and aggregate metrics.
    """
    # Use default training steps if not specified (for standalone CLI usage)
    effective_training_steps = training_steps if training_steps is not None else DEFAULT_TRAINING_STEPS
    
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
    log(f"Lambda values: {LAMBDA_VALUES}")
    log(f"Seeds per lambda: {N_SEEDS}")
    log(f"Qubits: {N_QUBITS}")
    log(f"Training steps per trial: {effective_training_steps}")
    log(f"Results directory: {results_dir}")
    
    # Get target state (4-qubit GHZ)
    target_state = get_ghz_state(N_QUBITS)
    
    # Store all results
    all_results = {}
    
    for lambda_val in LAMBDA_VALUES:
        log(f"\n--- Lambda = {lambda_val} ---")
        lambda_results = []
        
        for seed in range(N_SEEDS):
            log(f"  Seed {seed + 1}/{N_SEEDS}...")
            # Pass effective_training_steps to each trial for PPO training
            trial_result = run_single_trial(lambda_val, seed, target_state, effective_training_steps)
            lambda_results.append(trial_result)
            log(f"    Best fidelity: {trial_result['best_fidelity']:.4f}, "
                f"CNOT count: {trial_result['cnot_count']}, "
                f"Success: {trial_result['success']}")
        
        # Compute aggregate metrics for this lambda
        successes = [r['success'] for r in lambda_results]
        cnot_counts = [r['cnot_count'] for r in lambda_results]
        best_fidelities = [r['best_fidelity'] for r in lambda_results]
        
        success_rate = sum(successes) / len(successes)
        cnot_mean = np.mean(cnot_counts)
        cnot_median = np.median(cnot_counts)
        cnot_std = np.std(cnot_counts)  # Convergence variance
        
        all_results[str(lambda_val)] = {
            'trials': lambda_results,
            'success_rate': success_rate,
            'cnot_mean': cnot_mean,
            'cnot_median': cnot_median,
            'cnot_std': cnot_std,
            'fidelity_mean': np.mean(best_fidelities),
            'fidelity_std': np.std(best_fidelities),
        }
        
        log(f"  Summary for λ={lambda_val}:")
        log(f"    Success rate: {success_rate:.2%}")
        log(f"    CNOT count - mean: {cnot_mean:.2f}, median: {cnot_median:.2f}, std: {cnot_std:.2f}")
        log(f"    Fidelity - mean: {np.mean(best_fidelities):.4f}, std: {np.std(best_fidelities):.4f}")
    
    # Save results to JSON
    results_file = os.path.join(results_dir, 'lambda_sweep_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {results_file}")
    
    # Create summary file
    summary_file = os.path.join(results_dir, 'lambda_sweep_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Lambda Sweep Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Lambda values: {LAMBDA_VALUES}\n")
        f.write(f"Seeds per lambda: {N_SEEDS}\n")
        f.write(f"Qubits: {N_QUBITS}\n")
        f.write(f"Training steps per trial: {effective_training_steps}\n")
        f.write(f"Fidelity threshold: {FIDELITY_THRESHOLD}\n\n")
        f.write("Per-Lambda Results:\n")
        f.write("-" * 50 + "\n")
        for lambda_val in LAMBDA_VALUES:
            r = all_results[str(lambda_val)]
            f.write(f"\nλ = {lambda_val}:\n")
            f.write(f"  Success rate: {r['success_rate']:.2%}\n")
            f.write(f"  CNOT count - mean: {r['cnot_mean']:.2f}, median: {r['cnot_median']:.2f}, std: {r['cnot_std']:.2f}\n")
            f.write(f"  Fidelity - mean: {r['fidelity_mean']:.4f}, std: {r['fidelity_std']:.4f}\n")
    log(f"Summary saved to {summary_file}")
    
    return all_results


if __name__ == "__main__":
    # Run as standalone script
    results = run_lambda_sweep()
    print("\n=== Final Results ===")
    for lambda_val in LAMBDA_VALUES:
        r = results[str(lambda_val)]
        print(f"λ={lambda_val}: Success={r['success_rate']:.2%}, "
              f"CNOT mean={r['cnot_mean']:.2f}±{r['cnot_std']:.2f}")
