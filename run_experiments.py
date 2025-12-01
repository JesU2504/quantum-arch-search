#!/usr/bin/env python3
"""Orchestrator for the experiment pipeline.

Usage: run this script to run baseline, saboteur-only, adversarial co-evolution,
and comparison steps in sequence. Designed to be safe for quick/production runs.

Examples:
  python3 run_experiments.py --n-qubits 3  # Uses "Quick" settings from config.py
  python3 run_experiments.py --n-qubits 4  # Uses "Full" settings from config.py
"""

import argparse
import json
import os
import sys
from datetime import datetime
import shutil

# Add repository root and src to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)


def setup_logger(log_path):
    import logging
    logger = logging.getLogger('run_experiments')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def save_metadata(path, metadata):
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def run_pipeline(args):
    # Lazy imports of experiment entrypoints
    # Global reproducibility seeding
    if args.seed is not None:
        import random, numpy as _np
        random.seed(args.seed)
        _np.random.seed(args.seed)
        try:
            import torch as _torch
            _torch.manual_seed(args.seed)
        except Exception:
            pass

    from experiments.train_architect import train_baseline_architect
    from experiments.train_saboteur_only import train_saboteur_only
    from experiments.train_adversarial import train_adversarial
    from experiments.compare_circuits import compare_noise_resilience
    from experiments.lambda_sweep import run_lambda_sweep
    from experiments.parameter_recovery import run_parameter_recovery
    from experiments.cross_noise_robustness import run_cross_noise_robustness
    
    # UNIFICATION: Import config to act as Single Source of Truth
    from experiments import config as exp_config

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    base = args.base_dir or f"results/run_{timestamp}"
    os.makedirs(base, exist_ok=True)

    # Logger
    log_file = os.path.join(base, 'experiment.log')
    logger = setup_logger(log_file)

    # --- LOAD PARAMETERS FROM CONFIG.PY ---
    # We ignore the 'preset' argument's hardcoded values and look up
    # the correct params based on n_qubits directly.
    
    n_qubits = args.n_qubits
    if n_qubits not in exp_config.EXPERIMENT_PARAMS:
        logger.warning(f"No specific config found for {n_qubits} qubits. Defaulting to 4-qubit settings.")
        params = exp_config.EXPERIMENT_PARAMS[4]
    else:
        params = exp_config.EXPERIMENT_PARAMS[n_qubits]
    
    logger.info(f"Loading experimental parameters for {n_qubits} qubits from config.py")
    
    # Map config keys to local variables
    baseline_steps = params["ARCHITECT_STEPS"]
    baseline_n_steps = params["ARCHITECT_N_STEPS"]
    
    saboteur_steps = params["SABOTEUR_STEPS"]
    saboteur_n_steps = params["SABOTEUR_N_STEPS"]
    
    adversarial_gens = params["N_GENERATIONS"]
    adversarial_arch_steps = params["ARCHITECT_STEPS_PER_GENERATION"]
    adversarial_sab_steps = params["SABOTEUR_STEPS_PER_GENERATION"]

    # Allow CLI overrides (useful for testing small changes without editing config)
    if args.baseline_steps is not None:
        baseline_steps = args.baseline_steps
    if args.saboteur_steps is not None:
        saboteur_steps = args.saboteur_steps

    # Get effective n_seeds
    effective_n_seeds = args.n_seeds if args.n_seeds is not None else exp_config.N_SEEDS
    
    metadata = {
        'timestamp': timestamp,
        'n_qubits': args.n_qubits,
        'baseline_steps': baseline_steps,
        'saboteur_steps': saboteur_steps,
        'adversarial_gens': adversarial_gens,
        'adversarial_arch_steps': adversarial_arch_steps,
        'seed': args.seed,
        'n_seeds': effective_n_seeds,
        'source': 'experiments.config',
    }
    save_metadata(os.path.join(base, 'metadata.json'), metadata)
    logger.info('Starting pipeline with metadata: %s', metadata)

    # 1) Baseline
    baseline_dir = os.path.join(base, 'baseline')
    os.makedirs(baseline_dir, exist_ok=True)
    if not args.skip_baseline:
        logger.info('Running baseline architect (noiseless)')
        train_baseline_architect(
            results_dir=baseline_dir,
            n_qubits=args.n_qubits,
            architect_steps=baseline_steps,
            n_steps=baseline_n_steps,
            include_rotations=exp_config.INCLUDE_ROTATIONS,
            task_mode=args.task_mode
        )
    else:
        logger.info('Skipping baseline as requested')

    # 1.5) Lambda Sweep
    lambda_sweep_dir = os.path.join(base, 'lambda_sweep')
    os.makedirs(lambda_sweep_dir, exist_ok=True)
    if not args.skip_lambda_sweep:
        logger.info('Running lambda sweep experiment (ExpPlan Part 1, Exp 1.1) with n_seeds=%d', effective_n_seeds)
        run_lambda_sweep(results_dir=lambda_sweep_dir, logger=logger, training_steps=baseline_steps, 
                         n_seeds=effective_n_seeds, n_qubits=args.n_qubits)
        logger.info('Lambda sweep complete. Results saved to %s', lambda_sweep_dir)
    else:
        logger.info('Skipping lambda sweep as requested')

    # 2) Saboteur-only
    saboteur_dir = os.path.join(base, 'saboteur')
    os.makedirs(saboteur_dir, exist_ok=True)
    if not args.skip_saboteur:
        logger.info('Running saboteur-only (attacks baseline circuit)')
        vanilla_src = os.path.join(baseline_dir, 'circuit_vanilla.json')
        if not os.path.exists(vanilla_src):
            logger.error(f'Vanilla circuit not found at {vanilla_src}. Cannot run saboteur-only.')
        else:
            try:
                from qas_gym.utils import load_circuit
                static_circuit = load_circuit(vanilla_src)
                if not static_circuit.all_operations():
                    logger.error(f'Loaded circuit from {vanilla_src} is empty. Cannot run saboteur-only.')
                else:
                    train_saboteur_only(results_dir=saboteur_dir, n_qubits=args.n_qubits, 
                                      saboteur_steps=saboteur_steps, n_steps=saboteur_n_steps, 
                                      max_error_level=1, baseline_circuit_path=vanilla_src)
            except Exception as e:
                logger.error(f'Error loading circuit from {vanilla_src}: {e}. Cannot run saboteur-only.')
    else:
        logger.info('Skipping saboteur-only as requested')

    # 3) Adversarial co-evolution
    adversarial_dir = os.path.join(base, 'adversarial')
    os.makedirs(adversarial_dir, exist_ok=True)
    adv_training_dir = None
    if not args.skip_adversarial:
        logger.info(f'Running adversarial co-evolution ({adversarial_gens} generations)')
        train_adversarial(
            results_dir=adversarial_dir,
            n_qubits=args.n_qubits,
            n_generations=adversarial_gens,
            architect_steps_per_generation=adversarial_arch_steps,
            saboteur_steps_per_generation=adversarial_sab_steps,
            max_circuit_gates=args.max_circuit_gates,
            fidelity_threshold=args.fidelity_threshold,
            include_rotations=exp_config.INCLUDE_ROTATIONS,
            task_mode=args.task_mode
        )
        
        # Plotting trigger
        from glob import glob
        import re
        import subprocess
        adv_subdirs = glob(os.path.join(adversarial_dir, 'adversarial_training_*'))
        if adv_subdirs:
            # Sort by timestamp
            adv_subdirs.sort(key=lambda x: re.findall(r'adversarial_training_(\d+)-(\d+)', x)[0] if re.findall(r'adversarial_training_(\d+)-(\d+)', x) else x)
            adv_training_dir = adv_subdirs[-1]
            logger.info(f'Plotting coevolutionary process from {adv_training_dir}')
            try:
                result = subprocess.run([
                    'python', 'experiments/plot_coevolution.py',
                    '--run-dir', adv_training_dir
                ], capture_output=True, text=True)
                if result.stdout: logger.info(result.stdout)
                if result.stderr: logger.warning(result.stderr)
            except Exception as e:
                logger.error(f'Error running plot_coevolution.py: {e}')
    else:
        logger.info('Skipping adversarial as requested')

    # 4) Comparison & File Management
    compare_base = os.path.join(base, 'compare')
    run0 = os.path.join(compare_base, 'run_0')
    os.makedirs(run0, exist_ok=True)

    vanilla_src = os.path.join(baseline_dir, 'circuit_vanilla.json')
    if os.path.exists(vanilla_src):
        shutil.copy(vanilla_src, os.path.join(run0, 'circuit_vanilla.json'))
    
    # Locate robust circuit
    robust_src = os.path.join(adversarial_dir, 'circuit_robust.json')
    if not os.path.exists(robust_src) and adv_training_dir:
         # Fallback to the specific training dir if the main link isn't there
         robust_src = os.path.join(adv_training_dir, 'circuit_robust.json')

    if os.path.exists(robust_src):
         shutil.copy(robust_src, os.path.join(run0, 'circuit_robust.json'))
         logger.info(f'Robust circuit copied to comparison folder')
    else:
         logger.warning('Robust circuit not found for comparison.')

    # 5) Compare
    if not args.skip_compare:
        logger.info('Running comparison across runs')
        compare_noise_resilience(base_results_dir=compare_base, num_runs=1, n_qubits=args.n_qubits)

    # 6) Parameter Recovery
    param_recovery_dir = os.path.join(base, 'parameter_recovery')
    os.makedirs(param_recovery_dir, exist_ok=True)
    if not args.skip_parameter_recovery:
        if os.path.exists(vanilla_src):
            logger.info('Running parameter recovery experiment')
            robust_to_use = robust_src if (robust_src and os.path.exists(robust_src)) else vanilla_src
            run_parameter_recovery(
                results_dir=param_recovery_dir,
                n_qubits=args.n_qubits,
                baseline_circuit_path=vanilla_src,
                robust_circuit_path=robust_to_use,
                n_repetitions=effective_n_seeds,
                base_seed=args.seed if args.seed is not None else 42,
                logger=logger
            )

    # 7) Cross-Noise
    cross_noise_dir = os.path.join(base, 'cross_noise')
    os.makedirs(cross_noise_dir, exist_ok=True)
    if not args.skip_cross_noise:
        vanilla_circuit_path = os.path.join(run0, 'circuit_vanilla.json')
        robust_circuit_path = os.path.join(run0, 'circuit_robust.json')
        if os.path.exists(vanilla_circuit_path) and os.path.exists(robust_circuit_path):
             logger.info('Running cross-noise robustness experiment')
             run_cross_noise_robustness(
                baseline_circuit_path=vanilla_circuit_path,
                robust_circuit_path=robust_circuit_path,
                output_dir=cross_noise_dir,
                n_qubits=args.n_qubits,
                logger=logger
             )

    logger.info('Pipeline finished. Results in %s', base)


def parse_args():
    p = argparse.ArgumentParser(description='Run the experiment pipeline')
    # Preset arg removed or ignored in favor of config.py + n_qubits
    p.add_argument('--n-qubits', type=int, default=4, help="Number of qubits (3=Quick, 4=Full, 5=Long)")
    p.add_argument('--base-dir', type=str, default=None)
    p.add_argument('--skip-baseline', action='store_true')
    p.add_argument('--skip-lambda-sweep', action='store_true')
    p.add_argument('--skip-saboteur', action='store_true')
    p.add_argument('--skip-adversarial', action='store_true')
    p.add_argument('--skip-compare', action='store_true')
    p.add_argument('--skip-parameter-recovery', action='store_true')
    p.add_argument('--skip-cross-noise', action='store_true')
    p.add_argument('--baseline-steps', type=int, default=None)
    p.add_argument('--saboteur-steps', type=int, default=None)
    p.add_argument('--max-circuit-gates', type=int, default=20)
    p.add_argument('--fidelity-threshold', type=float, default=1.1) # Train by improvement
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--n-seeds', type=int, default=None)
    p.add_argument('--task-mode', type=str, default=None, choices=['state_preparation', 'unitary_preparation'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)