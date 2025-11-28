#!/usr/bin/env python3
"""Orchestrator for the experiment pipeline.

Usage: run this script to run baseline, saboteur-only, adversarial co-evolution,
and comparison steps in sequence. Designed to be safe for quick/production runs.

Examples:
  python3 run_experiments.py --preset quick --n-qubits 3
  python3 run_experiments.py --preset full --n-qubits 4 --base-dir results/prod_run
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
	# Global reproducibility seeding (Python, NumPy, optional Torch)
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

	timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
	base = args.base_dir or f"results/run_{timestamp}"
	os.makedirs(base, exist_ok=True)

	# Logger
	log_file = os.path.join(base, 'experiment.log')
	logger = setup_logger(log_file)

	# Presets
	if args.preset == 'quick':
		baseline_steps = 1000
		baseline_n_steps = 1000
		saboteur_steps = 1000
		saboteur_n_steps = 1000
		adversarial_gens = 2
		adversarial_arch_steps = 3000
		adversarial_sab_steps = 3000
	elif args.preset == 'full':
		baseline_steps = 200000
		baseline_n_steps = 2048
		saboteur_steps = 200000
		saboteur_n_steps = 2048
		adversarial_gens = 10
		adversarial_arch_steps = 100000
		adversarial_sab_steps = 100000
	elif args.preset == 'long':
		# Long experimental preset for extended training (multi-day / high compute)
		baseline_steps = 1000000
		baseline_n_steps = 4096
		saboteur_steps = 1000000
		saboteur_n_steps = 4096
		adversarial_gens = 20
		adversarial_arch_steps = 200000
		adversarial_sab_steps = 200000
	else:  # fallback to full
		baseline_steps = 200000
		baseline_n_steps = 2048
		saboteur_steps = 200000
		saboteur_n_steps = 2048
		adversarial_gens = 10
		adversarial_arch_steps = 100000
		adversarial_sab_steps = 100000

	# Allow CLI overrides
	if args.baseline_steps is not None:
		baseline_steps = args.baseline_steps
	if args.saboteur_steps is not None:
		saboteur_steps = args.saboteur_steps

	# Get effective n_seeds (from CLI or config default)
	from experiments import config as exp_config
	effective_n_seeds = args.n_seeds if args.n_seeds is not None else exp_config.N_SEEDS
	
	metadata = {
		'timestamp': timestamp,
		'preset': args.preset,
		'n_qubits': args.n_qubits,
		'baseline_steps': baseline_steps,
		'saboteur_steps': saboteur_steps,
		'adversarial_gens': adversarial_gens,
		'seed': args.seed,
		'n_seeds': effective_n_seeds,
		'statistical_protocol': {
			'n_seeds_per_setting': effective_n_seeds,
			'aggregation_method': 'mean ± std',
			'error_bars': True,
		},
	}
	save_metadata(os.path.join(base, 'metadata.json'), metadata)
	logger.info('Starting pipeline with metadata: %s', metadata)

	# 1) Baseline
	baseline_dir = os.path.join(base, 'baseline')
	os.makedirs(baseline_dir, exist_ok=True)
	if not args.skip_baseline:
		logger.info('Running baseline architect (noiseless)')
		train_baseline_architect(results_dir=baseline_dir, n_qubits=args.n_qubits, architect_steps=baseline_steps, n_steps=baseline_n_steps)
	else:
		logger.info('Skipping baseline as requested')

	# 1.5) Lambda Sweep (ExpPlan Part 1 - Brittleness Experiment)
	# This step implements Experiment 1.1 from ExpPlan.md:
	# - Tests hyperparameter sensitivity by sweeping λ ∈ [0.001, 0.005, 0.01, 0.05, 0.1]
	# - Runs n_seeds per lambda value for statistical significance
	# - Logs success rate (fidelity > 0.99) and CNOT count variance
	# - Uses baseline_steps (from preset or CLI) as training_steps for PPO consistency
	lambda_sweep_dir = os.path.join(base, 'lambda_sweep')
	os.makedirs(lambda_sweep_dir, exist_ok=True)
	if not args.skip_lambda_sweep:
		logger.info('Running lambda sweep experiment (ExpPlan Part 1, Exp 1.1) with n_seeds=%d', effective_n_seeds)
		# Pass baseline_steps as training_steps so PPO training is controlled by the pipeline preset
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
		# Check for valid baseline circuit file
		if not os.path.exists(vanilla_src):
			logger.error(f'Vanilla circuit not found at {vanilla_src}. Cannot run saboteur-only. Please run baseline architect step first.')
			return
		# Check if file is non-empty and contains a valid circuit
		try:
			from qas_gym.utils import load_circuit
			static_circuit = load_circuit(vanilla_src)
			if not static_circuit.all_operations():
				logger.error(f'Loaded circuit from {vanilla_src} is empty. Cannot run saboteur-only.')
				return
		except Exception as e:
			logger.error(f'Error loading circuit from {vanilla_src}: {e}. Cannot run saboteur-only.')
			return
		# Pass the baseline circuit path directly to the saboteur experiment
		train_saboteur_only(results_dir=saboteur_dir, n_qubits=args.n_qubits, saboteur_steps=saboteur_steps, n_steps=saboteur_n_steps, max_error_level=1, baseline_circuit_path=vanilla_src)
	else:
		logger.info('Skipping saboteur-only as requested')

	# 3) Adversarial co-evolution
	adversarial_dir = os.path.join(base, 'adversarial')
	os.makedirs(adversarial_dir, exist_ok=True)
	adv_plot_path = None
	adv_training_dir = None
	if not args.skip_adversarial:
		logger.info('Running adversarial co-evolution')
		train_adversarial(results_dir=adversarial_dir, n_qubits=args.n_qubits, n_generations=adversarial_gens, architect_steps_per_generation=adversarial_arch_steps, saboteur_steps_per_generation=adversarial_sab_steps, max_circuit_gates=args.max_circuit_gates, fidelity_threshold=args.fidelity_threshold)
		# Find the latest adversarial training subdir for plotting
		from glob import glob
		import re
		adv_subdirs = glob(os.path.join(adversarial_dir, 'adversarial_training_*'))
		if adv_subdirs:
			# Sort by timestamp in folder name
			adv_subdirs.sort(key=lambda x: re.findall(r'adversarial_training_(\d+)-(\d+)', x)[0] if re.findall(r'adversarial_training_(\d+)-(\d+)', x) else x)
			adv_training_dir = adv_subdirs[-1]
			logger.info(f'Plotting coevolutionary process from {adv_training_dir}')
			# Call plot_coevolution.py
			import subprocess
			try:
				result = subprocess.run([
					'python', 'experiments/plot_coevolution.py',
					'--run-dir', adv_training_dir
				], capture_output=True, text=True)
				logger.info(result.stdout)
				if result.stderr:
					logger.warning(result.stderr)
			except Exception as e:
				logger.error(f'Error running plot_coevolution.py: {e}')
		else:
			logger.warning('No adversarial_training_* subdirectory found for plotting.')
	else:
		logger.info('Skipping adversarial as requested')

	# 4) Prepare compare folder
	compare_base = os.path.join(base, 'compare')
	run0 = os.path.join(compare_base, 'run_0')
	os.makedirs(run0, exist_ok=True)

	vanilla_src = os.path.join(baseline_dir, 'circuit_vanilla.json')
	# Find robust circuit in latest adversarial_training_* subdir
	robust_src = os.path.join(adversarial_dir, 'circuit_robust.json')
	robust_found = False
	if os.path.exists(vanilla_src):
		shutil.copy(vanilla_src, os.path.join(run0, 'circuit_vanilla.json'))
		logger.info('Copied vanilla circuit to compare/run_0')
	else:
		logger.warning('Vanilla circuit not found; compare will be incomplete')

	# Try direct path first
	if os.path.exists(robust_src):
		shutil.copy(robust_src, os.path.join(run0, 'circuit_robust.json'))
		logger.info('Copied robust circuit to compare/run_0')
		robust_found = True
	else:
		# Search for latest adversarial_training_* subdir
		from glob import glob
		import re
		adv_subdirs = glob(os.path.join(adversarial_dir, 'adversarial_training_*'))
		if adv_subdirs:
			adv_subdirs.sort(key=lambda x: re.findall(r'adversarial_training_(\d+)-(\d+)', x)[0] if re.findall(r'adversarial_training_(\d+)-(\d+)', x) else x)
			adv_training_dir = adv_subdirs[-1]
			robust_candidate = os.path.join(adv_training_dir, 'circuit_robust.json')
			if os.path.exists(robust_candidate):
				shutil.copy(robust_candidate, os.path.join(run0, 'circuit_robust.json'))
				logger.info(f'Copied robust circuit from {robust_candidate} to compare/run_0')
				robust_found = True
		if not robust_found:
			logger.warning('Robust circuit not found; compare will be incomplete')

	# 5) Compare
	if not args.skip_compare:
		logger.info('Running comparison across runs in %s', compare_base)
		compare_noise_resilience(base_results_dir=compare_base, num_runs=1, n_qubits=args.n_qubits)
	else:
		logger.info('Skipping compare as requested')

	# 6) Parameter Recovery Experiment
	# Tests how well we can recover the true noise parameter from measurement statistics
	# for both baseline and robust circuits
	param_recovery_dir = os.path.join(base, 'parameter_recovery')
	os.makedirs(param_recovery_dir, exist_ok=True)
	if not args.skip_parameter_recovery:
		logger.info('Running parameter recovery experiment with n_seeds=%d', effective_n_seeds)
		# Look for circuits in expected locations
		vanilla_src = os.path.join(baseline_dir, 'circuit_vanilla.json')
		robust_src = None
		# Try to find robust circuit
		if os.path.exists(os.path.join(adversarial_dir, 'circuit_robust.json')):
			robust_src = os.path.join(adversarial_dir, 'circuit_robust.json')
		else:
			# Search in adversarial_training_* subdirs
			from glob import glob
			import re
			adv_subdirs = glob(os.path.join(adversarial_dir, 'adversarial_training_*'))
			if adv_subdirs:
				def sort_key(x):
					matches = re.findall(r'adversarial_training_(\d+)-(\d+)', x)
					return matches[0] if matches else ('', '')
				adv_subdirs.sort(key=sort_key)
				robust_candidate = os.path.join(adv_subdirs[-1], 'circuit_robust.json')
				if os.path.exists(robust_candidate):
					robust_src = robust_candidate
		
		run_parameter_recovery(
			results_dir=param_recovery_dir,
			n_qubits=args.n_qubits,
			baseline_circuit_path=vanilla_src if os.path.exists(vanilla_src) else None,
			robust_circuit_path=robust_src,
			n_repetitions=effective_n_seeds,
			base_seed=args.seed if args.seed is not None else 42,
			logger=logger
		)
		logger.info('Parameter recovery experiment complete. Results saved to %s', param_recovery_dir)
	else:
		logger.info('Skipping parameter recovery as requested')
	# 6) Cross-Noise Robustness (ExpPlan Part 2, Exp 2.1)
	# This step evaluates both baseline and robust circuits under:
	# - Coherent over-rotation (unseen by Saboteur)
	# - Asymmetric Pauli noise (Saboteur typically uses symmetric)
	cross_noise_dir = os.path.join(base, 'cross_noise')
	os.makedirs(cross_noise_dir, exist_ok=True)
	if not args.skip_cross_noise:
		# Find circuit paths from compare folder (which already collected them)
		vanilla_circuit_path = os.path.join(run0, 'circuit_vanilla.json')
		robust_circuit_path = os.path.join(run0, 'circuit_robust.json')

		if os.path.exists(vanilla_circuit_path) and os.path.exists(robust_circuit_path):
			logger.info('Running cross-noise robustness experiment (ExpPlan Part 2, Exp 2.1)')
			run_cross_noise_robustness(
				baseline_circuit_path=vanilla_circuit_path,
				robust_circuit_path=robust_circuit_path,
				output_dir=cross_noise_dir,
				n_qubits=args.n_qubits,
				logger=logger
			)
			logger.info('Cross-noise robustness complete. Results saved to %s', cross_noise_dir)
		else:
			missing = []
			if not os.path.exists(vanilla_circuit_path):
				missing.append('vanilla')
			if not os.path.exists(robust_circuit_path):
				missing.append('robust')
			logger.warning(f'Skipping cross-noise robustness: missing {", ".join(missing)} circuit(s)')
	else:
		logger.info('Skipping cross-noise robustness as requested')

	logger.info('Pipeline finished. Results in %s', base)


def parse_args():
	p = argparse.ArgumentParser(description='Run the experiment pipeline: baseline, lambda-sweep, saboteur-only, adversarial, compare, parameter-recovery, cross-noise')
	p.add_argument('--preset', choices=['quick', 'full', 'long'], default='quick')
	p.add_argument('--n-qubits', type=int, default=3)
	p.add_argument('--base-dir', type=str, default=None, help='Base results directory (default: results/run_<timestamp>)')
	p.add_argument('--skip-baseline', action='store_true')
	p.add_argument('--skip-lambda-sweep', action='store_true', help='Skip lambda sweep experiment (ExpPlan Part 1)')
	p.add_argument('--skip-saboteur', action='store_true')
	p.add_argument('--skip-adversarial', action='store_true')
	p.add_argument('--skip-compare', action='store_true')
	p.add_argument('--skip-parameter-recovery', action='store_true', help='Skip parameter recovery experiment')
	p.add_argument('--skip-cross-noise', action='store_true', help='Skip cross-noise robustness experiment (ExpPlan Part 2)')
	p.add_argument('--baseline-steps', type=int, default=None)
	p.add_argument('--saboteur-steps', type=int, default=None)
	p.add_argument('--max-circuit-gates', type=int, default=8)
	p.add_argument('--fidelity-threshold', type=float, default=0.9)
	p.add_argument('--seed', type=int, default=None)
	# Statistical reporting arguments
	p.add_argument('--n-seeds', type=int, default=None, 
		help=f'Number of random seeds per experiment setting for statistical reporting. '
		     f'Recommended: at least 5, ideally 10. Default uses config.N_SEEDS.')
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	run_pipeline(args)

