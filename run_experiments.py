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
from datetime import datetime
import shutil


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

	from experiments.train_architect_ghz import train_baseline_architect
	from experiments.train_saboteur_only import train_saboteur_only
	from experiments.train_adversarial import train_adversarial
	from experiments.compare_circuits import compare_noise_resilience

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

	metadata = {
		'timestamp': timestamp,
		'preset': args.preset,
		'n_qubits': args.n_qubits,
		'baseline_steps': baseline_steps,
		'saboteur_steps': saboteur_steps,
		'adversarial_gens': adversarial_gens,
		'seed': args.seed,
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

	logger.info('Pipeline finished. Results in %s', base)


def parse_args():
	p = argparse.ArgumentParser(description='Run the experiment pipeline: baseline, saboteur-only, adversarial, compare')
	p.add_argument('--preset', choices=['quick', 'full', 'long'], default='quick')
	p.add_argument('--n-qubits', type=int, default=3)
	p.add_argument('--base-dir', type=str, default=None, help='Base results directory (default: results/run_<timestamp>)')
	p.add_argument('--skip-baseline', action='store_true')
	p.add_argument('--skip-saboteur', action='store_true')
	p.add_argument('--skip-adversarial', action='store_true')
	p.add_argument('--skip-compare', action='store_true')
	p.add_argument('--baseline-steps', type=int, default=None)
	p.add_argument('--saboteur-steps', type=int, default=None)
	p.add_argument('--max-circuit-gates', type=int, default=8)
	p.add_argument('--fidelity-threshold', type=float, default=0.9)
	p.add_argument('--seed', type=int, default=None)
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	run_pipeline(args)

