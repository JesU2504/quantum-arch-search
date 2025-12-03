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
import glob
import itertools
import random
from pathlib import Path

# Add repository root and src to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from qas_gym.utils import fidelity_pure_target


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


def set_global_seed(seed, logger=None):
    """Best-effort global seeding for reproducibility."""
    if logger is not None:
        logger.info(f"Setting global seed = {seed}")
    import random
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        if logger is not None:
            logger.warning("Torch not available; skipping torch seeding.")


def summarize_fidelity_outputs(base_dir: str, analysis_dir: str, logger=None):
    """
    Collect fidelity stats from lambda_sweep (train_architect) and adversarial runs.
    Writes a raw CSV for downstream reporting.
    """
    import csv
    rows = []
    base_path = Path(base_dir)

    # Lambda sweep (train_architect)
    for path in base_path.glob("**/lambda_sweep/lambda_*/seed_*_results.json"):
        with open(path) as f:
            data = json.load(f)
        rel = path.relative_to(base_path)
        rows.append({
            "experiment": "train_architect_lambda_sweep",
            "run": str(rel.parts[0]) if rel.parts else "",
            "seed": data.get("seed"),
            "lambda_penalty": data.get("lambda"),
            "best_fidelity": data.get("best_fidelity"),
            "final_fidelity": data.get("final_fidelity"),
            "success": data.get("success"),
            "record_count": len(data.get("fidelity_history", [])),
            "source_path": str(rel),
        })

    # Adversarial runs (train_adversarial)
    for path in base_path.glob("**/adversarial/seed_*/adversarial_training_*/architect_fidelities.txt"):
        try:
            with open(path) as f:
                vals = [float(line.strip()) for line in f if line.strip()]
        except Exception:
            continue
        if not vals:
            continue
        seed_label = next((p for p in path.parts if p.startswith("seed_")), "")
        rel = path.relative_to(base_path)
        rows.append({
            "experiment": "train_adversarial",
            "run": str(rel.parts[0]) if rel.parts else "",
            "seed": seed_label,
            "lambda_penalty": None,
            "best_fidelity": max(vals),
            "final_fidelity": vals[-1],
            "success": None,
            "record_count": len(vals),
            "source_path": str(rel),
        })

    if not rows:
        if logger:
            logger.warning("No fidelity data found to summarize.")
        return

    rows_sorted = sorted(rows, key=lambda r: (r["experiment"], r.get("run", ""), str(r.get("seed"))))
    os.makedirs(analysis_dir, exist_ok=True)
    out_path = Path(analysis_dir) / "fidelity_raw_data.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    if logger:
        logger.info("Fidelity summary written to %s", out_path)


def _evaluate_circuit_under_attacks(circuit: "cirq.Circuit", target_state, rate: float, budget: int, max_samples: int):
    """
    Evaluate clean fidelity and sampled attacked fidelities for a circuit.
    Returns dict with clean and attacked stats.
    """
    import numpy as np
    import cirq

    qubits = sorted(circuit.all_qubits())
    ops = list(circuit.all_operations())
    clean_fid = float(fidelity_pure_target(circuit, target_state, qubits))

    combos = []
    max_k = min(budget, len(ops))
    for k in range(1, max_k + 1):
        combos.extend(itertools.combinations(range(len(ops)), k))
    if len(combos) > max_samples:
        combos = random.sample(combos, max_samples)

    def add_noise(indices):
        noisy_ops = []
        for idx, op in enumerate(ops):
            noisy_ops.append(op)
            if idx in indices:
                for q in op.qubits:
                    noisy_ops.append(cirq.DepolarizingChannel(rate).on(q))
        return cirq.Circuit(noisy_ops)

    attacked_vals = []
    for comb in combos:
        noisy_circ = add_noise(set(comb))
        attacked_vals.append(fidelity_pure_target(noisy_circ, target_state, qubits))

    if attacked_vals:
        attacked_mean = float(np.mean(attacked_vals))
        attacked_std = float(np.std(attacked_vals, ddof=0))
        attacked_min = float(np.min(attacked_vals))
        attacked_max = float(np.max(attacked_vals))
    else:
        attacked_mean = attacked_std = attacked_min = attacked_max = None

    return {
        "clean_fidelity": clean_fid,
        "attacked_mean": attacked_mean,
        "attacked_std": attacked_std,
        "attacked_min": attacked_min,
        "attacked_max": attacked_max,
        "n_attacks_evaluated": len(attacked_vals),
    }


def summarize_robustness(base_dir: str, analysis_dir: str, max_samples: int, logger=None):
    """
    Compute robustness under attack for both adversarial (robust) circuits and baseline circuits.
    Produces JSON/CSV in analysis_dir.
    """
    import csv
    import cirq
    import numpy as np
    from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
    from experiments import config as exp_config

    rate = max(SaboteurMultiGateEnv.all_error_rates)
    budget = 3
    random.seed(0)

    base_path = Path(base_dir)
    circuit_entries = []

    # Baseline circuit (single)
    baseline_path = base_path / "baseline" / "circuit_vanilla.json"
    if baseline_path.exists():
        circuit_entries.append({"group": "architect_baseline", "run": "baseline", "seed": None, "path": baseline_path})

    # Adversarial robust champions
    for path in base_path.glob("**/adversarial/seed_*/adversarial_training_*/circuit_robust.json"):
        seed_label = next((p for p in path.parts if p.startswith("seed_")), "")
        run_label = str(path.relative_to(base_path).parts[0]) if path.relative_to(base_path).parts else ""
        circuit_entries.append({"group": "adversarial", "run": run_label, "seed": seed_label, "path": path})

    # QuantumNAS baseline (if produced)
    for path in base_path.glob("**/quantumnas/circuit_quantumnas.json"):
        run_label = str(path.relative_to(base_path).parts[0]) if path.relative_to(base_path).parts else ""
        circuit_entries.append({"group": "quantumnas", "run": run_label, "seed": None, "path": path})

    rows = []
    for entry in sorted(circuit_entries, key=lambda e: (e["group"], e.get("run", ""), str(e.get("seed")), str(e["path"]))):
        path = entry["path"]
        try:
            circuit = cirq.read_json(json_text=path.read_text())
        except Exception as e:
            if logger:
                logger.error("Failed to load circuit at %s: %s", path, e)
            continue
        n_qubits = len(circuit.all_qubits())
        target_state = exp_config.get_target_state(n_qubits)

        stats = _evaluate_circuit_under_attacks(
            circuit=circuit,
            target_state=target_state,
            rate=rate,
            budget=budget,
            max_samples=max_samples,
        )

        rows.append({
            "group": entry["group"],
            "run": entry.get("run"),
            "seed": entry.get("seed"),
            "clean_fidelity": stats["clean_fidelity"],
            "attacked_mean": stats["attacked_mean"],
            "attacked_std": stats["attacked_std"],
            "attacked_min": stats["attacked_min"],
            "attacked_max": stats["attacked_max"],
            "n_attacks_evaluated": stats["n_attacks_evaluated"],
            "circuit_path": str(path.relative_to(base_path)),
        })

    if not rows:
        if logger:
            logger.warning("No circuits found for robustness summary.")
        return

    os.makedirs(analysis_dir, exist_ok=True)
    json_path = Path(analysis_dir) / "robustness_compare.json"
    csv_path = Path(analysis_dir) / "robustness_compare.csv"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if logger:
        logger.info("Robustness comparison written to %s and %s", json_path, csv_path)


def plot_coevolution_per_seed(run_dirs, logger=None):
    """
    Generate single-seed co-evolution plots for each adversarial training run.
    """
    import subprocess

    for idx, run_dir in enumerate(run_dirs):
        seed_label = next((p for p in Path(run_dir).parts if p.startswith("seed_")), f"seed{idx}")
        out_path = os.path.join(os.path.dirname(run_dir), f"coevolution_{seed_label}.png")
        try:
            result = subprocess.run(
                [
                    'python', 'experiments/plot_coevolution.py',
                    '--run-dir', run_dir,
                    '--out', out_path
                ],
                capture_output=True,
                text=True
            )
            if result.stdout and logger:
                logger.info(result.stdout.strip())
            if result.stderr and logger and result.stderr.strip():
                logger.warning(result.stderr.strip())
        except Exception as e:
            if logger:
                logger.error("Error plotting coevolution for %s: %s", run_dir, e)


def run_pipeline(args):
    # Lazy imports of experiment entrypoints
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

    # Effective n_seeds (for lambda sweep, adversarial multi-seed, parameter recovery)
    effective_n_seeds = args.n_seeds if args.n_seeds is not None else exp_config.N_SEEDS

    # Base seed for reproducibility
    base_seed = args.seed if args.seed is not None else 42

    # Top-level metadata
    metadata = {
        'timestamp': timestamp,
        'n_qubits': args.n_qubits,
        'baseline_steps': baseline_steps,
        'saboteur_steps': saboteur_steps,
        'adversarial_gens': adversarial_gens,
        'adversarial_arch_steps': adversarial_arch_steps,
        'seed': base_seed if args.seed is not None else None,
        'n_seeds': effective_n_seeds,
        'source': 'experiments.config',
        'champion_save_last_steps': args.champion_save_last_steps,
        'hall_of_fame_size': args.hall_of_fame_size,
    }
    save_metadata(os.path.join(base, 'metadata.json'), metadata)
    logger.info('Starting pipeline with metadata: %s', metadata)

    # 1) Baseline
    baseline_dir = os.path.join(base, 'baseline')
    os.makedirs(baseline_dir, exist_ok=True)
    if not args.skip_baseline:
        logger.info('Running baseline architect (noiseless)')
        # Seed once for baseline
        if args.seed is not None:
            set_global_seed(base_seed, logger=logger)
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

    # 1b) QuantumNAS baseline (experimental scaffold; off by default)
    quantumnas_dir = os.path.join(base, 'quantumnas')
    os.makedirs(quantumnas_dir, exist_ok=True)
    if getattr(args, "run_quantumnas", False):
        logger.info('Running QuantumNAS baseline')
        circuit_path = None
        try:
            import subprocess
            # Priority: import external circuit if provided
            if args.quantumnas_qasm or args.quantumnas_op_history:
                logger.info('Importing external QuantumNAS circuit')
                import_path = None
                if args.quantumnas_qasm:
                    import_path = Path(args.quantumnas_qasm).expanduser().resolve()
                elif args.quantumnas_op_history:
                    op_path = Path(args.quantumnas_op_history).expanduser().resolve()
                    try:
                        from experiments.export_tq_op_history import main as export_op_history_main
                        # Reuse export script logic via subprocess-like call
                        import sys as _sys
                        _argv_backup = list(_sys.argv)
                        _sys.argv = [
                            "export_tq_op_history",
                            "--op-history",
                            str(op_path),
                            "--qasm-out",
                            str(op_path.with_suffix(".qasm")),
                            "--cirq-out",
                            str(op_path.with_suffix(".json")),
                        ]
                        export_op_history_main()
                        import_path = op_path.with_suffix(".qasm")
                        _sys.argv = _argv_backup
                    except Exception as exc:
                        logger.error("Failed to convert op_history %s: %s", op_path, exc)
                        import_path = None

                if import_path and import_path.exists():
                    from utils.torchquantum_adapter import convert_qasm_file_to_cirq
                    cirq_out = Path(quantumnas_dir) / "circuit_quantumnas.json"
                    convert_qasm_file_to_cirq(import_path, cirq_out)
                    circuit_path = cirq_out
                    logger.info("Imported QuantumNAS circuit from %s -> %s", import_path, cirq_out)
                else:
                    logger.error("No valid QASM/op_history provided for QuantumNAS import.")
            else:
                # Run simple TorchQuantum trainer to produce a circuit from scratch
                task = 'ghz' if exp_config.TARGET_TYPE.lower() == 'ghz' else 'toffoli'
                epochs_default = 200 if task == 'ghz' else 800
                depth_default = 3 if task == 'ghz' else 8
                epochs = args.quantumnas_simple_epochs or epochs_default
                depth = args.quantumnas_simple_depth or depth_default
                lr = args.quantumnas_simple_lr
                logger.info('Running simple TorchQuantum baseline (task=%s, epochs=%d, depth=%d, lr=%.4f)', task, epochs, depth, lr)
                result = subprocess.run(
                    [
                        'python', 'experiments/tq_train_simple_baseline.py',
                        '--task', task,
                        '--epochs', str(epochs),
                        '--lr', str(lr),
                        '--depth', str(depth),
                        '--out-dir', quantumnas_dir,
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.stdout:
                    logger.info(result.stdout.strip())
                if result.stderr and result.stderr.strip():
                    logger.warning(result.stderr.strip())
                expected = Path(quantumnas_dir) / "circuit_quantumnas.json"
                if expected.exists():
                    circuit_path = expected
        except Exception as e:
            logger.error('QuantumNAS baseline failed: %s', e)
        if circuit_path:
            logger.info('QuantumNAS circuit saved to %s', circuit_path)
        else:
            logger.warning('QuantumNAS baseline did not produce a circuit. See logs for details.')
    else:
        logger.info('Skipping QuantumNAS baseline (disabled by default)')

    # 1c) Import external QuantumNAS circuit from QASM or op_history
    if args.quantumnas_qasm or args.quantumnas_op_history:
        logger.info('Importing external QuantumNAS circuit')
        try:
            import_path = None
            if args.quantumnas_qasm:
                import_path = Path(args.quantumnas_qasm).expanduser().resolve()
            elif args.quantumnas_op_history:
                op_path = Path(args.quantumnas_op_history).expanduser().resolve()
                try:
                    from experiments.export_tq_op_history import main as export_op_history_main
                    # Reuse export script logic via subprocess-like call
                    import sys as _sys
                    _argv_backup = list(_sys.argv)
                    _sys.argv = [
                        "export_tq_op_history",
                        "--op-history",
                        str(op_path),
                        "--qasm-out",
                        str(op_path.with_suffix(".qasm")),
                        "--cirq-out",
                        str(op_path.with_suffix(".json")),
                    ]
                    export_op_history_main()
                    import_path = op_path.with_suffix(".qasm")
                    _sys.argv = _argv_backup
                except Exception as exc:
                    logger.error("Failed to convert op_history %s: %s", op_path, exc)
                    import_path = None

            if import_path and import_path.exists():
                from utils.torchquantum_adapter import convert_qasm_file_to_cirq
                cirq_out = Path(quantumnas_dir) / "circuit_quantumnas.json"
                convert_qasm_file_to_cirq(import_path, cirq_out)
                logger.info("Imported QuantumNAS circuit from %s -> %s", import_path, cirq_out)
            else:
                logger.error("No valid QASM/op_history provided for QuantumNAS import.")
        except Exception as exc:
            logger.error("QuantumNAS import failed: %s", exc)

    # 1.5) Lambda Sweep
    lambda_sweep_dir = os.path.join(base, 'lambda_sweep')
    os.makedirs(lambda_sweep_dir, exist_ok=True)
    if not args.skip_lambda_sweep:
        logger.info('Running lambda sweep experiment (ExpPlan Part 1, Exp 1.1) with n_seeds=%d', effective_n_seeds)
        run_lambda_sweep(
            results_dir=lambda_sweep_dir,
            logger=logger,
            training_steps=baseline_steps, 
            n_seeds=effective_n_seeds,
            n_qubits=args.n_qubits
        )
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
                    # Optional: seed for saboteur-only
                    if args.seed is not None:
                        set_global_seed(base_seed + 1000, logger=logger)
                    train_saboteur_only(
                        results_dir=saboteur_dir,
                        n_qubits=args.n_qubits, 
                        saboteur_steps=saboteur_steps,
                        n_steps=saboteur_n_steps, 
                        baseline_circuit_path=vanilla_src
                    )
            except Exception as e:
                logger.error(f'Error loading circuit from {vanilla_src}: {e}. Cannot run saboteur-only.')
    else:
        logger.info('Skipping saboteur-only as requested')

    # 3) Adversarial co-evolution (MULTI-SEED)
    adversarial_dir = os.path.join(base, 'adversarial')
    os.makedirs(adversarial_dir, exist_ok=True)
    adv_training_dirs = []
    canonical_adv_training_dir = None

    if not args.skip_adversarial:
        logger.info(
            'Running adversarial co-evolution for %d seeds (%d generations each)',
            effective_n_seeds, adversarial_gens
        )

        # For each seed, run a full adversarial training
        for k in range(effective_n_seeds):
            seed_val = base_seed + 2000 + k  # offset to avoid overlap with baseline/sab-only
            logger.info(f'Adversarial training: seed index {k} (global seed={seed_val})')
            set_global_seed(seed_val, logger=logger)

            seed_dir = os.path.join(adversarial_dir, f'seed_{k}')
            os.makedirs(seed_dir, exist_ok=True)

            _, _, log_dir = train_adversarial(
                results_dir=seed_dir,
                n_qubits=args.n_qubits,
                n_generations=adversarial_gens,
                architect_steps_per_generation=adversarial_arch_steps,
                saboteur_steps_per_generation=adversarial_sab_steps,
                max_circuit_gates=args.max_circuit_gates,
                fidelity_threshold=args.fidelity_threshold,
                include_rotations=exp_config.INCLUDE_ROTATIONS,
                task_mode=args.task_mode,
                champion_save_last_steps=args.champion_save_last_steps,
                hall_of_fame_size=args.hall_of_fame_size,
            )
            adv_training_dirs.append(log_dir)

        # Choose the first seed's run as canonical for downstream (compare, cross-noise)
        if adv_training_dirs:
            canonical_adv_training_dir = adv_training_dirs[0]
            logger.info(f'Canonical adversarial run (for comparison): {canonical_adv_training_dir}')

        # Multi-seed plotting
        try:
            import subprocess
            logger.info('Plotting multi-seed coevolution from %s', adversarial_dir)
            result = subprocess.run(
                [
                    'python', 'experiments/plot_coevolution_multiseed.py',
                    '--root-dir', adversarial_dir,
                    '--out', os.path.join(adversarial_dir, 'coevolution_multiseed.png')
                ],
                capture_output=True,
                text=True
            )
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        except Exception as e:
            logger.error(f'Error running plot_coevolution_multiseed.py: {e}')

        # Per-seed plotting
        try:
            plot_coevolution_per_seed(adv_training_dirs, logger=logger)
        except Exception as e:
            logger.error('Error running per-seed coevolution plots: %s', e)
    else:
        logger.info('Skipping adversarial as requested')

    # If adversarial training was skipped, attempt to discover existing robust circuits
    if not adv_training_dirs:
        discovered = []
        for seed_dir in sorted(glob.glob(os.path.join(adversarial_dir, 'seed_*'))):
            if not os.path.isdir(seed_dir):
                continue
            candidates = []
            for root, _, files in os.walk(seed_dir):
                if 'circuit_robust.json' in files:
                    candidates.append(os.path.join(root, 'circuit_robust.json'))
            if candidates:
                best = max(candidates, key=os.path.getmtime)
                discovered.append(os.path.dirname(best))
        if discovered:
            adv_training_dirs.extend(discovered)
            canonical_adv_training_dir = canonical_adv_training_dir or discovered[0]
            logger.info('Discovered %d existing adversarial run(s) with robust circuits', len(discovered))

    # 4) Comparison & File Management
    compare_base = os.path.join(base, 'compare')
    vanilla_src = os.path.join(baseline_dir, 'circuit_vanilla.json')
    os.makedirs(compare_base, exist_ok=True)

    # Gather robust circuits from every adversarial seed so downstream steps can use all of them.
    compare_runs = []
    robust_for_downstream = None
    for idx, adv_dir in enumerate(adv_training_dirs):
        run_dir = os.path.join(compare_base, f'run_{idx}')
        os.makedirs(run_dir, exist_ok=True)
        if os.path.exists(vanilla_src):
            shutil.copy(vanilla_src, os.path.join(run_dir, 'circuit_vanilla.json'))

        robust_src = os.path.join(adv_dir, 'circuit_robust.json')
        if os.path.exists(robust_src):
            shutil.copy(robust_src, os.path.join(run_dir, 'circuit_robust.json'))
            compare_runs.append(run_dir)
            if robust_for_downstream is None:
                robust_for_downstream = os.path.join(run_dir, 'circuit_robust.json')
            logger.info(f'Robust circuit from {adv_dir} copied to {run_dir}')
            # Also copy the first robust circuit to adversarial root for convenience
            if idx == 0:
                target_root_robust = os.path.join(adversarial_dir, 'circuit_robust.json')
                shutil.copy(robust_src, target_root_robust)
        else:
            logger.warning(f'No robust circuit found at {robust_src}')

    # Backward compatibility: if no adversarial runs or no robust circuits were found, keep run_0 structure
    if not compare_runs:
        run0 = os.path.join(compare_base, 'run_0')
        os.makedirs(run0, exist_ok=True)
        if os.path.exists(vanilla_src):
            shutil.copy(vanilla_src, os.path.join(run0, 'circuit_vanilla.json'))
        # Try canonical robust fallback
        robust_src = os.path.join(adversarial_dir, 'circuit_robust.json')
        if not os.path.exists(robust_src) and canonical_adv_training_dir:
            robust_src = os.path.join(canonical_adv_training_dir, 'circuit_robust.json')
        if os.path.exists(robust_src):
            shutil.copy(robust_src, os.path.join(run0, 'circuit_robust.json'))
            robust_for_downstream = os.path.join(run0, 'circuit_robust.json')
            logger.info('Robust circuit copied to comparison folder (fallback run_0)')
        else:
            logger.warning('Robust circuit not found for comparison.')
        compare_runs.append(run0)

    # 5) Compare
    if not args.skip_compare:
        # Copy QuantumNAS circuit into compare runs before analysis
        qnas_circuit = Path(quantumnas_dir) / "circuit_quantumnas.json"
        if qnas_circuit.exists():
            try:
                for run_dir in compare_runs:
                    target_path = Path(run_dir) / "circuit_quantumnas.json"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(qnas_circuit.read_text())
                logger.info("QuantumNAS circuit copied into compare runs for analysis.")
            except Exception as exc:
                logger.error("Failed to copy QuantumNAS circuit into compare runs: %s", exc)

        logger.info('Running comparison across runs')
        compare_noise_resilience(
            base_results_dir=compare_base,
            num_runs=len(compare_runs),
            n_qubits=args.n_qubits
        )

    # 6) Parameter Recovery
    param_recovery_dir = os.path.join(base, 'parameter_recovery')
    os.makedirs(param_recovery_dir, exist_ok=True)
    if not args.skip_parameter_recovery:
        if os.path.exists(vanilla_src):
            logger.info('Running parameter recovery experiment')
            robust_to_use = robust_for_downstream if (robust_for_downstream and os.path.exists(robust_for_downstream)) else vanilla_src
            quantumnas_circuit = os.path.join(compare_runs[0], 'circuit_quantumnas.json') if compare_runs else None
            run_parameter_recovery(
                results_dir=param_recovery_dir,
                n_qubits=args.n_qubits,
                baseline_circuit_path=vanilla_src,
                robust_circuit_path=robust_to_use,
                quantumnas_circuit_path=quantumnas_circuit if (quantumnas_circuit and os.path.exists(quantumnas_circuit)) else None,
                n_repetitions=effective_n_seeds,
                base_seed=base_seed,
                logger=logger
            )

    # 7) Cross-Noise
    cross_noise_dir = os.path.join(base, 'cross_noise')
    os.makedirs(cross_noise_dir, exist_ok=True)
    if not args.skip_cross_noise:
        # Use the first available comparison run for cross-noise
        first_run = compare_runs[0] if compare_runs else os.path.join(compare_base, 'run_0')
        vanilla_circuit_path = os.path.join(first_run, 'circuit_vanilla.json')
        robust_circuit_path = os.path.join(first_run, 'circuit_robust.json')
        quantumnas_circuit_path = os.path.join(first_run, 'circuit_quantumnas.json')
        if os.path.exists(vanilla_circuit_path) and os.path.exists(robust_circuit_path):
            logger.info('Running cross-noise robustness experiment')
            run_cross_noise_robustness(
                baseline_circuit_path=vanilla_circuit_path,
                robust_circuit_path=robust_circuit_path,
                output_dir=cross_noise_dir,
                n_qubits=args.n_qubits,
                quantum_nas_circuit_path=quantumnas_circuit_path if os.path.exists(quantumnas_circuit_path) else None,
                logger=logger
            )

    # 8) Analysis summaries (fidelity + robustness under attack)
    analysis_dir = os.path.join(base, 'analysis')
    try:
        summarize_fidelity_outputs(base, analysis_dir, logger=logger)
    except Exception as e:
        logger.error('Error summarizing fidelities: %s', e)
    try:
        summarize_robustness(base, analysis_dir, max_samples=args.attack_samples, logger=logger)
    except Exception as e:
        logger.error('Error summarizing robustness: %s', e)

    logger.info('Pipeline finished. Results in %s', base)


def parse_args():
    p = argparse.ArgumentParser(description='Run the experiment pipeline')
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
    from experiments import config as exp_config  # import here to access defaults
    p.add_argument('--max-circuit-gates', type=int, default=exp_config.MAX_CIRCUIT_TIMESTEPS)
    p.add_argument('--fidelity-threshold', type=float, default=1.1)  # Train by improvement
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--n-seeds', type=int, default=None)
    p.add_argument('--champion-save-last-steps', type=int, default=None,
                   help='During adversarial training, only save champions in the last N architect steps (optional).')
    p.add_argument('--hall-of-fame-size', type=int, default=5,
                   help='Top-k champions to track during adversarial training for final evaluation.')
    p.add_argument(
        '--task-mode',
        type=str,
        default=None,
        choices=['state_preparation', 'unitary_preparation']
    )
    p.add_argument('--attack-samples', type=int, default=3000,
                   help='Max attack placements sampled per circuit when computing robustness summaries')
    # Experimental: QuantumNAS baseline scaffold (off by default)
    p.add_argument('--run-quantumnas', action='store_true',
                   help='Run the QuantumNAS baseline scaffold (placeholder implementation).')
    p.add_argument('--quantumnas-steps', type=int, default=None,
                   help='Override QuantumNAS optimization steps (defaults to baseline architect steps).')
    p.add_argument('--quantumnas-learning-rate', type=float, default=3e-4)
    p.add_argument('--quantumnas-batch-size', type=int, default=256)
    p.add_argument('--quantumnas-temp-start', type=float, default=5.0)
    p.add_argument('--quantumnas-temp-end', type=float, default=0.5)
    p.add_argument('--quantumnas-noise-rate', type=float, default=None,
                   help='Optional depolarizing rate during QuantumNAS training.')
    p.add_argument('--quantumnas-qasm', type=str, default=None,
                   help='Path to an external QuantumNAS QASM file to import as circuit_quantumnas.json.')
    p.add_argument('--quantumnas-op-history', type=str, default=None,
                   help='Path to a TorchQuantum op_history (json/pt) to convert and import as circuit_quantumnas.json.')
    p.add_argument('--quantumnas-simple-epochs', type=int, default=None,
                   help='Epochs for the simple TorchQuantum baseline (default: 200 GHZ, 800 Toffoli).')
    p.add_argument('--quantumnas-simple-depth', type=int, default=None,
                   help='Depth (rotation+CNOT layers) for the simple TorchQuantum baseline (default: 3 GHZ, 8 Toffoli).')
    p.add_argument('--quantumnas-simple-lr', type=float, default=0.05,
                   help='Learning rate for the simple TorchQuantum baseline.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
