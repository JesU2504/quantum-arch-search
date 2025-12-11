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
import cirq

# Add repository root and src to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from qas_gym.utils import fidelity_pure_target
from experiments.quantumnas.train_quantumnas_paper import (
    DEFAULT_SEARCH_CMD,
    SUPPORTED_CLASSIFICATION_DATASETS,
    SUPPORTED_VQE_MOLECULES,
)
from qas_gym.envs.saboteur_env import SUPPORTED_NOISE_FAMILIES
from experiments.analysis.robustness_sweep import sweep_circuit_entries


DEFAULT_COMPARE_ATTACK_MODES = [
    "random_high",
    "asymmetric_noise",
    "over_rotation",
    "amplitude_damping",
    "phase_damping",
    "readout",
]


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


def truncate_circuit_json(json_path: str, max_ops: int, logger=None) -> None:
    """
    Trim a Cirq JSON circuit file to the first max_ops operations in-place.
    This enforces the global gate budget (default 15) for external baselines like QuantumNAS.
    """
    log = logger.info if logger else print
    try:
        circuit = cirq.read_json(json_path)
        ops = list(circuit.all_operations())
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.warning("Could not read circuit %s for truncation: %s", json_path, exc)
        else:
            print(f"[truncate] Could not read {json_path}: {exc}")
        return

    if len(ops) <= max_ops:
        return

    trimmed = cirq.Circuit(ops[:max_ops])
    Path(json_path).write_text(cirq.to_json(trimmed))
    log(f"Truncated circuit at {json_path} from {len(ops)} to {max_ops} gates")


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


def summarize_robustness(
    base_dir: str,
    analysis_dir: str,
    max_samples: int,
    noise_families: list[str],
    budgets: list[int],
    rate: float,
    noise_kwargs: dict | None = None,
    logger=None,
):
    """
    Compute robustness under attack for circuits across multiple noise families/budgets.
    Produces JSON/CSV in analysis_dir.
    """
    import csv
    from experiments import config as exp_config

    random.seed(0)

    base_path = Path(base_dir)
    circuit_entries = []

    # Baseline circuit (single)
    baseline_path = base_path / "baseline" / "circuit_vanilla.json"
    if baseline_path.exists():
        circuit_entries.append({"group": "architect_baseline", "run": "baseline", "seed": None, "path": baseline_path})

    # Adversarial robust champions (various layouts)
    for pattern in [
        "**/adversarial/seed_*/adversarial_training_*/circuit_robust.json",
        "**/adversarial/seed_*/circuit_robust.json",
        "**/adversarial/circuit_robust.json",
        "**/compare/run_*/circuit_robust.json",
    ]:
        for path in base_path.glob(pattern):
            seed_label = next((p for p in path.parts if p.startswith("seed_")), "")
            try:
                run_label = str(path.relative_to(base_path).parts[0]) if path.relative_to(base_path).parts else ""
            except Exception:
                run_label = ""
            circuit_entries.append({"group": "adversarial", "run": run_label, "seed": seed_label, "path": path})

    # QuantumNAS baseline (if produced)
    for path in base_path.glob("**/quantumnas/circuit_quantumnas.json"):
        try:
            run_label = str(path.relative_to(base_path).parts[0]) if path.relative_to(base_path).parts else ""
        except Exception:
            run_label = ""
        circuit_entries.append({"group": "quantumnas", "run": run_label, "seed": None, "path": path})

    target_state_fn = exp_config.get_target_state
    rows = sweep_circuit_entries(
        circuit_entries=sorted(circuit_entries, key=lambda e: (e["group"], e.get("run", ""), str(e.get("seed")), str(e["path"]))),
        noise_families=noise_families,
        budgets=budgets,
        rate=rate,
        max_samples=max_samples,
        target_state_fn=target_state_fn,
        noise_kwargs=noise_kwargs,
    )

    if not rows:
        if logger:
            logger.warning("No circuits found for robustness summary.")
        return

    os.makedirs(analysis_dir, exist_ok=True)
    json_path = Path(analysis_dir) / "robustness_sweep.json"
    csv_path = Path(analysis_dir) / "robustness_sweep.csv"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Backward-compatible single-scenario summary (first family/budget)
    primary_family = noise_families[0] if noise_families else "depolarizing"
    primary_budget = budgets[0] if budgets else 1
    filtered = [r for r in rows if r["noise_family"] == primary_family and r["attack_budget"] == primary_budget]
    if filtered:
        legacy_json = Path(analysis_dir) / "robustness_compare.json"
        legacy_csv = Path(analysis_dir) / "robustness_compare.csv"
        with legacy_json.open("w") as f:
            json.dump(filtered, f, indent=2)
        with legacy_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(filtered[0].keys()))
            writer.writeheader()
            writer.writerows(filtered)
        if logger:
            logger.info(
                "Robustness sweep written to %s/%s (legacy compare uses family=%s, budget=%s)",
                json_path,
                csv_path,
                primary_family,
                primary_budget,
            )
    elif logger:
        logger.info("Robustness sweep written to %s and %s", json_path, csv_path)


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
                    'python', 'experiments/analysis/plot_coevolution.py',
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
    from experiments.architect.train_architect import train_baseline_architect
    from experiments.adversarial.train_saboteur_only import train_saboteur_only
    from experiments.adversarial.train_adversarial import train_adversarial
    from experiments.analysis.compare_circuits import compare_noise_resilience
    from experiments.architect.lambda_sweep import run_lambda_sweep
    from experiments.analysis.parameter_recovery import run_parameter_recovery
    from experiments.analysis.cross_noise_robustness import run_cross_noise_robustness
    
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

    quantumnas_paths: list[str] = []
    # 1) Baseline (RL) per seed
    baseline_dir = os.path.join(base, 'baseline')
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_paths: list[str] = []
    if not args.skip_baseline:
        logger.info('Running baseline architect (noiseless) for %d seed(s)', effective_n_seeds)
        for k in range(effective_n_seeds):
            seed_val = base_seed + k if args.seed is not None else None
            if seed_val is not None:
                set_global_seed(seed_val, logger=logger)
            seed_dir = os.path.join(baseline_dir, f'seed_{k}')
            os.makedirs(seed_dir, exist_ok=True)
            train_baseline_architect(
                results_dir=seed_dir,
                n_qubits=args.n_qubits,
                architect_steps=baseline_steps,
                n_steps=baseline_n_steps,
                include_rotations=exp_config.INCLUDE_ROTATIONS,
                task_mode=args.task_mode
            )
            candidate = os.path.join(seed_dir, 'circuit_vanilla.json')
            if os.path.exists(candidate):
                baseline_paths.append(candidate)
        if not baseline_paths:
            logger.warning('No baseline circuits produced.')
    else:
        logger.info('Skipping baseline as requested')

    # 1b) QuantumNAS baseline (experimental scaffold; off by default)
    quantumnas_dir = os.path.join(base, 'quantumnas')
    os.makedirs(quantumnas_dir, exist_ok=True)
    if getattr(args, "run_quantumnas_paper", False):
        logger.info('Running QuantumNAS paper benchmark')
        circuit_path = None
        try:
            import subprocess

            paper_task = args.quantumnas_paper_task
            if paper_task == 'auto':
                paper_task = 'vqe' if args.task_mode == 'vqe' else 'classification'

            cmd = [
                sys.executable,
                'experiments/quantumnas/train_quantumnas_paper.py',
                '--task', paper_task,
                '--out-dir', quantumnas_dir,
            ]
            if paper_task == 'classification':
                cmd.extend(['--dataset', args.quantumnas_paper_dataset])
            else:
                cmd.extend(['--vqe-molecule', args.quantumnas_paper_molecule])
            if args.seed is not None:
                cmd.extend(['--seed', str(base_seed)])
            if args.quantumnas_search_cmd:
                cmd.extend(['--search-cmd', args.quantumnas_search_cmd])
            if args.quantumnas_paper_config:
                cmd.extend(['--config', args.quantumnas_paper_config])
            if args.quantumnas_paper_max_epochs is not None:
                cmd.extend(['--max-epochs', str(args.quantumnas_paper_max_epochs)])
            if args.quantumnas_paper_extra:
                cmd.extend(args.quantumnas_paper_extra)

            logger.info('Executing: %s', ' '.join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                logger.info(result.stdout.strip())
            if result.stderr and result.stderr.strip():
                logger.warning(result.stderr.strip())
            expected = Path(quantumnas_dir) / "circuit_quantumnas.json"
            if expected.exists():
                circuit_path = expected
        except Exception as e:
            logger.error('QuantumNAS paper benchmark failed: %s', e)
        if circuit_path:
            logger.info('QuantumNAS circuit saved to %s', circuit_path)
        else:
            logger.warning('QuantumNAS paper benchmark did not produce a circuit. See logs for details.')
    elif getattr(args, "run_quantumnas", False):
        logger.info('Running HEA baseline per seed')
        quantumnas_paths = []
        try:
            import subprocess
            task = 'ghz' if exp_config.TARGET_TYPE.lower() == 'ghz' else 'toffoli'
            epochs_default = 400 if task == 'ghz' else 2000
            depth_default = max(2, 4 if task == 'ghz' else 12)
            augment_depth = True
            epochs = args.quantumnas_simple_epochs or epochs_default
            depth = args.quantumnas_simple_depth or depth_default
            lr = args.quantumnas_simple_lr
            for k in range(effective_n_seeds):
                seed_val = base_seed + 3000 + k if base_seed is not None else None
                seed_dir = Path(quantumnas_dir) / f"seed_{k}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    'python', 'experiments/architect/tq_train_simple_baseline.py',
                    '--task', task,
                    '--epochs', str(epochs),
                    '--lr', str(lr),
                    '--depth', str(depth),
                    '--max-gates', str(args.max_circuit_gates),
                    '--out-dir', str(seed_dir),
                ]
                if task == 'ghz':
                    cmd.extend(['--n-qubits', str(args.n_qubits)])
                if augment_depth:
                    cmd.append('--augment-depth')
                if seed_val is not None:
                    cmd.extend(['--seed', str(seed_val)])
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.stdout:
                    logger.info(result.stdout.strip())
                if result.stderr and result.stderr.strip():
                    logger.warning(result.stderr.strip())
                expected = seed_dir / "circuit_quantumnas.json"
                if expected.exists():
                    quantumnas_paths.append(str(expected))
            if quantumnas_paths:
                Path(quantumnas_dir, "circuit_quantumnas.json").write_text(Path(quantumnas_paths[0]).read_text())
            else:
                logger.warning('HEA baseline did not produce any circuits.')
        except Exception as e:
            logger.error('HEA baseline failed: %s', e)
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
                    from experiments.quantumnas.export_tq_op_history import main as export_op_history_main
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

    # Enforce gate budget on QuantumNAS circuit (if produced)
    qnas_json = Path(quantumnas_dir) / "circuit_quantumnas.json"
    if qnas_json.exists():
        truncate_circuit_json(str(qnas_json), max_ops=args.max_circuit_gates, logger=logger)

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
        vanilla_src = baseline_paths[0] if baseline_paths else os.path.join(baseline_dir, 'circuit_vanilla.json')
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
                saboteur_noise_family=args.saboteur_noise_family,
                saboteur_error_rates=args.saboteur_error_rates,
                saboteur_budget=args.saboteur_budget,
                saboteur_budget_fraction=args.saboteur_budget_fraction,
                saboteur_start_budget_scale=args.saboteur_start_budget_scale,
                alpha_start=args.alpha_start,
                alpha_end=args.alpha_end,
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
                    'python', 'experiments/analysis/plot_coevolution_multiseed.py',
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
    vanilla_fallback = baseline_paths[0] if baseline_paths else os.path.join(baseline_dir, 'circuit_vanilla.json')
    os.makedirs(compare_base, exist_ok=True)

    # Gather robust circuits from every adversarial seed so downstream steps can use all of them.
    compare_runs = []
    robust_for_downstream = None
    for idx, adv_dir in enumerate(adv_training_dirs):
        run_dir = os.path.join(compare_base, f'run_{idx}')
        os.makedirs(run_dir, exist_ok=True)
        # baseline per seed (fallback to first)
        vanilla_src = baseline_paths[idx % len(baseline_paths)] if baseline_paths else vanilla_fallback
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
        vanilla_src = baseline_paths[0] if baseline_paths else vanilla_fallback
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
        if quantumnas_paths:
            qnas_fallback = Path(quantumnas_paths[0])
        else:
            qnas_fallback = qnas_circuit if qnas_circuit.exists() else None
        if qnas_fallback:
            try:
                for idx, run_dir in enumerate(compare_runs):
                    src = Path(quantumnas_paths[idx % len(quantumnas_paths)]) if quantumnas_paths else qnas_fallback
                    target_path = Path(run_dir) / "circuit_quantumnas.json"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(src.read_text())
                    truncate_circuit_json(str(target_path), max_ops=args.max_circuit_gates, logger=logger)
                logger.info("HEA baseline circuits copied into compare runs for analysis.")
            except Exception as exc:
                logger.error("Failed to copy HEA circuits into compare runs: %s", exc)

        logger.info('Running comparison across runs')
        mitigation_kwargs = {}
        if args.rc_zne_scales:
            mitigation_kwargs["rc_zne_scales"] = tuple(args.rc_zne_scales)
        if args.mitigation_mode == "rc_zne":
            mitigation_kwargs["rc_zne_fit"] = args.rc_zne_fit
            mitigation_kwargs["rc_zne_reps"] = args.rc_zne_reps
        compare_noise_resilience(
            base_results_dir=compare_base,
            num_runs=len(compare_runs),
            n_qubits=args.n_qubits,
            samples=args.compare_samples if args.compare_samples is not None else 32,
            attack_modes=args.compare_attack_modes if args.compare_attack_modes else DEFAULT_COMPARE_ATTACK_MODES,
            mitigation_mode=args.mitigation_mode,
            **mitigation_kwargs,
        )

    # 6) Parameter Recovery
    param_recovery_dir = os.path.join(base, 'parameter_recovery')
    os.makedirs(param_recovery_dir, exist_ok=True)
    if not args.skip_parameter_recovery:
        if os.path.exists(vanilla_src):
            logger.info('Running parameter recovery experiment')
            robust_to_use = robust_for_downstream if (robust_for_downstream and os.path.exists(robust_for_downstream)) else vanilla_src
            quantumnas_circuit = os.path.join(compare_runs[0], 'circuit_quantumnas.json') if compare_runs else None
            param_recovery_kwargs = {
                "results_dir": param_recovery_dir,
                "n_qubits": args.n_qubits,
                "baseline_circuit_path": vanilla_src,
                "robust_circuit_path": robust_to_use,
                "n_repetitions": effective_n_seeds,
                "base_seed": base_seed,
                "logger": logger,
            }
            if quantumnas_circuit and os.path.exists(quantumnas_circuit):
                param_recovery_kwargs["quantumnas_circuit_path"] = quantumnas_circuit
            run_parameter_recovery(**param_recovery_kwargs)

    # 7) Cross-Noise
    cross_noise_dir = os.path.join(base, 'cross_noise')
    os.makedirs(cross_noise_dir, exist_ok=True)
    if not args.skip_cross_noise:
        # Use all available comparison runs for cross-noise (aggregate across seeds)
        baseline_list = [os.path.join(rd, 'circuit_vanilla.json') for rd in compare_runs if os.path.exists(os.path.join(rd, 'circuit_vanilla.json'))]
        robust_list = [os.path.join(rd, 'circuit_robust.json') for rd in compare_runs if os.path.exists(os.path.join(rd, 'circuit_robust.json'))]
        qnas_list = [os.path.join(rd, 'circuit_quantumnas.json') for rd in compare_runs if os.path.exists(os.path.join(rd, 'circuit_quantumnas.json'))]
        if baseline_list and robust_list:
            logger.info('Running cross-noise robustness experiment (aggregated across %d seeds)', len(compare_runs))
            run_cross_noise_robustness(
                baseline_circuit_path=baseline_list,
                robust_circuit_path=robust_list,
                output_dir=cross_noise_dir,
                n_qubits=args.n_qubits,
                quantum_nas_circuit_path=qnas_list if qnas_list else None,
                n_seeds=len(compare_runs),
                base_seed=base_seed,
                logger=logger,
                randomized_compile_flag=args.randomized_compiling,
            )

    # 8) Hardware-style evaluation on Fake IBM backends (optional)
    if args.run_hw_eval and not getattr(args, "skip_hw_eval", False):
        try:
            from experiments.analysis.qiskit_hw_eval import run_hw_eval, parse_success_bitstrings
            logger.info("Running hardware-style eval on backends: %s", args.hw_backends)
            hw_output_dir = os.path.join(base, "hardware_eval")
            success_bits = parse_success_bitstrings("", args.n_qubits)
            # Build per-run lists
            baseline_list = [os.path.join(rd, 'circuit_vanilla.json') for rd in compare_runs if os.path.exists(os.path.join(rd, 'circuit_vanilla.json'))]
            robust_list = [os.path.join(rd, 'circuit_robust.json') for rd in compare_runs if os.path.exists(os.path.join(rd, 'circuit_robust.json'))]
            qnas_list = [os.path.join(rd, 'circuit_quantumnas.json') for rd in compare_runs if os.path.exists(os.path.join(rd, 'circuit_quantumnas.json'))]
            run_hw_eval(
                baseline_circuits=baseline_list,
                robust_circuits=robust_list,
                quantumnas_circuits=qnas_list,
                backends=args.hw_backends,
                shots=args.hw_shots,
                opt_level=args.hw_opt_level,
                seed=base_seed,
                target_bitstrings=success_bits,
                output_dir=hw_output_dir,
                randomized_compile_flag=args.randomized_compiling,
                readout_mitigation=args.hw_readout_mitigation,
            )
        except Exception as exc:
            logger.error("Hardware eval failed: %s", exc)

    # 8) Analysis summaries (fidelity + robustness under attack)
    analysis_dir = os.path.join(base, 'analysis')
    try:
        summarize_fidelity_outputs(base, analysis_dir, logger=logger)
    except Exception as e:
        logger.error('Error summarizing fidelities: %s', e)
    try:
        from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv

        robustness_families = args.robustness_noise_families or list(
            dict.fromkeys([args.saboteur_noise_family, "depolarizing"])
        )
        default_budget = args.saboteur_budget if args.saboteur_budget is not None else 3
        robustness_budgets = args.robustness_budgets or [1, default_budget]
        robustness_budgets = [b for b in robustness_budgets if b is not None and b >= 0]
        if not robustness_budgets:
            robustness_budgets = [default_budget or 1]
        robustness_rate = (
            args.robustness_rate
            if args.robustness_rate is not None
            else (max(args.saboteur_error_rates) if args.saboteur_error_rates else max(SaboteurMultiGateEnv.all_error_rates))
        )
        summarize_robustness(
            base,
            analysis_dir,
            max_samples=args.attack_samples,
            noise_families=robustness_families,
            budgets=robustness_budgets,
            rate=robustness_rate,
            logger=logger,
        )
    except Exception as e:
        logger.error('Error summarizing robustness: %s', e)

    # 9) Plots for realistic robustness and hardware (best-effort)
    try:
        import subprocess
        robust_eval = os.path.join(base, 'compare', 'robust_eval.json')
        if os.path.exists(robust_eval):
            subprocess.run([
                sys.executable, 'experiments/analysis/plot_robustness_realistic.py',
                '--robust-eval', robust_eval,
                '--out', os.path.join(analysis_dir, 'robustness_realistic.png'),
            ], check=True)
        hw_json = os.path.join(base, 'hardware_eval', 'hardware_eval_results.json')
        if os.path.exists(hw_json):
            subprocess.run([
                sys.executable, 'experiments/analysis/plot_hw_fidelity.py',
                '--results-json', hw_json,
                '--out', os.path.join(base, 'hardware_eval', 'hardware_eval_plot.png'),
            ], check=True)
    except Exception as e:
        logger.error('Error generating plots: %s', e)

    logger.info('Pipeline finished. Results in %s', base)


def parse_args():
    def _parse_error_rates(value):
        if value is None:
            return None
        try:
            return [float(x) for x in value.split(",") if x.strip() != ""]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid saboteur-error-rates '{value}': {exc}") from exc
    def _parse_int_list(value):
        if value is None:
            return None
        try:
            return [int(x) for x in value.split(",") if x.strip() != ""]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid budget list '{value}': {exc}") from exc

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
    p.add_argument('--saboteur-noise-family', type=str, default='depolarizing',
                   choices=sorted(SUPPORTED_NOISE_FAMILIES),
                   help='Noise family used by the saboteur (for training + evaluation).')
    p.add_argument('--saboteur-error-rates', type=_parse_error_rates, default=None,
                   help='Comma-separated list of saboteur error rates (e.g., "0.0,0.01,0.02").')
    p.add_argument('--saboteur-budget', type=int, default=3,
                   help='Max gates the saboteur can attack per episode (before fractional scaling).')
    p.add_argument('--saboteur-budget-fraction', type=float, default=0.2,
                   help='Fractional attack budget relative to circuit length (set None to disable).')
    p.add_argument('--saboteur-start-budget-scale', type=float, default=0.3,
                   help='Initial scale for the saboteur budget ramp (1.0 disables ramp-up).')
    p.add_argument('--alpha-start', type=float, default=0.6,
                   help='Starting alpha for clean vs attacked reward mixing.')
    p.add_argument('--alpha-end', type=float, default=0.0,
                   help='Ending alpha for clean vs attacked reward mixing.')
    p.add_argument('--robustness-noise-families', nargs='+', default=None,
                   choices=sorted(SUPPORTED_NOISE_FAMILIES),
                   help='Noise families to sweep in robustness summary (default: saboteur noise + depolarizing).')
    p.add_argument('--robustness-budgets', type=_parse_int_list, default=None,
                   help='Comma-separated attack budgets to sweep in robustness summary (default: 1 and saboteur budget).')
    p.add_argument('--robustness-rate', type=float, default=None,
                   help='Noise rate used in robustness sweep (default: max saboteur error rate).')
    p.add_argument('--attack-samples', type=int, default=3000,
                   help='Max attack placements sampled per circuit when computing robustness summaries')
    # Experimental: QuantumNAS paper benchmark harness
    p.add_argument('--run-quantumnas-paper', action='store_true',
                   help='Run the official QuantumNAS paper benchmark via train_quantumnas_paper.py and export circuit_quantumnas.json.')
    p.add_argument('--quantumnas-paper-task', type=str, default='auto', choices=['auto', 'classification', 'vqe'],
                   help='Paper task to launch in the QuantumNAS harness (classification, VQE, or auto to follow task_mode).')
    dataset_choices = ', '.join(sorted(SUPPORTED_CLASSIFICATION_DATASETS.keys()))
    p.add_argument(
        '--quantumnas-paper-dataset',
        type=str,
        default='mnist4',
        choices=sorted(SUPPORTED_CLASSIFICATION_DATASETS.keys()),
        help=(
            'Dataset forwarded to the QuantumNAS paper harness for classification tasks '
            f'(options: {dataset_choices}).'
        ),
    )
    molecule_choices = ', '.join(sorted(SUPPORTED_VQE_MOLECULES.keys()))
    p.add_argument(
        '--quantumnas-paper-molecule',
        type=str,
        default='H2',
        choices=sorted(SUPPORTED_VQE_MOLECULES.keys()),
        help=(
            'Molecule forwarded to the QuantumNAS paper harness for VQE tasks '
            f'(options: {molecule_choices}).'
        ),
    )
    p.add_argument('--quantumnas-paper-max-epochs', type=int, default=None,
                   help='Optional epoch override passed to the QuantumNAS paper harness.')
    p.add_argument('--quantumnas-search-cmd', type=str, default=DEFAULT_SEARCH_CMD,
                   help='QuantumNAS search command invoked by the paper harness (CLI string).')
    p.add_argument('--quantumnas-paper-config', type=str, default=None,
                   help='Optional config file forwarded to the QuantumNAS paper harness CLI.')
    p.add_argument('--quantumnas-paper-extra', nargs='*', default=None,
                   help='Additional arguments forwarded verbatim to the QuantumNAS paper harness CLI.')
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
    p.add_argument('--run-hw-eval', action='store_true',
                   help='Run IBM-style hardware evaluation (Fake backends via Qiskit Aer).')
    p.add_argument('--skip-hw-eval', action='store_true',
                   help='Skip hardware-style evaluation even if --run-hw-eval is set (compatibility flag).')
    p.add_argument('--hw-backends', type=str, nargs='+', default=['fake_quito', 'fake_belem'],
                   help='Backends for hardware eval (e.g., fake_quito fake_belem fake_athens fake_yorktown).')
    p.add_argument('--hw-shots', type=int, default=4096, help='Shots for hardware eval.')
    p.add_argument('--hw-opt-level', type=int, default=3, help='Transpiler optimization level for hardware eval.')
    p.add_argument('--mitigation-mode', choices=['none', 'twirl', 'rc_zne'], default='none',
                   help="Mitigation strategy for compare/plot steps: 'none', 'twirl', or 'rc_zne' (randomized compiling + zero-noise extrapolation).")
    p.add_argument('--rc-zne-scales', type=float, nargs='+', default=None,
                   help='Noise scale factors for rc_zne mitigation (default: 1.0 1.5 2.0).')
    p.add_argument('--rc-zne-fit', type=str, default='linear', choices=['linear', 'quadratic'],
                   help='Extrapolation model for rc_zne mitigation (default: linear).')
    p.add_argument('--rc-zne-reps', type=int, default=1,
                   help='Number of RC draws averaged per scale before rc_zne extrapolation (default: 1).')
    p.add_argument('--hw-readout-mitigation', action='store_true', default=False,
                   help='Enable readout error mitigation for hardware eval (matrix inversion via Ignis).')
    p.add_argument('--randomized-compiling', action='store_true', default=False,
                   help='Deprecated alias for --mitigation-mode twirl. Retained for backward compatibility.')
    p.add_argument(
        '--compare-attack-modes',
        nargs='+',
        default=None,
        help=(
            'Attack modes for compare_circuits (defaults to random_high asymmetric_noise '
            'over_rotation amplitude_damping phase_damping readout).'
        ),
    )
    p.add_argument('--compare-samples', type=int, default=None,
                   help='Number of saboteur attack samples per circuit for compare_circuits (default 32).')
    args = p.parse_args()
    if getattr(args, "randomized_compiling", False) and args.mitigation_mode == 'none':
        args.mitigation_mode = 'twirl'
    if args.mitigation_mode != 'twirl':
        args.randomized_compiling = False
    return args


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
