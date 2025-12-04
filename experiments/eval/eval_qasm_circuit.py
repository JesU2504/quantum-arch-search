#!/usr/bin/env python3
"""
Evaluate a QASM circuit (e.g., exported from TorchQuantum) against repo targets.

Computes clean fidelity (state or process) and robustness under multi-gate
depolarizing attacks, then saves metrics JSON for downstream analysis.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import itertools
import random

import numpy as np
import cirq

# Add repository root to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from experiments import config as exp_config
from qas_gym.utils import fidelity_pure_target
from utils.metrics import unitary_from_basis_columns, process_fidelity
from utils.torchquantum_adapter import qasm_to_cirq, save_cirq_circuit
from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv


def evaluate_under_attacks(circuit: cirq.Circuit, target_state, rate: float, budget: int, max_samples: int):
    """
    Evaluate clean fidelity and sampled attacked fidelities for a circuit.
    Returns dict with clean and attacked stats.
    """
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


def evaluate_process_fidelity(circuit: cirq.Circuit, ideal_unitary: np.ndarray, qubits) -> float:
    """Compute process fidelity for small circuits (n<=5 recommended)."""
    dim = 2 ** len(qubits)
    columns = []
    sim = cirq.Simulator()
    for idx in range(dim):
        init_bits = [(idx >> (len(qubits) - 1 - b)) & 1 for b in range(len(qubits))]
        prep_ops = [cirq.X(qubits[b]) for b, bit in enumerate(init_bits) if bit == 1]
        test_circuit = cirq.Circuit()
        test_circuit.append(prep_ops)
        test_circuit += circuit
        result = sim.simulate(test_circuit, qubit_order=qubits)
        columns.append(result.final_state_vector)
    U_noisy = unitary_from_basis_columns(columns)
    return float(process_fidelity(ideal_unitary, U_noisy))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a QASM circuit for fidelity and robustness.")
    p.add_argument('--qasm', required=True, help="Path to OpenQASM 2.0 file.")
    p.add_argument('--out-dir', default=None, help="Directory to save Cirq JSON and metrics (default: alongside QASM).")
    p.add_argument('--n-qubits', type=int, default=None, help="Override qubit count (auto from circuit if omitted).")
    p.add_argument('--target-type', type=str, default=exp_config.TARGET_TYPE, choices=['ghz', 'toffoli'])
    p.add_argument('--task-mode', type=str, default=exp_config.TASK_MODE, choices=['state_preparation', 'unitary_preparation'])
    p.add_argument('--attack-rate', type=float, default=None, help="Depolarizing rate per attacked gate (default: saboteur max).")
    p.add_argument('--attack-budget', type=int, default=3, help="Max gates attacked in combination sampling.")
    p.add_argument('--attack-samples', type=int, default=3000, help="Max attack placements sampled.")
    return p.parse_args()


def main():
    args = parse_args()
    qasm_path = Path(args.qasm).expanduser().resolve()
    if not qasm_path.exists():
        raise FileNotFoundError(f"QASM file not found: {qasm_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else qasm_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    circuit = qasm_to_cirq(qasm_path.read_text())
    save_path = out_dir / "circuit_quantumnas.json"
    save_cirq_circuit(circuit, save_path)

    qubits = sorted(circuit.all_qubits())
    n_qubits = args.n_qubits or len(qubits)
    target_state = exp_config.get_target_state(n_qubits, args.target_type)

    rate = args.attack_rate
    if rate is None:
        rate = max(SaboteurMultiGateEnv.all_error_rates)

    robustness = evaluate_under_attacks(
        circuit=circuit,
        target_state=target_state,
        rate=rate,
        budget=args.attack_budget,
        max_samples=args.attack_samples,
    )

    metrics = {
        "n_qubits": n_qubits,
        "target_type": args.target_type,
        "task_mode": args.task_mode,
        "attack_rate": rate,
        "attack_budget": args.attack_budget,
        "attack_samples": args.attack_samples,
    }
    metrics.update(robustness)

    if args.task_mode == 'unitary_preparation':
        ideal = exp_config.get_target_circuit(n_qubits, args.target_type, include_input_prep=True)
        try:
            ideal_unitary = ideal.unitary(qubit_order=qubits)
            metrics["process_fidelity"] = evaluate_process_fidelity(circuit, ideal_unitary, qubits)
        except Exception as exc:
            metrics["process_fidelity_error"] = str(exc)

    metrics_path = out_dir / "metrics_quantumnas.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved Cirq circuit to {save_path}")
    print(f"Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
