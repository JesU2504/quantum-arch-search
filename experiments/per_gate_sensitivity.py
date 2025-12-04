#!/usr/bin/env python3
"""
Per-gate sensitivity analysis for a circuit under a chosen noise model.

Evaluates how much fidelity drops when noise is injected after each gate
individually. Useful for understanding why circuits win/lose under different
noise models (over-rotation vs asymmetric Pauli).
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add repository root to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import cirq
import numpy as np

from qas_gym.utils import load_circuit, fidelity_pure_target
from experiments import config


def apply_single_gate_noise(circuit: cirq.Circuit, gate_index: int, mode: str,
                            epsilon: float, p_x: float, p_y: float, p_z: float) -> cirq.Circuit:
    ops = list(circuit.all_operations())
    noisy_ops = []
    for idx, op in enumerate(ops):
        noisy_ops.append(op)
        if idx == gate_index:
            if mode == "over_rotation":
                for q in op.qubits:
                    noisy_ops.append(cirq.rx(epsilon).on(q))
            elif mode == "asymmetric":
                for q in op.qubits:
                    noisy_ops.append(cirq.asymmetric_depolarize(p_x=p_x, p_y=p_y, p_z=p_z).on(q))
    return cirq.Circuit(noisy_ops)


def analyze_circuit(path: str, n_qubits: int, mode: str, epsilon: float, p_x: float, p_y: float, p_z: float):
    circuit = load_circuit(path)
    qubits = sorted(circuit.all_qubits())
    target_state = config.get_target_state(n_qubits)
    clean_fid = fidelity_pure_target(circuit, target_state, qubits)
    drops = []
    for idx, _ in enumerate(circuit.all_operations()):
        noisy = apply_single_gate_noise(circuit, idx, mode, epsilon, p_x, p_y, p_z)
        fid = fidelity_pure_target(noisy, target_state, qubits)
        drops.append({
            "gate_index": idx,
            "drop": float(clean_fid - fid),
            "noisy_fidelity": float(fid),
        })
    drops_sorted = sorted(drops, key=lambda d: d["drop"], reverse=True)
    return {
        "path": path,
        "mode": mode,
        "params": {"epsilon": epsilon, "p_x": p_x, "p_y": p_y, "p_z": p_z},
        "clean_fidelity": float(clean_fid),
        "drops": drops_sorted,
    }


def main():
    ap = argparse.ArgumentParser(description="Per-gate sensitivity analysis.")
    ap.add_argument("--circuit", type=str, required=True, help="Path to circuit JSON")
    ap.add_argument("--n-qubits", type=int, required=True, help="Number of qubits")
    ap.add_argument("--mode", type=str, default="over_rotation", choices=["over_rotation", "asymmetric"],
                    help="Noise mode to inject after each gate")
    ap.add_argument("--epsilon", type=float, default=0.1, help="Over-rotation angle (radians) if mode=over_rotation")
    ap.add_argument("--p-x", type=float, default=0.05, dest="p_x", help="Asymmetric noise p_x if mode=asymmetric")
    ap.add_argument("--p-y", type=float, default=0.0, dest="p_y", help="Asymmetric noise p_y if mode=asymmetric")
    ap.add_argument("--p-z", type=float, default=0.0, dest="p_z", help="Asymmetric noise p_z if mode=asymmetric")
    ap.add_argument("--output", type=str, default=None, help="Output JSON path (default: results/per_gate_sensitivity_<ts>.json)")
    args = ap.parse_args()

    res = analyze_circuit(args.circuit, args.n_qubits, args.mode, args.epsilon, args.p_x, args.p_y, args.p_z)

    out_path = args.output or os.path.join(
        "results",
        f"per_gate_sensitivity_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    # Print top-5 drops for quick view
    top5 = res["drops"][:5]
    print(f"Clean fidelity: {res['clean_fidelity']:.6f}")
    print(f"Top 5 gate drops ({args.mode}):")
    for d in top5:
        print(f"  gate {d['gate_index']}: drop {d['drop']:.4f}, noisy fidelity {d['noisy_fidelity']:.4f}")
    print(f"Saved full results to {out_path}")


if __name__ == "__main__":
    main()
