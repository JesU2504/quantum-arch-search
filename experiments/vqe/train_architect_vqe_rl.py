#!/usr/bin/env python3
"""
RL-based VQE architecture search using ArchitectEnv with a VQE reward.

This reuses the ArchitectEnv (originally for state/unitary prep) but switches
task_mode="vqe" so the reward is -energy (minus a small complexity penalty).

The policy is trained with PPO (stable-baselines3) on discrete gate actions
including rotations (Rx/Ry/Rz). After training, we roll out one deterministic
episode to extract the circuit and export QASM/Cirq JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys

# Ensure repo root and src on path for standalone execution
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_src_root = _repo_root / "src"
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

import numpy as np
import cirq
from stable_baselines3 import PPO

from qas_gym.envs.architect_env import ArchitectEnv
from utils.standard_hamiltonians import get_standard_hamiltonian
from utils.torchquantum_adapter import convert_qasm_file_to_cirq
from src.utils.metrics import state_energy
from experiments import config as exp_config


_MOLECULE_QUBITS = {
    "H2": 2,
    "HeH+": 2,
    "LiH": 4,
    "BeH2": 6,
}


def _canonicalize_circuit(circ: cirq.Circuit, n_qubits: int) -> cirq.Circuit:
    """Rewire circuit onto LineQubit[0..n_qubits-1] and add idle identities for missing wires."""
    current = sorted(circ.all_qubits())
    target = [cirq.LineQubit(i) for i in range(n_qubits)]
    if len(current) > n_qubits:
        return circ  # do not drop qubits; leave as-is
    mapping = {current[i]: target[i] for i in range(len(current))}
    rewired = circ.transform_qubits(lambda q: mapping.get(q, q))
    if len(current) < n_qubits:
        # Add explicit identities so diagrams/QASM include the full register
        rewired += cirq.Circuit([cirq.I(q) for q in target[len(current) :]])
    return rewired


def _zz_chain_hamiltonian(n_qubits: int) -> np.ndarray:
    """Simple ZZ chain Hamiltonian sum_i Z_i Z_{i+1}."""
    import functools

    Z = np.array([[1, 0], [0, -1]], dtype=float)
    I = np.eye(2, dtype=float)

    def kron_all(mats):
        return functools.reduce(np.kron, mats)

    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=float)
    for i in range(n_qubits - 1):
        mats = []
        for j in range(n_qubits):
            mats.append(Z if j == i or j == i + 1 else I)
        H += kron_all(mats)
    return H


def _dump_qasm(circ: cirq.Circuit) -> str:
    try:
        return cirq.qasm(circ)
    except Exception:
        return circ.to_text_diagram(use_unicode=False)


def make_env(molecule: str, max_gates: int, complexity_penalty: float):
    try:
        ham = get_standard_hamiltonian(molecule)
        n_qubits = ham["n_qubits"]
        h_mat = ham["matrix"]
        hf_energy = ham.get("hf_energy", float("nan"))
        fci_energy = ham.get("fci_energy", float(np.min(np.linalg.eigvalsh(h_mat))))
    except Exception:
        n_qubits = _MOLECULE_QUBITS.get(molecule, 3)
        h_mat = _zz_chain_hamiltonian(n_qubits)
        # Fallback HF/FCI energies from simple reference/diagonalization
        target = np.zeros(2**n_qubits, dtype=complex)
        target[0] = 1.0
        hf_energy = float(state_energy(target, h_mat))
        fci_energy = float(np.min(np.linalg.eigvalsh(h_mat)))
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1.0  # |0...0>
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    action_gates = exp_config.get_action_gates(qubits, include_rotations=True)
    env = ArchitectEnv(
        target=target,
        fidelity_threshold=2.0,  # effectively never terminate early on fidelity
        reward_penalty=0.0,
        max_timesteps=max_gates,
        include_rotations=True,
        qubits=qubits,
        action_gates=action_gates,
        task_mode="vqe",
        hamiltonian_matrix=h_mat,
        complexity_penalty_weight=complexity_penalty,
    )
    return env, {"n_qubits": n_qubits, "matrix": h_mat, "hf_energy": hf_energy, "fci_energy": fci_energy}


def rollout_best(model: PPO, env: ArchitectEnv):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    circ = env._get_cirq()
    energy = info.get("energy")
    return circ, energy


def save_artifacts(out_dir: Path, circ: cirq.Circuit, molecule: str, energy: float, ham_info: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    circ = _canonicalize_circuit(circ, ham_info["n_qubits"])
    qasm_path = out_dir / "circuit_architect_vqe.qasm"
    qasm_path.write_text(_dump_qasm(circ))
    convert_qasm_file_to_cirq(qasm_path, out_dir / "circuit_architect_vqe.json")
    meta = {
        "molecule": molecule,
        "energy": energy,
        "n_qubits": ham_info["n_qubits"],
        "hf_energy": ham_info["hf_energy"],
        "fci_energy": ham_info["fci_energy"],
    }
    (out_dir / "results.json").write_text(json.dumps(meta, indent=2))


def parse_args():
    ap = argparse.ArgumentParser(description="PPO on ArchitectEnv with VQE reward.")
    ap.add_argument("--molecule", default="H2", choices=["H2", "HeH+", "LiH", "BeH2"])
    ap.add_argument("--max-gates", type=int, default=12)
    ap.add_argument("--total-timesteps", type=int, default=50000)
    ap.add_argument("--complexity-penalty", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()

    env, ham = make_env(args.molecule, args.max_gates, args.complexity_penalty)
    model = PPO("MlpPolicy", env, learning_rate=args.lr, verbose=0)
    model.learn(total_timesteps=args.total_timesteps)

    circ, energy = rollout_best(model, env)
    save_artifacts(out_dir, circ, args.molecule, energy, ham)
    print(f"Saved RL architect VQE circuit to {out_dir} (energy {energy:.6f} Ha)")


if __name__ == "__main__":
    main()
