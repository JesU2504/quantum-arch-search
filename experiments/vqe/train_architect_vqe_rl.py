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

import numpy as np
import cirq
from stable_baselines3 import PPO

from qas_gym.envs.architect_env import ArchitectEnv
from utils.standard_hamiltonians import get_standard_hamiltonian
from utils.torchquantum_adapter import convert_qasm_file_to_cirq


def _dump_qasm(circ: cirq.Circuit) -> str:
    try:
        return cirq.qasm(circ)
    except Exception:
        return circ.to_text_diagram(use_unicode=False)


def make_env(molecule: str, max_gates: int, complexity_penalty: float):
    ham = get_standard_hamiltonian(molecule)
    n_qubits = ham["n_qubits"]
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1.0  # |0...0>
    env = ArchitectEnv(
        target=target,
        fidelity_threshold=2.0,  # effectively never terminate early on fidelity
        reward_penalty=0.0,
        max_timesteps=max_gates,
        include_rotations=True,
        task_mode="vqe",
        hamiltonian_matrix=ham["matrix"],
        complexity_penalty_weight=complexity_penalty,
    )
    return env, ham


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
    ap.add_argument("--total-timesteps", type=int, default=10000)
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
