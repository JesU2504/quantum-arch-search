#!/usr/bin/env python3
"""
Adversarial VQE with learned architectures (ArchitectEnv + Saboteur).

This mirrors the state-prep coevolution setup but uses ArchitectEnv with
task_mode="vqe" so rewards come from energy (lower is better). The saboteur
learns to inject noise that maximizes the energy; the architect is trained
against the saboteur using a mixed clean/attacked energy reward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import cirq
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments import config  # noqa: E402
from qas_gym.envs import SaboteurMultiGateEnv, ArchitectEnv  # noqa: E402
from qas_gym.utils import save_circuit  # noqa: E402
from utils.metrics import state_energy  # noqa: E402
from utils.standard_hamiltonians import get_standard_hamiltonian  # noqa: E402
from utils.torchquantum_adapter import convert_qasm_file_to_cirq  # noqa: E402


_MOLECULE_QUBITS = {
    "H2": 2,
    "HeH+": 2,
    "LiH": 4,
    "BeH2": 6,
}


def _canonical_noise_families(noise_families: Optional[Sequence[str]], fallback: str) -> List[str]:
    """Return a de-duplicated list of noise families with at least one entry."""
    if noise_families is None:
        return [fallback]
    families = [str(f).strip() for f in noise_families if str(f).strip()]
    if not families:
        families = [fallback]
    # Preserve order but drop duplicates
    seen = set()
    ordered = []
    for fam in families:
        if fam not in seen:
            seen.add(fam)
            ordered.append(fam)
    return ordered


def _zz_chain_hamiltonian(n_qubits: int) -> np.ndarray:
    """Simple ZZ chain Hamiltonian used as a fallback when qiskit-nature is unavailable."""
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


# ---------------- Saboteur for VQE (energy-increase objective) ----------------
class SaboteurVQEnv(gym.Env):
    """
    Saboteur environment that rewards increasing the circuit energy.

    Observation/action spaces mirror SaboteurMultiGateEnv to reuse the same
    PPO configuration.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        architect_circuit: cirq.Circuit,
        hamiltonian_matrix: np.ndarray,
        max_circuit_timesteps: int,
        error_rates: List[float],
        noise_families: Sequence[str],
        noise_kwargs: Optional[dict],
        lambda_penalty: float = 0.5,
        n_qubits: int = 3,
    ):
        super().__init__()
        self.hamiltonian_matrix = hamiltonian_matrix
        self.lambda_penalty = lambda_penalty
        self.n_qubits = n_qubits
        self.noise_families = list(noise_families) if noise_families else ["depolarizing"]
        self.noise_kwargs = noise_kwargs.copy() if noise_kwargs is not None else {}
        self.error_rates = list(error_rates)
        self.max_circuit_timesteps = max_circuit_timesteps

        # Backing env for observation/action shapes
        dummy_target = np.zeros(2**n_qubits, dtype=complex)
        self._base_env = SaboteurMultiGateEnv(
            architect_circuit=architect_circuit,
            target_state=dummy_target,
            max_circuit_timesteps=max_circuit_timesteps,
            n_qubits=n_qubits,
            error_rates=self.error_rates,
            noise_family=self.noise_families[0],
            noise_kwargs=self.noise_kwargs,
            max_concurrent_attacks=max_circuit_timesteps,
            lambda_penalty=lambda_penalty,
        )
        self.action_space = self._base_env.action_space
        self.observation_space = self._base_env.observation_space
        self.sim = cirq.Simulator()

        self.current_circuit = architect_circuit
        self.clean_energy = self._energy(self.current_circuit)

    def _sample_noise_family(self) -> str:
        if len(self.noise_families) == 1:
            return self.noise_families[0]
        return str(np.random.choice(self.noise_families))

    def _energy(self, circuit: cirq.Circuit) -> float:
        res = self.sim.simulate(circuit, qubit_order=cirq.LineQubit.range(self.n_qubits))
        return state_energy(res.final_state_vector, self.hamiltonian_matrix)

    def set_circuit(self, circuit: cirq.Circuit):
        self.current_circuit = circuit
        self._base_env.set_circuit(circuit)
        self.clean_energy = self._energy(circuit)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._base_env.set_circuit(self.current_circuit)
        return self._base_env.reset(seed=seed, options=options)

    def step(self, action):
        noise_family = self._sample_noise_family()
        noisy_circuit, applied_rates, _ = SaboteurMultiGateEnv.build_noisy_circuit(
            circuit=self.current_circuit,
            action=action,
            error_rates=self.error_rates,
            noise_family=noise_family,
            max_concurrent_attacks=self.max_circuit_timesteps,
            max_gates=self.max_circuit_timesteps,
            noise_kwargs=self.noise_kwargs,
        )
        noisy_energy = self._energy(noisy_circuit)
        mean_err = float(np.mean(applied_rates)) if applied_rates else 0.0
        reward = (noisy_energy - self.clean_energy) - self.lambda_penalty * mean_err
        obs = self._base_env._get_obs(self.current_circuit)
        info = {
            "energy_noisy": noisy_energy,
            "mean_error_rate": mean_err,
            "noise_family": noise_family,
        }
        return obs, reward, True, False, info


# ---------------- Architect side: mixed clean/attacked energy reward ---------
class AdversarialVQEEnv(ArchitectEnv):
    """
    ArchitectEnv with task_mode='vqe' that mixes clean and attacked energy.
    """

    def __init__(
        self,
        saboteur_agent: PPO | None,
        saboteur_error_rates: List[float],
        saboteur_noise_families: Sequence[str],
        saboteur_noise_kwargs: Optional[dict],
        alpha_start: float,
        alpha_end: float,
        total_training_steps: int,
        saboteur_budget: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.saboteur_agent = saboteur_agent
        self.saboteur_error_rates = saboteur_error_rates
        self.saboteur_noise_families = list(saboteur_noise_families) if saboteur_noise_families else ["depolarizing"]
        self.saboteur_noise_kwargs = saboteur_noise_kwargs.copy() if saboteur_noise_kwargs is not None else {}
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_training_steps = float(total_training_steps)
        self.global_step = 0
        self.saboteur_budget = saboteur_budget

    def _sample_noise_family(self) -> str:
        if len(self.saboteur_noise_families) == 1:
            return self.saboteur_noise_families[0]
        return str(np.random.choice(self.saboteur_noise_families))

    def step(self, action):
        obs, clean_reward, terminated, truncated, info = super().step(action)
        clean_energy = info.get("energy", None)
        gate_count = info.get("total_gates", info.get("gate_count", 0))
        self.global_step += 1
        reward = clean_reward
        attacked_energy = clean_energy

        if terminated and self.saboteur_agent is not None and clean_energy is not None:
            final_circuit = self._get_cirq()
            if final_circuit is not None:
                noise_family = self._sample_noise_family()
                sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(
                    final_circuit,
                    n_qubits=len(self.qubits),
                    max_circuit_timesteps=self.max_timesteps,
                    error_rates=self.saboteur_error_rates,
                    noise_family=noise_family,
                    noise_kwargs=self.saboteur_noise_kwargs,
                )
                sab_action, _ = self.saboteur_agent.predict(sab_obs, deterministic=True)
                noisy_circuit, applied_rates, _ = SaboteurMultiGateEnv.build_noisy_circuit(
                    circuit=final_circuit,
                    action=sab_action,
                    error_rates=self.saboteur_error_rates,
                    noise_family=noise_family,
                    max_concurrent_attacks=min(self.saboteur_budget, self.max_timesteps),
                    max_gates=self.max_timesteps,
                    noise_kwargs=self.saboteur_noise_kwargs,
                )
                sim = cirq.Simulator()
                result = sim.simulate(noisy_circuit, qubit_order=self.qubits)
                attacked_energy = state_energy(result.final_state_vector, self.hamiltonian_matrix)
                info["mean_error_rate"] = float(np.mean(applied_rates)) if applied_rates else 0.0
                info["noise_family"] = noise_family

                # Anneal alpha from start -> end over training
                t = min(1.0, self.global_step / self.total_training_steps) if self.total_training_steps > 0 else 1.0
                alpha = self.alpha_start + t * (self.alpha_end - self.alpha_start)
                mixed_energy = alpha * clean_energy + (1.0 - alpha) * attacked_energy
                reward = -mixed_energy - self.complexity_penalty_weight * gate_count

            info["energy_under_attack"] = attacked_energy
            info["circuit"] = final_circuit

        return obs, reward, terminated, truncated, info


# -------------------------- Callbacks & helpers ------------------------------
class EnergyChampionCallback(BaseCallback):
    """Track lowest attacked energy and save circuits."""

    def __init__(
        self,
        robust_path: Path,
        clean_path: Path,
        offset_steps: int = 0,
        min_step_to_save: int = 0,
        hall_of_fame: Optional[list] = None,
        hall_of_fame_size: int = 5,
    ):
        super().__init__(verbose=0)
        self.robust_path = robust_path
        self.clean_path = clean_path
        self.best_attacked = None
        self.best_clean = None
        self.offset_steps = offset_steps
        self.min_step_to_save = min_step_to_save
        self.hall_of_fame = hall_of_fame if hall_of_fame is not None else []
        self.hall_of_fame_size = hall_of_fame_size

    def _record_hof(self, entry: dict):
        self.hall_of_fame.append(entry)
        self.hall_of_fame.sort(key=lambda e: e.get("metric", float("inf")))
        if len(self.hall_of_fame) > self.hall_of_fame_size:
            self.hall_of_fame[:] = self.hall_of_fame[: self.hall_of_fame_size]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        global_step = self.num_timesteps + self.offset_steps
        for info in infos:
            circ = info.get("circuit")
            if circ is None or global_step < self.min_step_to_save:
                continue
            attacked = info.get("energy_under_attack", info.get("energy", None))
            clean = info.get("energy", None)
            if attacked is None:
                continue
            # Hall-of-fame (lower is better)
            self._record_hof(
                {
                    "metric": float(attacked),
                    "clean_energy": float(clean) if clean is not None else None,
                    "step": int(global_step),
                    "circuit": json.loads(cirq.to_json(circ)),
                }
            )
            if self.best_attacked is None or attacked < self.best_attacked:
                self.best_attacked = attacked
                save_circuit(str(self.robust_path), circ)
            if clean is not None and (self.best_clean is None or clean < self.best_clean):
                self.best_clean = clean
                save_circuit(str(self.clean_path), circ)
        return True


class EnergyLoggerCallback(BaseCallback):
    """Collect clean/attacked energies for plotting."""

    def __init__(self, lists_dict: dict, step_list: list, offset_steps: int = 0):
        super().__init__(verbose=0)
        self.lists_dict = lists_dict
        self.step_list = step_list
        self.offset_steps = offset_steps

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            step = self.num_timesteps + self.offset_steps
            self.step_list.append(step)
            if "energy" in self.lists_dict:
                self.lists_dict["energy"].append(info.get("energy", 0.0))
            if "energy_under_attack" in self.lists_dict:
                self.lists_dict["energy_under_attack"].append(info.get("energy_under_attack", info.get("energy", 0.0)))
            if "complexity" in self.lists_dict:
                self.lists_dict["complexity"].append(info.get("total_gates", info.get("gate_count", 0)))
        return True


def export_circuit_json(circ: cirq.Circuit, path: Path):
    path.write_text(cirq.to_json(circ))
    try:
        qasm_path = path.with_suffix(".qasm")
        qasm_path.write_text(cirq.qasm(circ))
        convert_qasm_file_to_cirq(qasm_path, path.with_suffix(".cirq.json"))
    except Exception:
        pass


def evaluate_attacked_energy(
    circuit: cirq.Circuit,
    saboteur_agent: PPO,
    saboteur_error_rates: List[float],
    saboteur_noise_families: Sequence[str],
    saboteur_noise_kwargs: dict,
    max_timesteps: int,
    hamiltonian_matrix: np.ndarray,
    qubits: List[cirq.LineQubit],
) -> float:
    """Deterministic saboteur attack energy; return the worst across noise families."""
    worst_energy = None
    families = saboteur_noise_families if saboteur_noise_families else ["depolarizing"]
    for noise_family in families:
        sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(
            circuit,
            n_qubits=len(qubits),
            max_circuit_timesteps=max_timesteps,
            error_rates=saboteur_error_rates,
            noise_family=noise_family,
            noise_kwargs=saboteur_noise_kwargs,
        )
        sab_action, _ = saboteur_agent.predict(sab_obs, deterministic=True)
        noisy_circuit, _, _ = SaboteurMultiGateEnv.build_noisy_circuit(
            circuit=circuit,
            action=sab_action,
            error_rates=saboteur_error_rates,
            noise_family=noise_family,
            max_concurrent_attacks=max_timesteps,
            max_gates=max_timesteps,
            noise_kwargs=saboteur_noise_kwargs,
        )
        sim = cirq.Simulator()
        result = sim.simulate(noisy_circuit, qubit_order=qubits)
        energy = state_energy(result.final_state_vector, hamiltonian_matrix)
        if worst_energy is None or energy > worst_energy:
            worst_energy = energy
    return float(worst_energy if worst_energy is not None else 0.0)


# ------------------------------ Training loop --------------------------------
def train_adversarial_vqe_architect(
    molecule: str,
    n_generations: int,
    architect_steps_per_gen: int,
    saboteur_steps_per_gen: int,
    max_circuit_gates: int,
    complexity_penalty: float,
    alpha_start: float,
    alpha_end: float,
    saboteur_budget: int,
    saboteur_noise_families: Sequence[str],
    saboteur_error_rates: List[float],
    saboteur_noise_kwargs: dict,
    seed: int,
    out_dir: Path,
):
    try:
        ham_info = get_standard_hamiltonian(molecule)
        n_qubits = ham_info["n_qubits"]
        hamiltonian_matrix = ham_info["matrix"]
        hf_energy = ham_info.get("hf_energy")
        fci_energy = ham_info.get("fci_energy")
    except Exception:
        # Fallback to a simple ZZ-chain Hamiltonian when qiskit-nature stack is unavailable
        n_qubits = _MOLECULE_QUBITS.get(molecule, 3)
        hamiltonian_matrix = _zz_chain_hamiltonian(n_qubits)
        eigs = np.linalg.eigvalsh(hamiltonian_matrix)
        hf_energy = float(np.min(eigs))
        fci_energy = float(np.min(eigs))
        ham_info = {
            "n_qubits": n_qubits,
            "matrix": hamiltonian_matrix,
            "hf_energy": hf_energy,
            "fci_energy": fci_energy,
        }

    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    action_gates = config.get_action_gates(qubits, include_rotations=True)

    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Saboteur env/agent
    dummy_circuit = cirq.Circuit()
    noise_families = list(saboteur_noise_families) if saboteur_noise_families else ["depolarizing"]

    sab_env = SaboteurVQEnv(
        architect_circuit=dummy_circuit,
        hamiltonian_matrix=hamiltonian_matrix,
        max_circuit_timesteps=max_circuit_gates,
        error_rates=saboteur_error_rates,
        noise_families=noise_families,
        noise_kwargs=saboteur_noise_kwargs,
        n_qubits=n_qubits,
    )
    sab_agent = PPO("MultiInputPolicy", sab_env, **config.AGENT_PARAMS)
    total_steps = n_generations * architect_steps_per_gen

    # Architect agent (adversarial env)
    arch_env_kwargs = dict(
        target=np.zeros(2**n_qubits, dtype=complex),
        fidelity_threshold=2.0,
        reward_penalty=0.0,
        max_timesteps=max_circuit_gates,
        include_rotations=True,
        action_gates=action_gates,
        qubits=qubits,
        task_mode="vqe",
        hamiltonian_matrix=hamiltonian_matrix,
        complexity_penalty_weight=complexity_penalty,
    )

    arch_env = AdversarialVQEEnv(
        saboteur_agent=sab_agent,
        saboteur_error_rates=saboteur_error_rates,
        saboteur_noise_families=noise_families,
        saboteur_noise_kwargs=saboteur_noise_kwargs,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        total_training_steps=total_steps,
        saboteur_budget=saboteur_budget,
        **arch_env_kwargs,
    )
    arch_agent = PPO("MlpPolicy", arch_env, **config.AGENT_PARAMS)

    # Logging
    hall_of_fame: list = []
    energies, attacked_energies, complexities, steps = [], [], [], []
    total_arch_steps = 0
    total_sab_steps = 0

    robust_path = out_dir / "circuit_robust.json"
    clean_path = out_dir / "circuit_clean_best.json"
    for gen in range(n_generations):
        print(f"[Adversarial VQE] Generation {gen+1}/{n_generations}")
        env_gen = AdversarialVQEEnv(
            saboteur_agent=sab_agent,
            saboteur_error_rates=saboteur_error_rates,
            saboteur_noise_families=noise_families,
            saboteur_noise_kwargs=saboteur_noise_kwargs,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            total_training_steps=total_steps,
            saboteur_budget=saboteur_budget,
            **arch_env_kwargs,
        )
        arch_agent.set_env(env_gen)

        champ_cb = EnergyChampionCallback(
            robust_path=robust_path,
            clean_path=clean_path,
            offset_steps=total_arch_steps,
            hall_of_fame=hall_of_fame,
        )
        log_cb = EnergyLoggerCallback(
            lists_dict={
                "energy": energies,
                "energy_under_attack": attacked_energies,
                "complexity": complexities,
            },
            step_list=steps,
            offset_steps=total_arch_steps,
        )
        arch_agent.learn(total_timesteps=architect_steps_per_gen, callback=[champ_cb, log_cb])
        total_arch_steps += architect_steps_per_gen

        # Update saboteur circuit
        if robust_path.exists():
            circ = cirq.read_json(json_text=robust_path.read_text())
            sab_env.set_circuit(circ)

        # Train saboteur
        sab_agent.learn(total_timesteps=saboteur_steps_per_gen)
        total_sab_steps += saboteur_steps_per_gen

    # Save logs
    np.savetxt(out_dir / "architect_energy.txt", np.array(energies))
    np.savetxt(out_dir / "architect_energy_under_attack.txt", np.array(attacked_energies))
    np.savetxt(out_dir / "architect_complexity.txt", np.array(complexities))
    np.savetxt(out_dir / "architect_steps.txt", np.array(steps))
    (out_dir / "hall_of_fame.json").write_text(json.dumps(hall_of_fame, indent=2))

    # Pick best circuit (lowest attacked energy)
    best_entry = None
    for entry in hall_of_fame:
        if best_entry is None or entry["metric"] < best_entry["metric"]:
            best_entry = entry

    if best_entry is not None:
        best_circ = cirq.read_json(json_text=json.dumps(best_entry["circuit"]))
        export_circuit_json(best_circ, out_dir / "circuit_best_attacked.json")
        attacked_energy = evaluate_attacked_energy(
            best_circ,
            sab_agent,
            saboteur_error_rates,
            noise_families,
            saboteur_noise_kwargs,
            max_circuit_gates,
            hamiltonian_matrix,
            qubits,
        )
        clean_sim = cirq.Simulator().simulate(best_circ, qubit_order=qubits)
        clean_energy = state_energy(clean_sim.final_state_vector, hamiltonian_matrix)
        gate_count = len(list(best_circ.all_operations()))
    else:
        attacked_energy = None
        clean_energy = None
        gate_count = None

    results = {
        "molecule": molecule,
        "n_qubits": n_qubits,
        "seed": seed,
        "best_attacked_energy": attacked_energy,
        "best_clean_energy": clean_energy,
        "best_gate_count": gate_count,
        "hf_energy": ham_info.get("hf_energy"),
        "fci_energy": ham_info.get("fci_energy"),
        "hall_of_fame_size": len(hall_of_fame),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[Adversarial VQE] Saved results to {out_dir}")


# ---------------------------------- CLI -------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial VQE with ArchitectEnv + saboteur.")
    p.add_argument("--molecule", default="H2", choices=["H2", "HeH+", "LiH", "BeH2"])
    p.add_argument("--n-generations", type=int, default=5)
    p.add_argument("--architect-steps-per-gen", type=int, default=4000)
    p.add_argument("--saboteur-steps-per-gen", type=int, default=2000)
    p.add_argument("--max-gates", type=int, default=config.MAX_CIRCUIT_TIMESTEPS)
    p.add_argument("--complexity-penalty", type=float, default=0.01)
    p.add_argument("--alpha-start", type=float, default=0.5)
    p.add_argument("--alpha-end", type=float, default=0.0)
    p.add_argument("--saboteur-budget", type=int, default=3)
    p.add_argument("--saboteur-noise-family", type=str, default="depolarizing")
    p.add_argument("--saboteur-noise-families", type=str, nargs="+", default=None)
    p.add_argument("--saboteur-error-rates", type=float, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    error_rates = args.saboteur_error_rates
    if error_rates is None or len(error_rates) == 0:
        error_rates = list(SaboteurMultiGateEnv.all_error_rates)
    noise_families = _canonical_noise_families(args.saboteur_noise_families, args.saboteur_noise_family)
    train_adversarial_vqe_architect(
        molecule=args.molecule,
        n_generations=args.n_generations,
        architect_steps_per_gen=args.architect_steps_per_gen,
        saboteur_steps_per_gen=args.saboteur_steps_per_gen,
        max_circuit_gates=args.max_gates,
        complexity_penalty=args.complexity_penalty,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        saboteur_budget=args.saboteur_budget,
        saboteur_noise_families=noise_families,
        saboteur_error_rates=error_rates,
        saboteur_noise_kwargs={},
        seed=args.seed,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
