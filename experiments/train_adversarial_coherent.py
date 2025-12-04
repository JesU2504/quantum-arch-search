#!/usr/bin/env python3
"""
Adversarial co-evolution focusing on coherent over-rotation noise.

Uses new CoherentSaboteurEnv/CoherentAdversarialArchitectEnv to attack circuits
with Rx over-rotations (angles from a discrete set).
"""
import os
import sys
import time
import json
import numpy as np
import cirq
from stable_baselines3 import PPO

# Path setup
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from experiments import config
from qas_gym.utils import save_circuit, get_ideal_unitary
from qas_gym.envs import ArchitectEnv
from qas_gym.envs.coherent_saboteur_env import CoherentSaboteurEnv, CoherentAdversarialArchitectEnv
from experiments.train_adversarial import ChampionCircuitCallback, AdversarialLoggerCallback


def main(args):
    # Config summary
    print("\n=== Coherent Adversarial Training (Over-Rotation) ===")
    print(f"n_qubits={args.n_qubits}, generations={args.n_generations}")
    print(f"architect_steps/gen={args.architect_steps_per_generation}, saboteur_steps/gen={args.saboteur_steps_per_generation}")
    print(f"max_circuit_gates={args.max_circuit_gates}, budget={args.saboteur_budget}, budget_frac={args.saboteur_budget_fraction}")
    print(f"angle_set={CoherentSaboteurEnv.all_error_angles}")

    # Results dir
    os.makedirs(args.results_dir, exist_ok=True)
    log_dir = args.results_dir

    # Target
    if args.task_mode == 'state_preparation':
        target_state = config.get_target_state(args.n_qubits)
        ideal_U = None
    else:
        target_state = None
        ideal_U = get_ideal_unitary(args.n_qubits, config.TARGET_TYPE, silent=False)

    qubits = [cirq.LineQubit(i) for i in range(args.n_qubits)]
    action_gates = config.get_action_gates(qubits, include_rotations=args.include_rotations)

    arch_env_kwargs = dict(
        target=target_state,
        fidelity_threshold=args.fidelity_threshold,
        max_timesteps=args.max_circuit_gates,
        reward_penalty=0.01,
        complexity_penalty_weight=0.01,
        include_rotations=args.include_rotations,
        action_gates=action_gates,
        qubits=qubits,
        task_mode=args.task_mode,
        ideal_unitary=ideal_U,
    )

    # Saboteur env (used only for its spaces and training)
    dummy_qubits = list(cirq.LineQubit.range(args.n_qubits))
    dummy_circuit = cirq.Circuit([cirq.I(q) for q in dummy_qubits])
    sab_env = CoherentSaboteurEnv(
        architect_circuit=dummy_circuit,
        target_state=target_state,
        max_circuit_timesteps=args.max_circuit_gates,
        max_concurrent_attacks=args.saboteur_budget,
        lambda_penalty=args.lambda_penalty,
        n_qubits=args.n_qubits,
    )
    saboteur_agent = PPO('MultiInputPolicy', sab_env, **config.AGENT_PARAMS)

    # Architect base env + coherent wrapper
    base_arch_env = ArchitectEnv(**arch_env_kwargs)
    total_arch_steps = args.n_generations * args.architect_steps_per_generation
    coherent_arch_env = CoherentAdversarialArchitectEnv(
        base_arch_env=base_arch_env,
        saboteur_agent=saboteur_agent,
        total_training_steps=total_arch_steps,
        saboteur_budget=args.saboteur_budget,
        saboteur_budget_fraction=args.saboteur_budget_fraction,
        saboteur_start_budget_scale=args.saboteur_start_budget_scale,
    )
    architect_agent = PPO('MlpPolicy', coherent_arch_env, **config.AGENT_PARAMS)

    # Logs
    arch_fid, arch_complex, arch_steps = [], [], []
    sab_fid, sab_angle, sab_steps = [], [], []
    hall_of_fame = []
    current_champion_path = os.path.join(log_dir, "circuit_robust.json")

    total_arch_so_far = 0
    total_sab_so_far = 0
    for gen in range(args.n_generations):
        print(f"\n--- Generation {gen+1}/{args.n_generations} ---")
        # callbacks
        arch_champ_cb = ChampionCircuitCallback(
            circuit_filename=current_champion_path,
            clean_circuit_filename=os.path.join(log_dir, "circuit_clean_best.json"),
            offset_steps=total_arch_so_far,
            hall_of_fame=hall_of_fame,
            hall_of_fame_size=args.hall_of_fame_size,
        )
        arch_champ_cb.verify_unitary = (args.task_mode == 'unitary_preparation')
        arch_champ_cb.n_qubits = args.n_qubits
        arch_log_cb = AdversarialLoggerCallback(
            lists_dict={'fidelity': arch_fid, 'complexity': arch_complex},
            step_list=arch_steps,
            offset_steps=total_arch_so_far
        )
        architect_agent.set_env(coherent_arch_env)
        architect_agent.learn(total_timesteps=args.architect_steps_per_generation,
                              callback=[arch_champ_cb, arch_log_cb])
        total_arch_so_far += args.architect_steps_per_generation

        # Update saboteur circuit
        if os.path.exists(current_champion_path):
            with open(current_champion_path, 'r') as f:
                data = json.load(f)
            champion_circuit = cirq.read_json(json_text=json.dumps(data))
            sab_env.set_circuit(champion_circuit)

        # Train saboteur
        sab_log_cb = AdversarialLoggerCallback(
            lists_dict={'fidelity': sab_fid, 'error_rate': sab_angle},
            step_list=sab_steps,
            offset_steps=total_sab_so_far
        )
        saboteur_agent.learn(total_timesteps=args.saboteur_steps_per_generation,
                             callback=[sab_log_cb])
        total_sab_so_far += args.saboteur_steps_per_generation

    # Save logs
    np.savetxt(os.path.join(log_dir, "architect_fidelities.txt"), np.array(arch_fid))
    np.savetxt(os.path.join(log_dir, "architect_complexity.txt"), np.array(arch_complex))
    np.savetxt(os.path.join(log_dir, "architect_steps.txt"), np.array(arch_steps))
    np.savetxt(os.path.join(log_dir, "saboteur_trained_on_architect_fidelities.txt"), np.array(sab_fid))
    np.savetxt(os.path.join(log_dir, "saboteur_trained_on_architect_steps.txt"), np.array(sab_steps))

    with open(os.path.join(log_dir, "hall_of_fame.json"), "w") as f:
        json.dump(hall_of_fame, f, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Adversarial training with coherent over-rotation saboteur.")
    ap.add_argument("--n-qubits", type=int, default=3)
    ap.add_argument("--n-generations", type=int, default=5)
    ap.add_argument("--architect-steps-per-generation", type=int, default=2000)
    ap.add_argument("--saboteur-steps-per-generation", type=int, default=1500)
    ap.add_argument("--max-circuit-gates", type=int, default=20)
    ap.add_argument("--fidelity-threshold", type=float, default=1.1)
    ap.add_argument("--include-rotations", action="store_true")
    ap.add_argument("--task-mode", type=str, default="state_preparation", choices=["state_preparation", "unitary_preparation"])
    ap.add_argument("--saboteur-budget", type=int, default=3)
    ap.add_argument("--saboteur-budget-fraction", type=float, default=0.2)
    ap.add_argument("--saboteur-start-budget-scale", type=float, default=0.4)
    ap.add_argument("--lambda-penalty", type=float, default=0.5)
    ap.add_argument("--results-dir", type=str, default="results/adversarial_coherent")
    ap.add_argument("--hall-of-fame-size", type=int, default=5)
    args = ap.parse_args()

    main(args)
