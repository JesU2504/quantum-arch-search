"""
Adversarial Co-evolutionary Training Script (Enhanced Logging).

Updates:
- Logs 'Architect Complexity' (Gate Count) to track if circuits are getting deeper.
- Logs 'Saboteur Error Rate' to track attack intensity.
- Uses persistent Saboteur (no reset) for Red Queen dynamics.
- Uses config.py defaults for circuit size.
"""

import os
import sys

# Add repository root to sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import time
import json
import numpy as np
import cirq
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from experiments import config
from qas_gym.envs import SaboteurMultiGateEnv, AdversarialArchitectEnv
from qas_gym.utils import save_circuit, verify_toffoli_unitary, get_ideal_unitary, fidelity_pure_target


# --- 1. Robust Champion Callback ---
class ChampionCircuitCallback(BaseCallback):
    """
    A callback to save the best circuit found during training.
    Checks ALL parallel environments and prints in real-time.
    """
    def __init__(
        self,
        circuit_filename,
        clean_circuit_filename=None,
        verbose=0,
        offset_steps=0,
        min_step_to_save=0,
        hall_of_fame=None,
        hall_of_fame_size=5,
    ):
        super().__init__(verbose)
        self.circuit_filename = circuit_filename
        # Optional: track a separate best-by-clean-fidelity circuit
        self.clean_circuit_filename = clean_circuit_filename
        self.best_saved_fidelity = None
        self.best_clean_saved_fidelity = None
        self.unitary_mode = False
        self.verify_unitary = False
        self.n_qubits = None
        # Offset allows us to enforce "start saving only after X global steps"
        self.offset_steps = offset_steps
        self.min_step_to_save = min_step_to_save
        self.hall_of_fame = hall_of_fame if hall_of_fame is not None else []
        self.hall_of_fame_size = hall_of_fame_size

    def _record_hall_of_fame(self, entry):
        """Maintain a top-k hall of fame by attacked/process fidelity."""
        self.hall_of_fame.append(entry)
        # Sort descending by metric
        self.hall_of_fame.sort(key=lambda e: e.get("metric", -1.0), reverse=True)
        # Trim to size
        if len(self.hall_of_fame) > self.hall_of_fame_size:
            self.hall_of_fame[:] = self.hall_of_fame[: self.hall_of_fame_size]

    def _on_step(self) -> bool:
        # We assume training_env is a VecEnv with possibly multiple ArchitectEnv instances.
        envs = self.training_env.envs if hasattr(self.training_env, "envs") else [self.training_env]
        updated = False
        global_step = self.num_timesteps + self.offset_steps

        for i, env in enumerate(envs):
            if hasattr(env, 'env'):
                env = env.env  # unwrap Monitor or other wrappers

            # Try to get the last info dict
            infos = self.locals.get('infos', [])
            if len(infos) <= i:
                continue

            info = infos[i]
            if 'circuit' not in info and not hasattr(env, 'get_circuit'):
                continue

            # Extract candidate circuit and fidelity
            try:
                if not self.unitary_mode and self.verify_unitary:
                    self.unitary_mode = True

                candidate_circuit = info.get('circuit')
                candidate_fid = info.get('fidelity', 0.0)
                clean_metric = candidate_fid

                if candidate_circuit is None:
                    try:
                        candidate_circuit = self.training_env.env_method('get_circuit', indices=[i])[0]
                    except Exception:
                        continue
                # Enforce minimum step before considering champions
                if global_step < self.min_step_to_save:
                    continue

                if candidate_circuit:
                    # Prefer robustness (fidelity_under_attack) if available
                    robust_fid = info.get('fidelity_under_attack', None)
                    candidate_fid = info.get('fidelity', 0.0)

                    if robust_fid is not None:
                        metric_to_check = robust_fid
                        metric_name = "Fidelity Under Attack"
                    else:
                        metric_to_check = candidate_fid
                        metric_name = "Fidelity"

                    # For unitary tasks, optionally override with process fidelity
                    if self.verify_unitary and self.n_qubits is not None:
                        try:
                            _, process_fid = verify_toffoli_unitary(
                                candidate_circuit, self.n_qubits, silent=True
                            )
                            metric_to_check = process_fid
                            metric_name = "Process Fidelity"
                        except Exception:
                            # Fall back to robustness / state fidelity
                            pass

                    # Hall-of-fame tracking for best-overall evaluation
                    if candidate_circuit is not None:
                        hof_entry = {
                            "metric": float(metric_to_check),
                            "metric_name": metric_name,
                            "clean_metric": float(clean_metric) if clean_metric is not None else None,
                            "step": int(global_step),
                            "circuit": json.loads(cirq.to_json(candidate_circuit)),
                        }
                        self._record_hall_of_fame(hof_entry)

                    # Compare and potentially save
                    if (self.best_saved_fidelity is None) or (metric_to_check > self.best_saved_fidelity):
                        self.best_saved_fidelity = metric_to_check

                        # NB: save_circuit expects (path, circuit)
                        save_circuit(self.circuit_filename, candidate_circuit)
                        if self.verbose > 0:
                            print(
                                f"\n[ChampionCircuitCallback] New champion! {metric_name}={metric_to_check:.6f} "
                                f"-> saved to {self.circuit_filename}"
                            )
                        updated = True

                    # Track best-by-clean-fidelity separately if requested
                    if self.clean_circuit_filename is not None:
                        clean_metric = candidate_fid
                        if (self.best_clean_saved_fidelity is None) or (clean_metric > self.best_clean_saved_fidelity):
                            self.best_clean_saved_fidelity = clean_metric
                            save_circuit(self.clean_circuit_filename, candidate_circuit)
                            if self.verbose > 0:
                                print(
                                    f"[ChampionCircuitCallback] New clean-best! Fidelity={clean_metric:.6f} "
                                    f"-> saved to {self.clean_circuit_filename}"
                                )

            except Exception as e:
                if self.verbose > 0:
                    print(f"[ChampionCircuitCallback] Error while processing environment {i}: {e}")

        return True


# --- 2. Adversarial Logger Callback ---
class AdversarialLoggerCallback(BaseCallback):
    """
    Simple callback that logs fidelity, complexity, and error rate arrays in-place.
    """
    def __init__(self, lists_dict, step_list, offset_steps=0, verbose=0):
        """
        lists_dict: dict like {
            'fidelity': list_ref,
            'complexity': list_ref,
            'error_rate': list_ref,
        }
        step_list: list_ref for steps
        """
        super().__init__(verbose)
        self.lists_dict = lists_dict
        self.step_list = step_list
        self.offset_steps = offset_steps

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        # sb3's rollouts give us a batch of infos per step
        for info in infos:
            if not isinstance(info, dict):
                continue

            # Step index (global-like)
            timestep = self.num_timesteps + self.offset_steps
            self.step_list.append(timestep)

            # Log Fidelity
            if 'fidelity' in self.lists_dict:
                self.lists_dict['fidelity'].append(info.get('fidelity', 0.0))

            # Log Complexity (Architect specific)
            if 'complexity' in self.lists_dict:
                gate_metric = info.get('total_gates', info.get('gate_count', 0))
                self.lists_dict['complexity'].append(gate_metric)

            # Log Error Rate (Saboteur specific)
            if 'error_rate' in self.lists_dict:
                self.lists_dict['error_rate'].append(info.get('mean_error_rate', 0.0))

        return True


def train_adversarial(
    results_dir: str,
    n_qubits: int,
    n_generations: int,
    architect_steps_per_generation: int,
    saboteur_steps_per_generation: int,
    max_circuit_gates: int = config.MAX_CIRCUIT_TIMESTEPS, # UPDATED: Use Config
    fidelity_threshold: float = 0.99,
    lambda_penalty: float = 0.5,
    include_rotations: bool | None = None,
    task_mode: str = None,
    champion_save_last_steps: int | None = None,
    hall_of_fame_size: int = 5,
    saboteur_noise_family: str = "depolarizing",
    saboteur_error_rates: list[float] | None = None,
    saboteur_noise_kwargs: dict | None = None,
    saboteur_budget: int | None = 3,
    saboteur_budget_fraction: float | None = 0.2,
    saboteur_start_budget_scale: float = 0.3,
    alpha_start: float = 0.6,
    alpha_end: float = 0.0,
):
    # --- Configuration ---
    effective_task_mode = task_mode if task_mode is not None else config.TASK_MODE
    if include_rotations is None:
        include_rotations = config.INCLUDE_ROTATIONS
    effective_target_type = config.TARGET_TYPE
    rotation_status = "with rotation gates" if include_rotations else "with Clifford+T gates only"
    
    print("\n" + "=" * 60)
    print("ADVERSARIAL TRAINING CONFIGURATION (ENHANCED LOGGING)")
    print("=" * 60)
    print(f"Target Type: {effective_target_type}")
    print(f"Task Mode: {effective_task_mode}")
    print(f"Number of Qubits: {n_qubits}")
    print(f"Max Circuit Gates: {max_circuit_gates}")
    print(f"Generations: {n_generations}")
    print(f"Architect Steps per Generation: {architect_steps_per_generation}")
    print(f"Saboteur Steps per Generation: {saboteur_steps_per_generation}")
    print(f"Lambda Penalty (Saboteur): {lambda_penalty}")
    print(f"Rotation Gates: {rotation_status}")
    print(f"Hall-of-Fame Size: {hall_of_fame_size}")
    print("=" * 60 + "\n")

    total_architect_steps = n_generations * architect_steps_per_generation
    champion_min_step = 0
    if champion_save_last_steps is not None and champion_save_last_steps > 0:
        champion_min_step = max(0, total_architect_steps - champion_save_last_steps)
        print(
            f"Champion circuits will only be saved from step {champion_min_step} onward "
            f"(last {champion_save_last_steps} architect steps)."
        )

    # --- Prepare Results Directory ---
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = results_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # --- Target State / Unitary ---
    if effective_task_mode == 'state_preparation':
        target_state = config.get_target_state(n_qubits)
        ideal_U = None
    else:
        target_state = None
        ideal_U = get_ideal_unitary(n_qubits, config.TARGET_TYPE, silent=False)

    saboteur_target_state = target_state if target_state is not None else config.get_target_state(n_qubits)
    sab_error_rates = list(saboteur_error_rates) if saboteur_error_rates is not None else list(SaboteurMultiGateEnv.all_error_rates)
    if len(sab_error_rates) == 0:
        sab_error_rates = list(SaboteurMultiGateEnv.all_error_rates)

    # --- Initialize Environments ---
    max_saboteur_level = len(sab_error_rates)
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    action_gates = config.get_action_gates(qubits, include_rotations=include_rotations)
    
    arch_env_kwargs = dict(
        target=target_state,
        fidelity_threshold=fidelity_threshold,
        max_timesteps=max_circuit_gates,
        reward_penalty=0.01,
        complexity_penalty_weight=0.01,
        include_rotations=include_rotations,
        action_gates=action_gates,
        qubits=qubits,
        task_mode=effective_task_mode,
        ideal_unitary=ideal_U,
    )

    dummy_qubits = list(cirq.LineQubit.range(int(np.log2(len(target_state))))) if target_state is not None else qubits
    dummy_circuit = cirq.Circuit([cirq.I(q) for q in dummy_qubits])
    
    # NOTE: Saboteur environment also needs to know the max_circuit_gates
    # to set its observation space size correctly.
    saboteur_env = SaboteurMultiGateEnv(
        architect_circuit=dummy_circuit,
        target_state=saboteur_target_state,
        max_circuit_timesteps=max_circuit_gates, # Ensure this matches architect
        max_concurrent_attacks=saboteur_budget if saboteur_budget is not None else max_circuit_gates,
        discrete=True,
        lambda_penalty=lambda_penalty,
        error_rates=sab_error_rates,
        noise_family=saboteur_noise_family,
        noise_kwargs=saboteur_noise_kwargs,
        n_qubits=n_qubits,
    )

    # --- Initialize Agents (Persistent Saboteur) ---
    saboteur_agent = PPO('MultiInputPolicy', saboteur_env, **config.AGENT_PARAMS)
    total_steps = n_generations * saboteur_steps_per_generation
    
    initial_arch_env = AdversarialArchitectEnv(
        saboteur_agent=saboteur_agent,
        saboteur_max_error_level=max_saboteur_level,
        total_training_steps=total_steps,
        saboteur_budget=saboteur_budget,
        saboteur_budget_fraction=saboteur_budget_fraction,
        saboteur_start_budget_scale=saboteur_start_budget_scale,
        saboteur_error_rates=sab_error_rates,
        saboteur_noise_family=saboteur_noise_family,
        saboteur_noise_kwargs=saboteur_noise_kwargs,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        **arch_env_kwargs
    )
    architect_agent = PPO('MlpPolicy', initial_arch_env, **config.AGENT_PARAMS)

    # --- Data Storage ---
    architect_fidelities = []
    architect_complexity = [] # NEW: Track gate count
    architect_steps = []
    
    saboteur_fidelities = []
    saboteur_error_rates = [] # NEW: Track attack intensity
    saboteur_steps = []
    
    champion_circuit = None
    total_arch_steps = 0
    total_sab_steps = 0
    hall_of_fame = []

    # --- Main Loop ---
    for gen in range(n_generations):
        print(f"\n--- Generation {gen+1}/{n_generations} ---")

        arch_env = AdversarialArchitectEnv(
            saboteur_agent=saboteur_agent,
            saboteur_max_error_level=max_saboteur_level,
            total_training_steps=total_steps,
            saboteur_budget=saboteur_budget,
            saboteur_budget_fraction=saboteur_budget_fraction,
            saboteur_start_budget_scale=saboteur_start_budget_scale,
            saboteur_error_rates=sab_error_rates,
            saboteur_noise_family=saboteur_noise_family,
            saboteur_noise_kwargs=saboteur_noise_kwargs,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            **arch_env_kwargs
        )
        architect_agent.set_env(arch_env)
        
        # Real-time champion detection
        arch_champ_callback = ChampionCircuitCallback(
            circuit_filename=os.path.join(log_dir, "circuit_robust.json"),
            clean_circuit_filename=os.path.join(log_dir, "circuit_clean_best.json"),
            offset_steps=total_arch_steps,
            min_step_to_save=champion_min_step,
            hall_of_fame=hall_of_fame,
            hall_of_fame_size=hall_of_fame_size,
        )
        arch_champ_callback.verify_unitary = (effective_task_mode == 'unitary_preparation')
        arch_champ_callback.n_qubits = n_qubits
        
        # Data logging (Fidelity + Complexity)
        arch_log_callback = AdversarialLoggerCallback(
            lists_dict={
                'fidelity': architect_fidelities,
                'complexity': architect_complexity
            },
            step_list=architect_steps,
            offset_steps=total_arch_steps
        )

        # Train Architect
        architect_agent.learn(
            total_timesteps=architect_steps_per_generation,
            callback=[arch_champ_callback, arch_log_callback]
        )
        total_arch_steps += architect_steps_per_generation

        # --- Update Saboteur Circuit to Current Champion ---
        if os.path.exists(os.path.join(log_dir, "circuit_robust.json")):
            with open(os.path.join(log_dir, "circuit_robust.json"), 'r') as f:
                circuit_data = json.load(f)
            champion_circuit = cirq.read_json(json_text=json.dumps(circuit_data))
            saboteur_env.set_circuit(champion_circuit)

        # --- SABOTEUR TRAINING LOOP ---
        print("  -> Training Saboteur against Architect's current best circuit...")
        sab_log_callback = AdversarialLoggerCallback(
            lists_dict={
                'fidelity': saboteur_fidelities,
                'error_rate': saboteur_error_rates
            },
            step_list=saboteur_steps,
            offset_steps=total_sab_steps
        )
        
        saboteur_agent.learn(
            total_timesteps=saboteur_steps_per_generation,
            callback=[sab_log_callback]
        )
        total_sab_steps += saboteur_steps_per_generation

    # --- Save Logs ---
    np.savetxt(os.path.join(log_dir, "architect_fidelities.txt"), np.array(architect_fidelities))
    np.savetxt(os.path.join(log_dir, "architect_complexity.txt"), np.array(architect_complexity))
    np.savetxt(os.path.join(log_dir, "architect_steps.txt"), np.array(architect_steps))

    np.savetxt(os.path.join(log_dir, "saboteur_trained_on_architect_fidelities.txt"), np.array(saboteur_fidelities))
    np.savetxt(os.path.join(log_dir, "saboteur_error_rates.txt"), np.array(saboteur_error_rates))
    np.savetxt(os.path.join(log_dir, "saboteur_trained_on_architect_steps.txt"), np.array(saboteur_steps))

    hof_path = os.path.join(log_dir, "hall_of_fame.json")
    with open(hof_path, "w") as f:
        json.dump(hall_of_fame, f, indent=2)

    # --- Post-training evaluation of champions under final saboteur ---
    def evaluate_attacked_fidelity(circuit: cirq.Circuit) -> float:
        """Evaluate attacked fidelity with current saboteur_agent using budgeted top-k noise."""
        if circuit is None or saboteur_agent is None:
            return -1.0
        from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
        ops = list(circuit.all_operations())
        qubits = sorted(list(circuit.all_qubits()))
        sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(
            circuit, n_qubits=n_qubits, max_circuit_timesteps=max_circuit_gates
        )
        sab_action, _ = saboteur_agent.predict(sab_obs, deterministic=True)
        all_rates = SaboteurMultiGateEnv.all_error_rates
        max_idx = len(all_rates) - 1
        valid_gate_count = min(len(ops), max_circuit_gates)
        raw_action = np.array(sab_action[:valid_gate_count], dtype=int)
        budget = min(3, valid_gate_count)
        effective_action = np.zeros_like(raw_action)
        if budget > 0 and len(raw_action) > 0:
            top_k = np.argsort(raw_action)[-budget:]
            effective_action[top_k] = raw_action[top_k]
        noisy_ops = []
        for i, op in enumerate(ops):
            noisy_ops.append(op)
            if i < len(effective_action):
                idx = int(effective_action[i])
                idx = max(0, min(idx, max_idx))
                rate = all_rates[idx]
                if rate > 0:
                    for q in op.qubits:
                        noisy_ops.append(cirq.DepolarizingChannel(rate).on(q))
        noisy_circuit = cirq.Circuit(noisy_ops)
        return float(fidelity_pure_target(noisy_circuit, target_state, qubits))

    candidates = []
    # Hall-of-fame candidates (already serialized circuits)
    for idx, entry in enumerate(hall_of_fame):
        candidates.append({
            "label": f"hof_rank_{idx+1}",
            "circuit_data": entry.get("circuit"),
            "metric": entry.get("metric"),
            "clean_metric": entry.get("clean_metric"),
            "metric_name": entry.get("metric_name", "Fidelity"),
            "step": entry.get("step"),
        })

    # Fallback: include last saved champions from disk if any
    robust_path = os.path.join(log_dir, "circuit_robust.json")
    clean_best_path = os.path.join(log_dir, "circuit_clean_best.json")
    if os.path.exists(robust_path):
        with open(robust_path, "r") as f:
            data = json.load(f)
        candidates.append({"label": "robust_champion_file", "circuit_data": data})
    if os.path.exists(clean_best_path):
        with open(clean_best_path, "r") as f:
            data = json.load(f)
        candidates.append({"label": "clean_best_file", "circuit_data": data})

    best_eval = -1.0
    best_label = None
    best_circuit_data = None
    evaluated_candidates = []
    for cand in candidates:
        label = cand.get("label", "candidate")
        data = cand.get("circuit_data")
        if data is None:
            continue
        circuit = cirq.read_json(json_text=json.dumps(data))
        attacked_fid = evaluate_attacked_fidelity(circuit)
        try:
            best_clean = float(fidelity_pure_target(circuit, target_state, sorted(list(circuit.all_qubits()))))
        except Exception:
            best_clean = float("nan")
        evaluated_candidates.append({
            "label": label,
            "attacked_fidelity": attacked_fid,
            "clean_fidelity": best_clean,
            "source_metric": cand.get("metric"),
            "source_clean_metric": cand.get("clean_metric"),
            "metric_name": cand.get("metric_name"),
            "step": cand.get("step"),
        })
        print(f"[ChampionEval] {label}: clean={best_clean:.4f}, attacked={attacked_fid:.4f}")
        if attacked_fid > best_eval:
            best_eval = attacked_fid
            best_label = label
            best_circuit_data = data

    if best_circuit_data is not None:
        # Save evaluated best-under-attack as canonical robust circuit
        with open(os.path.join(log_dir, "circuit_robust.json"), "w") as f:
            json.dump(best_circuit_data, f, indent=2)
        with open(os.path.join(log_dir, "circuit_robust_final.json"), "w") as f:
            json.dump(best_circuit_data, f, indent=2)
        print(f"[ChampionEval] Selected '{best_label}' as robust champion (attacked fidelity={best_eval:.4f})")
    with open(os.path.join(log_dir, "hall_of_fame_evaluated.json"), "w") as f:
        json.dump(evaluated_candidates, f, indent=2)

    # --- Summary JSON ---
    exp_summary = {
        "n_qubits": n_qubits,
        "n_generations": n_generations,
        "architect_steps_per_generation": architect_steps_per_generation,
        "saboteur_steps_per_generation": saboteur_steps_per_generation,
        "max_circuit_gates": max_circuit_gates,
        "lambda_penalty": lambda_penalty,
        "task_mode": effective_task_mode,
        "target_type": effective_target_type,
        "timestamp": timestamp,
        "champion_save_last_steps": champion_save_last_steps,
        "champion_min_step": champion_min_step,
        "hall_of_fame_size": hall_of_fame_size,
        "hall_of_fame_count": len(hall_of_fame),
    }
    with open(os.path.join(log_dir, "experiment_summary.json"), "w") as f:
        json.dump(exp_summary, f, indent=2)

    return architect_agent, saboteur_agent, log_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--n-generations", type=int, default=None,
                        help="Override generations. Defaults to config.EXPERIMENT_PARAMS for the given qubit count.")
    parser.add_argument("--architect-steps", type=int, default=None,
                        help="Override architect steps per generation. Defaults to config.EXPERIMENT_PARAMS.")
    parser.add_argument("--saboteur-steps", type=int, default=None,
                        help="Override saboteur steps per generation. Defaults to config.EXPERIMENT_PARAMS.")
    parser.add_argument("--max-circuit-gates", type=int, default=config.MAX_CIRCUIT_TIMESTEPS)
    parser.add_argument("--fidelity-threshold", type=float, default=0.99)
    parser.add_argument("--lambda-penalty", type=float, default=0.5)
    parser.add_argument("--champion-save-last-steps", type=int, default=None,
                        help="Only save champion circuits during the last N architect steps.")
    parser.add_argument("--hall-of-fame-size", type=int, default=5,
                        help="Keep top-k champions across the whole run for end-of-training evaluation.")
    # Gate set controlled via experiments/config.py (INCLUDE_ROTATIONS/ROTATION_TYPES)
    parser.add_argument("--task-mode", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    start_time = time.time()

    # Resolve per-qubit defaults when user did not override via CLI
    qubit_params = config.get_params_for_qubits(args.n_qubits)
    n_generations = args.n_generations if args.n_generations is not None else qubit_params["N_GENERATIONS"]
    architect_steps = args.architect_steps if args.architect_steps is not None else qubit_params["ARCHITECT_STEPS_PER_GENERATION"]
    saboteur_steps = args.saboteur_steps if args.saboteur_steps is not None else qubit_params["SABOTEUR_STEPS_PER_GENERATION"]

    # Gate set is controlled centrally in experiments/config.py
    include_rotations = config.INCLUDE_ROTATIONS

    architect_agent, saboteur_agent, log_dir = train_adversarial(
        results_dir=args.results_dir,
        n_qubits=args.n_qubits,
        n_generations=n_generations,
        architect_steps_per_generation=architect_steps,
        saboteur_steps_per_generation=saboteur_steps,
        max_circuit_gates=args.max_circuit_gates,
        fidelity_threshold=args.fidelity_threshold,
        lambda_penalty=args.lambda_penalty,
        include_rotations=include_rotations,
        task_mode=args.task_mode,
        champion_save_last_steps=args.champion_save_last_steps,
        hall_of_fame_size=args.hall_of_fame_size,
    )

    architect_agent.save(os.path.join(log_dir, "architect_adversarial.zip"))
    saboteur_agent.save(os.path.join(log_dir, "saboteur_adversarial.zip"))

    print(f"\n--- Training Complete ---")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"Final data and models saved in {log_dir}")
