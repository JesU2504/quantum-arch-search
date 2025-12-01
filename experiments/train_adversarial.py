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
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import time
import argparse
from datetime import datetime
import numpy as np
import cirq
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from experiments import config
from qas_gym.envs import SaboteurMultiGateEnv, AdversarialArchitectEnv
from qas_gym.utils import save_circuit, verify_toffoli_unitary, get_ideal_unitary

# --- 1. Robust Champion Callback ---
class ChampionCircuitCallback(BaseCallback):
    """
    A callback to save the best circuit found during training.
    Checks ALL parallel environments and prints in real-time.
    """
    def __init__(self, circuit_filename, verbose=0):
        super().__init__(verbose)
        self.circuit_filename = circuit_filename
        self.best_saved_fidelity = -1.0
        self.circuit_saved = False
        self.verify_unitary = False
        self.n_qubits = None
        self.unitary_mode = False
        self.last_champion_circuit = None

    def _on_step(self) -> bool:
        if 'infos' in self.locals and self.locals['infos'] and self.locals['dones'] is not None:
            dones = self.locals['dones']
            infos = self.locals['infos']
            
            for i, done in enumerate(dones):
                if not done:
                    continue
                
                info = infos[i]
                if not self.unitary_mode and self.verify_unitary:
                    self.unitary_mode = True

                candidate_circuit = info.get('circuit')
                candidate_fid = info.get('fidelity', 0.0)

                if candidate_circuit is None:
                    try:
                        candidate_circuit = self.training_env.env_method('get_circuit', indices=[i])[0]
                    except Exception:
                        continue

                if candidate_circuit:
                    # Metric selection (Unitary vs State)
                    metric_to_check = candidate_fid
                    metric_name = "Fidelity"
                    
                    if self.verify_unitary and self.n_qubits is not None:
                        try:
                            _, process_fid = verify_toffoli_unitary(candidate_circuit, self.n_qubits, silent=True)
                            metric_to_check = process_fid
                            metric_name = "Process Fidelity"
                        except Exception:
                            pass

                    # Save if improved
                    if (self.best_saved_fidelity < 0) or (metric_to_check > self.best_saved_fidelity):
                        print(f"\n--- New Champion Circuit Found (Env {i}) ---")
                        print(f"{metric_name}: {metric_to_check:.6f}")
                        print(candidate_circuit)
                        save_circuit(self.circuit_filename, candidate_circuit)
                        print(f"Saved new champion circuit to {self.circuit_filename}\n")
                        
                        self.best_saved_fidelity = metric_to_check
                        self.last_champion_circuit = candidate_circuit
                        self.circuit_saved = True
        return True

# --- 2. Enhanced Logger Callback ---
class AdversarialLoggerCallback(BaseCallback):
    """
    Generic logger that can track Fidelity, Complexity (Gate Count), 
    and Error Rates from the environment info dict.
    """
    def __init__(self, lists_dict, step_list, offset_steps, verbose=0):
        super().__init__(verbose)
        self.lists_dict = lists_dict  # {'fidelity': [], 'complexity': [], 'error_rate': []}
        self.step_list = step_list
        self.offset_steps = offset_steps 

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    
                    # Log Step
                    current_timesteps = self.num_timesteps
                    self.step_list.append(self.offset_steps + current_timesteps)

                    # Log Fidelity (Common to both)
                    if 'fidelity' in self.lists_dict:
                        # For Architect, this is clean fidelity. For Saboteur, this is noisy fidelity.
                        self.lists_dict['fidelity'].append(info.get('fidelity', 0.0))
                    
                    # Log Complexity (Architect specific)
                    if 'complexity' in self.lists_dict:
                        circ = info.get('circuit')
                        if circ:
                            self.lists_dict['complexity'].append(len(list(circ.all_operations())))
                        else:
                            self.lists_dict['complexity'].append(0)

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
    include_rotations: bool = False,
    task_mode: str = None,
):
    # --- Configuration ---
    effective_task_mode = task_mode if task_mode is not None else config.TASK_MODE
    effective_target_type = config.TARGET_TYPE
    rotation_status = "with rotation gates" if include_rotations else "with Clifford+T gates only"
    
    print("\n" + "=" * 60)
    print("ADVERSARIAL TRAINING CONFIGURATION (ENHANCED LOGGING)")
    print("=" * 60)
    print(f"Target Type: {effective_target_type}")
    print(f"Task Mode: {effective_task_mode}")
    print(f"Number of Qubits: {n_qubits}")
    print(f"Generations: {n_generations}")
    print(f"Architect Steps/Gen: {architect_steps_per_generation}")
    print(f"Saboteur Steps/Gen: {saboteur_steps_per_generation}")
    print(f"Max Circuit Gates: {max_circuit_gates}")
    print(f"Results Directory: {results_dir}")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    log_dir = os.path.join(results_dir, f"adversarial_training_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    target_state = config.get_target_state(n_qubits, effective_target_type)
    
    # Ideal unitary for verification
    ideal_U = None
    if effective_task_mode == 'unitary_preparation':
        try:
            ideal_U = get_ideal_unitary(n_qubits, effective_target_type)
        except Exception:
            ideal_U = None

    # --- Initialize Environments ---
    max_saboteur_level = len(SaboteurMultiGateEnv.all_error_rates)
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

    dummy_qubits = list(cirq.LineQubit.range(int(np.log2(len(target_state)))))
    dummy_circuit = cirq.Circuit([cirq.I(q) for q in dummy_qubits])
    
    # NOTE: Saboteur environment also needs to know the max_circuit_gates
    # to set its observation space size correctly.
    saboteur_env = SaboteurMultiGateEnv(
        architect_circuit=dummy_circuit,
        target_state=target_state,
        max_circuit_timesteps=max_circuit_gates, # Ensure this matches architect
        max_error_level=4,
        discrete=True,
        lambda_penalty=lambda_penalty
    )

    # --- Initialize Agents (Persistent Saboteur) ---
    saboteur_agent = PPO('MultiInputPolicy', saboteur_env, **config.AGENT_PARAMS)

    initial_arch_env = AdversarialArchitectEnv(
        saboteur_agent=None,
        saboteur_max_error_level=max_saboteur_level,
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

    # --- Main Loop ---
    for gen in range(n_generations):
        print(f"\n--- Generation {gen+1}/{n_generations} ---")

        # 1. Train Architect
        arch_env = AdversarialArchitectEnv(
            saboteur_agent=saboteur_agent,
            saboteur_max_error_level=max_saboteur_level,
            **arch_env_kwargs
        )
        architect_agent.set_env(arch_env)
        
        # Real-time champion detection
        arch_champ_callback = ChampionCircuitCallback(
            circuit_filename=os.path.join(log_dir, "circuit_robust.json")
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
        
        architect_agent.learn(
            total_timesteps=architect_steps_per_generation, 
            reset_num_timesteps=False, 
            callback=[arch_log_callback, arch_champ_callback]
        )
        total_arch_steps += architect_steps_per_generation

        # Retrieve champion
        if arch_champ_callback.last_champion_circuit is not None:
            champion_circuit = arch_champ_callback.last_champion_circuit
            print(f"Gen {gen+1} Architect finished. Best circuit updated.")
        elif hasattr(arch_env, 'champion_circuit') and arch_env.champion_circuit is not None:
             champion_circuit = arch_env.champion_circuit

        # 2. Train Saboteur
        if champion_circuit is not None:
            saboteur_env.set_circuit(champion_circuit)
            print(f"Saboteur now attacking champion with {len(list(champion_circuit.all_operations()))} gates.")
        else:
            saboteur_env.set_circuit(dummy_circuit)
            print("Saboteur attacking dummy circuit (no champion found yet).")

        # Data logging (Fidelity + Error Rate)
        sab_log_callback = AdversarialLoggerCallback(
            lists_dict={
                'fidelity': saboteur_fidelities,
                'error_rate': saboteur_error_rates
            },
            step_list=saboteur_steps,
            offset_steps=total_sab_steps
        )

        saboteur_agent.set_env(saboteur_env)
        saboteur_agent.learn(
            total_timesteps=saboteur_steps_per_generation, 
            reset_num_timesteps=False, 
            callback=sab_log_callback
        )
        total_sab_steps += saboteur_steps_per_generation

        # 3. Save Data (Including New Metrics)
        np.savetxt(os.path.join(log_dir, "architect_fidelities.txt"), architect_fidelities)
        np.savetxt(os.path.join(log_dir, "architect_complexity.txt"), architect_complexity) # Saved!
        np.savetxt(os.path.join(log_dir, "architect_steps.txt"), architect_steps, fmt='%d')
        
        np.savetxt(os.path.join(log_dir, "saboteur_trained_on_architect_fidelities.txt"), saboteur_fidelities)
        np.savetxt(os.path.join(log_dir, "saboteur_error_rates.txt"), saboteur_error_rates) # Saved!
        np.savetxt(os.path.join(log_dir, "saboteur_trained_on_architect_steps.txt"), saboteur_steps, fmt='%d')

    return architect_agent, saboteur_agent, arch_env, log_dir, start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--n-qubits', type=int, required=True, help='Number of qubits')
    parser.add_argument('--n-generations', type=int, required=True, help='Number of generations')
    parser.add_argument('--architect-steps', type=int, required=True, help='Architect steps per generation')
    parser.add_argument('--saboteur-steps', type=int, required=True, help='Saboteur steps per generation')
    
    # UPDATED: Use config.MAX_CIRCUIT_TIMESTEPS as default
    parser.add_argument('--max-circuit-gates', type=int, default=config.MAX_CIRCUIT_TIMESTEPS, 
                        help=f'Max circuit gates (default: {config.MAX_CIRCUIT_TIMESTEPS} from config)')
                        
    parser.add_argument('--fidelity-threshold', type=float, default=0.99, help='Fidelity threshold')
    parser.add_argument('--lambda-penalty', type=float, default=0.5, help='Penalty coefficient for mean error rate')
    parser.add_argument('--include-rotations', action='store_true',
                        help='Include parameterized rotation gates (Rx, Ry, Rz) in action space')
    parser.add_argument('--task-mode', type=str, default=None, choices=['state_preparation', 'unitary_preparation'],
        help='Task mode for training. Overrides config.TASK_MODE if set.')
    args = parser.parse_args()

    architect_agent, saboteur_agent, arch_env, log_dir, start_time = train_adversarial(
        results_dir=args.results_dir,
        n_qubits=args.n_qubits,
        n_generations=args.n_generations,
        architect_steps_per_generation=args.architect_steps,
        saboteur_steps_per_generation=args.saboteur_steps,
        max_circuit_gates=args.max_circuit_gates,
        fidelity_threshold=args.fidelity_threshold,
        lambda_penalty=args.lambda_penalty,
        include_rotations=args.include_rotations,
        task_mode=args.task_mode,
    )

    architect_agent.save(os.path.join(log_dir, "architect_adversarial.zip"))
    saboteur_agent.save(os.path.join(log_dir, "saboteur_adversarial.zip"))

    print(f"\n--- Training Complete ---")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"Final data and models saved in {log_dir}")