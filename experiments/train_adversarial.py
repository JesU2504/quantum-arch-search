import os
import sys

# Add repository root to sys.path for standalone execution
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
# Import your custom environments and utilities
from qas_gym.envs import SaboteurMultiGateEnv, AdversarialArchitectEnv
from qas_gym.utils import get_bell_state, get_ghz_state, get_toffoli_state, save_circuit

# --- 1. Improved Logger Callback ---
class FidelityLoggerCallback(BaseCallback):
    def __init__(self, trace_list, step_list, offset_steps, verbose=0):
        super().__init__(verbose)
        self.trace_list = trace_list
        self.step_list = step_list
        self.offset_steps = offset_steps # The total steps from previous generations

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    # Get fidelity from info
                    info = self.locals['infos'][i]
                    fid = info.get('fidelity', None)
                    
                    if fid is not None:
                        self.trace_list.append(fid)
                        # LOGIC FIX: Current global timestep = offset + current_run_steps
                        current_timesteps = self.num_timesteps
                        self.step_list.append(self.offset_steps + current_timesteps)
        return True

def train_adversarial(
    results_dir: str,
    n_qubits: int,
    n_generations: int,
    architect_steps_per_generation: int,
    saboteur_steps_per_generation: int,
    max_circuit_gates: int = 20,
    fidelity_threshold: float = 0.99,
    lambda_penalty: float = 0.5,
):
    print("--- Starting Adversarial Co-evolutionary Training ---")
    start_time = time.time()
    log_dir = os.path.join(results_dir, f"adversarial_training_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    # --- Target State ---
    # Use n-controlled Toffoli gate as the default target for 3+ qubits
    # For 2 qubits, use Bell state (legacy behavior)
    if n_qubits == 2:
        target_state = get_bell_state()
    else:
        # n-controlled Toffoli: CCNOT for 3 qubits, CCCNOT for 4 qubits, etc.
        target_state = get_toffoli_state(n_qubits)

    # --- Initialize Environments ---
    max_saboteur_level = len(SaboteurMultiGateEnv.all_error_rates)

    # Architect env args
    arch_env_kwargs = dict(
        target=target_state,
        fidelity_threshold=fidelity_threshold,
        max_timesteps=max_circuit_gates,
        reward_penalty=0.01,
        complexity_penalty_weight=0.01,
    )

    # Dummy circuit for initialization
    dummy_qubits = list(cirq.LineQubit.range(int(np.log2(len(target_state)))))
    dummy_circuit = cirq.Circuit([cirq.I(q) for q in dummy_qubits])
    
    # Initialize Saboteur Environment
    saboteur_env = SaboteurMultiGateEnv(
        architect_circuit=dummy_circuit,
        target_state=target_state,
        max_circuit_timesteps=max_circuit_gates,
        max_error_level=4,
        discrete=True,
        lambda_penalty=lambda_penalty
    )

    # Initialize Architect Environment
    # We start with no saboteur agent for the very first initialization
    initial_arch_env = AdversarialArchitectEnv(
        saboteur_agent=None,
        saboteur_max_error_level=max_saboteur_level,
        **arch_env_kwargs
    )

    # --- Initialize Agents ---
    architect_agent = PPO('MlpPolicy', initial_arch_env, **config.AGENT_PARAMS)
    saboteur_agent = PPO('MultiInputPolicy', saboteur_env, **config.AGENT_PARAMS)

    # Data Storage
    architect_fidelities = []
    architect_steps = []
    saboteur_fidelities = []
    saboteur_steps = []
    
    best_fidelity = -1.0
    champion_circuit = None
    
    # Track total steps cumulatively to avoid gaps in plot
    total_arch_steps = 0
    total_sab_steps = 0

    for gen in range(n_generations):
        print(f"\n--- Generation {gen+1}/{n_generations} ---")

        # -----------------------------
        # 1. Train Architect
        # -----------------------------
        arch_env = AdversarialArchitectEnv(
            saboteur_agent=saboteur_agent,
            saboteur_max_error_level=max_saboteur_level,
            **arch_env_kwargs
        )
        architect_agent.set_env(arch_env)
        
        arch_callback = FidelityLoggerCallback(
            trace_list=architect_fidelities,
            step_list=architect_steps,
            offset_steps=total_arch_steps # Pass current total
        )
        
        architect_agent.learn(
            total_timesteps=architect_steps_per_generation, 
            reset_num_timesteps=False, 
            callback=arch_callback
        )
        total_arch_steps += architect_steps_per_generation # Update total

        # Check for champion
        if hasattr(arch_env, 'champion_circuit') and arch_env.champion_circuit is not None:
            fid = arch_env.best_fidelity
            if fid > best_fidelity:
                best_fidelity = fid
                champion_circuit = arch_env.champion_circuit

        # -----------------------------
        # 2. Train Saboteur
        # -----------------------------
        if champion_circuit is not None:
            saboteur_env.set_circuit(champion_circuit)
        else:
            saboteur_env.set_circuit(dummy_circuit)

        # CRITICAL FIX: Dict observations require MultiInputPolicy
        saboteur_agent = PPO('MultiInputPolicy', saboteur_env, **config.AGENT_PARAMS)
        
        sab_callback = FidelityLoggerCallback(
            trace_list=saboteur_fidelities,
            step_list=saboteur_steps,
            offset_steps=total_sab_steps
        )

        saboteur_agent.learn(
            total_timesteps=saboteur_steps_per_generation, 
            reset_num_timesteps=False, 
            callback=sab_callback
        )
        total_sab_steps += saboteur_steps_per_generation

        # -----------------------------
        # 3. SAFETY SAVE (Inside Loop)
        # -----------------------------
        np.savetxt(os.path.join(log_dir, "architect_fidelities.txt"), architect_fidelities)
        np.savetxt(os.path.join(log_dir, "architect_steps.txt"), architect_steps, fmt='%d')
        np.savetxt(os.path.join(log_dir, "saboteur_trained_on_architect_fidelities.txt"), saboteur_fidelities)
        np.savetxt(os.path.join(log_dir, "saboteur_trained_on_architect_steps.txt"), saboteur_steps, fmt='%d')
        
        if champion_circuit is not None:
            save_circuit(os.path.join(log_dir, "circuit_robust.json"), champion_circuit)

    return architect_agent, saboteur_agent, arch_env, log_dir, start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--n-qubits', type=int, required=True, help='Number of qubits')
    parser.add_argument('--n-generations', type=int, required=True, help='Number of generations')
    parser.add_argument('--architect-steps', type=int, required=True, help='Architect steps per generation')
    parser.add_argument('--saboteur-steps', type=int, required=True, help='Saboteur steps per generation')
    parser.add_argument('--max-circuit-gates', type=int, default=20, help='Max circuit gates')
    parser.add_argument('--fidelity-threshold', type=float, default=0.99, help='Fidelity threshold')
    parser.add_argument('--lambda-penalty', type=float, default=0.5, help='Penalty coefficient for mean error rate')
    args = parser.parse_args()

    architect_agent, saboteur_agent, arch_env, log_dir, start_time = train_adversarial(
        results_dir=args.results_dir,
        n_qubits=args.n_qubits,
        n_generations=args.n_generations,
        architect_steps_per_generation=args.architect_steps,
        saboteur_steps_per_generation=args.saboteur_steps,
        max_circuit_gates=args.max_circuit_gates,
        fidelity_threshold=args.fidelity_threshold,
        lambda_penalty=args.lambda_penalty
    )

    # Final Save
    architect_agent.save(os.path.join(log_dir, "architect_adversarial.zip"))
    saboteur_agent.save(os.path.join(log_dir, "saboteur_adversarial.zip"))

    print(f"\n--- Training Complete ---")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"Final data and models saved in {log_dir}")