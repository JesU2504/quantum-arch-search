import os
import sys

# Add repository root to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import gymnasium as gym
import cirq  # qas_gym is implicitly imported via gym.make
import numpy as np
from experiments import config

def verify_saboteur_noise_application(n_qubits):
    """
    This experiment verifies that the SaboteurEnv correctly applies noise to a circuit.
    It creates a simple circuit, calculates its initial fidelity, and then steps through
    each possible saboteur action, printing the resulting noisy circuit and the new fidelity.

    Args:
        n_qubits (int): The number of qubits to use for the verification circuit.
    """
    print(f"--- Experiment 1: Verify Saboteur's Noise Application for {n_qubits} Qubits ---")
    print(f"Target Type: {config.TARGET_TYPE}")

    # 1. Get the target circuit and state from the central config
    original_circuit, _ = config.get_target_circuit(n_qubits)
    target_state = config.get_target_state(n_qubits)
    
    print("Original Circuit:")
    print(original_circuit)
    print("\n")

    # 2. Calculate and print the fidelity of the original circuit
    initial_state_vector = cirq.Simulator().simulate(original_circuit).final_state_vector
    original_fidelity = np.abs(np.vdot(target_state, initial_state_vector))**2
    print(f"Fidelity of original circuit: {original_fidelity:.4f}\n")

    # 3. Iterate through every possible per-gate noise vector (for small circuits)
    from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
    env = SaboteurMultiGateEnv(
        architect_circuit=original_circuit,
        target_state=target_state,
        max_error_level=4,
        discrete=True
    )
    num_gates = env.num_gates
    num_error_levels = env.max_error_level
    print(f"Action space dimensions: num_gates={num_gates}, num_error_levels={num_error_levels}\n")

    # For small circuits, sweep all combinations
    from itertools import product
    error_indices = list(range(num_error_levels))
    for action in product(error_indices, repeat=num_gates):
        print(f"--- Applying per-gate error indices: {action} ---")
        env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        noisy_fidelity = info['fidelity']
        noisy_circuit = info['noisy_circuit']
        per_gate_error = info.get('per_gate_error', None)
        print(f"Noisy Circuit (after action {action}):")
        print(noisy_circuit)
        print(f"Applied per-gate noise levels: {per_gate_error}")
        print(f"Fidelity after noise: {noisy_fidelity:.6f}\n")
    env.close()

if __name__ == "__main__":
    verify_saboteur_noise_application(n_qubits=3) # Default to 3 qubits for standalone run
