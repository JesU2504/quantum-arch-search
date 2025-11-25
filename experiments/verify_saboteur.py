import gymnasium as gym
import cirq # qas_gym is implicitly imported via gym.make
import numpy as np
from experiments import config
from qas_gym.utils import get_ghz_state

def verify_saboteur_noise_application(n_qubits):
    """
    This experiment verifies that the SaboteurEnv correctly applies noise to a circuit.
    It creates a simple circuit, calculates its initial fidelity, and then steps through
    each possible saboteur action, printing the resulting noisy circuit and the new fidelity.

    Args:
        n_qubits (int): The number of qubits to use for the verification circuit.
    """
    print(f"--- Experiment 1: Verify Saboteur's Noise Application for {n_qubits} Qubits ---")

    # 1. Create a simple, fixed quantum circuit
    qubits = cirq.LineQubit.range(n_qubits)
    # Use the canonical GHZ state preparation circuit for consistency with the rest of the project.
    original_circuit = cirq.Circuit()
    original_circuit.append(cirq.H(qubits[0]))
    if n_qubits > 1:
        # Append explicit CNOT operations for qubits 1..n_qubits-1
        cnot_ops = [cirq.CNOT(qubits[0], qubits[i]) for i in range(1, n_qubits)]
        original_circuit.append(cnot_ops)
    print("Original Circuit:")
    print(original_circuit)
    print("\n")

    # 2. Get the canonical GHZ target state using the project's utility function.
    target_state = get_ghz_state(n_qubits)

    # Calculate and print the fidelity of the original circuit (should be 1.0)
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
