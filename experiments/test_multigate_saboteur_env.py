import numpy as np
import cirq
import gymnasium as gym
from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
from qas_gym.utils import get_ghz_state

def test_saboteur_env_structure():
    print("--- Starting Saboteur Env Smoke Test ---")

    n_qubits = 3
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    for i in range(n_qubits-1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    target = get_ghz_state(n_qubits)
    
    # max_circuit_timesteps matches MAX_CIRCUIT_TIMESTEPS in config
    max_gates = 20
    env = SaboteurMultiGateEnv(
        architect_circuit=circuit, 
        target_state=target, 
        max_circuit_timesteps=max_gates
    )

    print("\n[Check 1] Observation Space")
    obs, info = env.reset()
    
    # This verifies the new Dictionary structure
    assert isinstance(obs, dict), "Observation must be a dictionary!"
    assert 'projected_state' in obs, "Obs missing 'projected_state' key"
    assert 'gate_structure' in obs, "Obs missing 'gate_structure' key"
    
    print(f"  Keys found: {list(obs.keys())}")
    print(f"  State shape: {obs['projected_state'].shape}")
    print(f"  Structure shape: {obs['gate_structure'].shape}")
    
    print("\n[Check 2] Action Space")
    print(f"  Action shape: {env.action_space.shape} (Expected: {max_gates})")

    print("\n[Check 3] Stepping")
    action = env.action_space.sample()
    print(f"  Sampled Action (first 5): {action[:5]}...")
    
    obs2, reward, terminated, truncated, info = env.step(action)
    
    print("  Step Successful!")
    print(f"  Reward: {reward}")
    
    print("\n--- Test Passed! ---")

if __name__ == "__main__":
    test_saboteur_env_structure()