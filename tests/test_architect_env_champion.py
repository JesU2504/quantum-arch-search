import os
import cirq
import numpy as np
from qas_gym.envs import ArchitectEnv
from qas_gym.utils import create_ghz_circuit_and_qubits, get_ghz_state


def test_architect_env_sets_champion():
    n_qubits = 3
    target_state = get_ghz_state(n_qubits)

    # Create an ArchitectEnv with small max_timesteps so the GHZ sequence fits
    env = ArchitectEnv(
        target=target_state,
        fidelity_threshold=1.0,
        reward_penalty=0.0,
        max_timesteps=10,
        complexity_penalty_weight=0.0,
    )

    # Build the canonical GHZ circuit and use its operations as the allowed action gates
    ghz_circuit, ghz_qubits = create_ghz_circuit_and_qubits(n_qubits)
    ghz_ops = list(ghz_circuit.all_operations())

    # Replace the env's action set with the GHZ ops so stepping through them reconstructs GHZ
    env.action_gates = ghz_ops

    obs, info = env.reset()

    # Step through the GHZ operations by taking actions that select each op in order
    for i in range(len(ghz_ops)):
        action = i  # index into env.action_gates
        obs, reward, terminated, truncated, info = env.step(action)

    # After applying the full GHZ sequence, the env should have recorded a champion
    assert env.champion_circuit is not None, "ArchitectEnv did not record a champion circuit"
    assert env.best_fidelity >= 0.9999, f"Expected fidelity ~1.0, got {env.best_fidelity}"
    # The info from the last step should also indicate champion (or the env state reflects it)
    assert 'is_champion' in info or env.best_fidelity >= 0.9999

    # Ensure the champion circuit produces a high fidelity
    final_fid = env.get_fidelity(env.champion_circuit)
    assert final_fid >= 0.9999
