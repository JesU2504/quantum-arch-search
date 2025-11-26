"""
Stage 7.1 - Saboteur efficacy check.

See ExpPlan.md, Part 7.1:
  - Test: Load a perfect GHZ circuit. Let the Saboteur act for 1 step
    with max budget.
  - Verify: Fidelity must drop significantly below 1.0.
  - Why: If fidelity stays 1.0, noise injection is broken and the
    Architect trains against a phantom.

This test validates that the Saboteur environment correctly injects
noise into quantum circuits.

TODO: Implement full test once Saboteur class is complete.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cirq

from utils.metrics import (
    ghz_circuit,
    ideal_ghz_state,
    state_fidelity,
    simulate_circuit,
)
from envs.saboteur import Saboteur


def test_saboteur_drops_ghz_fidelity():
    """
    Test that Saboteur reduces GHZ circuit fidelity.

    Steps:
    1. Create a perfect GHZ circuit
    2. Compute initial fidelity (should be ~1.0)
    3. Apply Saboteur attack with maximum budget
    4. Verify fidelity drops significantly (< 0.95)

    This ensures the noise injection mechanism is working correctly.
    """
    n_qubits = 4

    # Step 1: Create a perfect 4-qubit GHZ circuit
    circuit, qubits = ghz_circuit(n_qubits)

    # Step 2: Compute initial fidelity (should be ~1.0)
    ideal_state = ideal_ghz_state(n_qubits)
    output_state = simulate_circuit(circuit, qubits)
    initial_fidelity = state_fidelity(output_state, ideal_state)

    # Verify initial fidelity is close to 1.0
    assert initial_fidelity > 0.99, (
        f"Initial fidelity should be ~1.0, got {initial_fidelity}"
    )

    # Step 3: Create Saboteur and apply maximum noise attack
    saboteur = Saboteur(
        target_circuit=circuit,
        target_state=ideal_state,
        qubits=qubits,
    )

    # Apply max noise for one step
    noisy_circuit, noisy_qubits = saboteur.apply_max_noise()

    # Step 4: Compute fidelity after noise and verify it dropped
    # Use density matrix simulator for noisy circuit
    dm_simulator = cirq.DensityMatrixSimulator()
    dm_result = dm_simulator.simulate(noisy_circuit, qubit_order=noisy_qubits)
    noisy_dm = dm_result.final_density_matrix

    # Compute fidelity: F = <ideal|rho|ideal>
    noisy_fidelity = np.real(
        np.dot(np.conj(ideal_state), np.dot(noisy_dm, ideal_state))
    )

    # Verify fidelity dropped significantly (< 0.95)
    assert noisy_fidelity < 0.95, (
        f"Noisy fidelity should be < 0.95, got {noisy_fidelity}. "
        "Noise injection may not be working correctly."
    )


def test_saboteur_respects_attack_budget():
    """
    Test that Saboteur respects the max_concurrent_attacks budget.

    The Saboteur should only attack a limited number of gates per step,
    even if the action requests attacks on more gates.
    """
    n_qubits = 4
    max_gates = 20

    # Step 1: Create a circuit with many gates
    circuit, qubits = ghz_circuit(n_qubits)

    # Step 2: Create Saboteur with small attack budget (2)
    attack_budget = 2
    saboteur = Saboteur(
        target_circuit=circuit,
        target_state=ideal_ghz_state(n_qubits),
        qubits=qubits,
        max_concurrent_attacks=attack_budget,
        max_gates=max_gates,
    )

    # Step 3: Request attacks on ALL gates (more than budget allows)
    num_ops = len(list(circuit.all_operations()))
    # Action: attack all gates with max error level (index 3 = 0.01)
    action = [3] * num_ops + [0] * (max_gates - num_ops)

    # Step 4: Apply noise and verify only budget-limited attacks applied
    noisy_circuit, num_attacks = saboteur.apply_noise(circuit, action)

    # Verify budget was respected
    assert num_attacks <= attack_budget, (
        f"Number of attacks ({num_attacks}) should not exceed budget "
        f"({attack_budget})"
    )
    assert num_attacks == attack_budget, (
        f"Expected exactly {attack_budget} attacks when {num_ops} were "
        f"requested, got {num_attacks}"
    )


def test_saboteur_zero_noise_preserves_fidelity():
    """
    Test that zero-noise action preserves circuit fidelity.

    When the Saboteur chooses error_rate=0.0 for all gates,
    the circuit fidelity should remain unchanged.
    """
    n_qubits = 4
    max_gates = 20

    # Step 1: Create a perfect GHZ circuit
    circuit, qubits = ghz_circuit(n_qubits)
    ideal_state = ideal_ghz_state(n_qubits)

    # Step 2: Create Saboteur
    saboteur = Saboteur(
        target_circuit=circuit,
        target_state=ideal_state,
        qubits=qubits,
        max_gates=max_gates,
    )

    # Step 3: Apply zero-noise action (all error levels = 0)
    action = [0] * max_gates  # All zeros = no noise

    noisy_circuit, num_attacks = saboteur.apply_noise(circuit, action)

    # Step 4: Verify fidelity remains ~1.0
    # Since we didn't add noise, we can use pure state simulation
    output_state = simulate_circuit(noisy_circuit, qubits)
    fidelity = state_fidelity(output_state, ideal_state)

    assert fidelity > 0.99, (
        f"Zero-noise fidelity should be ~1.0, got {fidelity}"
    )
    assert num_attacks == 0, (
        f"Zero-noise action should result in 0 attacks, got {num_attacks}"
    )


def test_saboteur_observation_shape():
    """
    Test that Saboteur produces correct observation shape.

    The observation should be a dict with:
      - 'projected_state': shape (2 * n_qubits,)
      - 'gate_structure': shape (max_gates,)
    """
    n_qubits = 4
    max_gates = 20

    # Step 1: Create a GHZ circuit
    circuit, qubits = ghz_circuit(n_qubits)
    ideal_state = ideal_ghz_state(n_qubits)

    # Step 2: Create Saboteur environment
    saboteur = Saboteur(
        target_circuit=circuit,
        target_state=ideal_state,
        qubits=qubits,
        max_gates=max_gates,
    )

    # Step 3: Reset and get observation
    observation, info = saboteur.reset()

    # Step 4: Verify observation structure and shapes
    assert isinstance(observation, dict), (
        f"Observation should be a dict, got {type(observation)}"
    )
    assert 'projected_state' in observation, (
        "Observation should contain 'projected_state'"
    )
    assert 'gate_structure' in observation, (
        "Observation should contain 'gate_structure'"
    )

    # Verify shapes
    expected_projected_shape = (2 * n_qubits,)
    expected_gate_shape = (max_gates,)

    assert observation['projected_state'].shape == expected_projected_shape, (
        f"projected_state shape should be {expected_projected_shape}, "
        f"got {observation['projected_state'].shape}"
    )
    assert observation['gate_structure'].shape == expected_gate_shape, (
        f"gate_structure shape should be {expected_gate_shape}, "
        f"got {observation['gate_structure'].shape}"
    )


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("Running Stage 7.1 tests...")
    test_saboteur_drops_ghz_fidelity()
    test_saboteur_respects_attack_budget()
    test_saboteur_zero_noise_preserves_fidelity()
    test_saboteur_observation_shape()
    print("All Stage 7.1 tests passed (or skipped with TODO).")
