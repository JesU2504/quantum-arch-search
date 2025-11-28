"""
Tests for mixed_state_fidelity function in src/utils/metrics.py.

This test module validates the fidelity computation between a pure target
state and a noisy (density matrix) output using F = <psi|rho|psi>.
"""

import numpy as np
import pytest
import cirq

from src.utils.metrics import mixed_state_fidelity, ideal_ghz_state, ghz_circuit


def test_mixed_state_fidelity_pure_state():
    """
    Test fidelity of a pure state with itself (as density matrix).

    When rho = |psi><psi|, fidelity should be 1.0.
    """
    # Create a simple pure state |0>
    pure_state = np.array([1, 0], dtype=complex)
    # Density matrix for |0><0|
    density_matrix = np.outer(pure_state, np.conj(pure_state))

    fidelity = mixed_state_fidelity(pure_state, density_matrix)

    assert np.isclose(fidelity, 1.0), f"Expected 1.0, got {fidelity}"


def test_mixed_state_fidelity_orthogonal_state():
    """
    Test fidelity of orthogonal states.

    When target state is orthogonal to the density matrix's support,
    fidelity should be 0.0.
    """
    # Target state |0>
    pure_state = np.array([1, 0], dtype=complex)
    # Density matrix for |1><1| (orthogonal)
    other_state = np.array([0, 1], dtype=complex)
    density_matrix = np.outer(other_state, np.conj(other_state))

    fidelity = mixed_state_fidelity(pure_state, density_matrix)

    assert np.isclose(fidelity, 0.0), f"Expected 0.0, got {fidelity}"


def test_mixed_state_fidelity_maximally_mixed():
    """
    Test fidelity with maximally mixed state.

    For a single qubit, rho = I/2 should give fidelity = 0.5 for any pure state.
    """
    # Target state |0>
    pure_state = np.array([1, 0], dtype=complex)
    # Maximally mixed state I/2
    density_matrix = np.eye(2) / 2

    fidelity = mixed_state_fidelity(pure_state, density_matrix)

    assert np.isclose(fidelity, 0.5), f"Expected 0.5, got {fidelity}"


def test_mixed_state_fidelity_partial_noise():
    """
    Test fidelity with partially mixed state.

    rho = 0.9|0><0| + 0.1|1><1| should give F = 0.9 for target |0>.
    """
    pure_state = np.array([1, 0], dtype=complex)
    # Partial mixture
    density_matrix = np.diag([0.9, 0.1])

    fidelity = mixed_state_fidelity(pure_state, density_matrix)

    assert np.isclose(fidelity, 0.9), f"Expected 0.9, got {fidelity}"


def test_mixed_state_fidelity_superposition():
    """
    Test fidelity with superposition state.

    |+> = (|0> + |1>)/sqrt(2), rho = |+><+| should give F = 1.0.
    """
    # |+> state
    pure_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
    # Density matrix for |+><+|
    density_matrix = np.outer(pure_state, np.conj(pure_state))

    fidelity = mixed_state_fidelity(pure_state, density_matrix)

    assert np.isclose(fidelity, 1.0), f"Expected 1.0, got {fidelity}"


def test_mixed_state_fidelity_two_qubits():
    """
    Test fidelity with 2-qubit GHZ state.
    """
    # 2-qubit GHZ: (|00> + |11>)/sqrt(2)
    pure_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    # Density matrix for GHZ state
    density_matrix = np.outer(pure_state, np.conj(pure_state))

    fidelity = mixed_state_fidelity(pure_state, density_matrix)

    assert np.isclose(fidelity, 1.0), f"Expected 1.0, got {fidelity}"


def test_mixed_state_fidelity_with_cirq_simulation():
    """
    Test mixed_state_fidelity with actual Cirq density matrix simulation.

    This is an integration test that validates the function works correctly
    with outputs from Cirq's DensityMatrixSimulator.
    """
    n_qubits = 3
    circuit, qubits = ghz_circuit(n_qubits)
    ideal_state = ideal_ghz_state(n_qubits)

    # Simulate with density matrix simulator (no noise - should give F=1)
    dm_simulator = cirq.DensityMatrixSimulator()
    result = dm_simulator.simulate(circuit, qubit_order=qubits)
    density_matrix = result.final_density_matrix

    fidelity = mixed_state_fidelity(ideal_state, density_matrix)

    assert fidelity > 0.99, f"Expected ~1.0, got {fidelity}"


def test_mixed_state_fidelity_noisy_circuit():
    """
    Test mixed_state_fidelity with noisy Cirq simulation.

    Adding depolarizing noise should reduce fidelity below 1.0.
    """
    n_qubits = 3
    circuit, qubits = ghz_circuit(n_qubits)
    ideal_state = ideal_ghz_state(n_qubits)

    # Add depolarizing noise to all qubits
    noisy_circuit = circuit.copy()
    for q in qubits:
        noisy_circuit.append(cirq.depolarize(p=0.1).on(q))

    # Simulate noisy circuit
    dm_simulator = cirq.DensityMatrixSimulator()
    result = dm_simulator.simulate(noisy_circuit, qubit_order=qubits)
    noisy_dm = result.final_density_matrix

    fidelity = mixed_state_fidelity(ideal_state, noisy_dm)

    # Fidelity should be reduced by noise but still positive
    assert 0.0 < fidelity < 0.99, f"Expected fidelity between 0 and 0.99, got {fidelity}"


def test_mixed_state_fidelity_dimension_mismatch():
    """
    Test that dimension mismatch raises ValueError.
    """
    pure_state = np.array([1, 0], dtype=complex)  # 2-dimensional
    density_matrix = np.eye(4) / 4  # 4x4 matrix

    with pytest.raises(ValueError, match="Dimension mismatch"):
        mixed_state_fidelity(pure_state, density_matrix)


def test_mixed_state_fidelity_non_square_matrix():
    """
    Test that non-square density matrix raises ValueError.
    """
    pure_state = np.array([1, 0, 0], dtype=complex)
    density_matrix = np.array([[1, 0], [0, 1], [0, 0]])  # 3x2 matrix

    with pytest.raises(ValueError, match="square matrix"):
        mixed_state_fidelity(pure_state, density_matrix)


def test_mixed_state_fidelity_consistency_with_inline_formula():
    """
    Test that mixed_state_fidelity gives the same result as inline computation.

    This validates that our function produces identical results to the
    formula used in saboteur_env.py and test_saboteur_efficacy.py.
    """
    n_qubits = 4
    circuit, qubits = ghz_circuit(n_qubits)
    ideal_state = ideal_ghz_state(n_qubits)

    # Add noise
    noisy_circuit = circuit.copy()
    for q in qubits:
        noisy_circuit.append(cirq.depolarize(p=0.05).on(q))

    # Get density matrix
    dm_simulator = cirq.DensityMatrixSimulator()
    result = dm_simulator.simulate(noisy_circuit, qubit_order=qubits)
    noisy_dm = result.final_density_matrix

    # Compute fidelity using our function
    fidelity_func = mixed_state_fidelity(ideal_state, noisy_dm)

    # Compute fidelity using inline formula from test_saboteur_efficacy.py
    fidelity_inline = np.real(
        np.dot(np.conj(ideal_state), np.dot(noisy_dm, ideal_state))
    )

    # They should be identical
    assert np.isclose(fidelity_func, fidelity_inline), (
        f"Function result {fidelity_func} != inline result {fidelity_inline}"
    )


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("Running mixed_state_fidelity tests...")
    test_mixed_state_fidelity_pure_state()
    test_mixed_state_fidelity_orthogonal_state()
    test_mixed_state_fidelity_maximally_mixed()
    test_mixed_state_fidelity_partial_noise()
    test_mixed_state_fidelity_superposition()
    test_mixed_state_fidelity_two_qubits()
    test_mixed_state_fidelity_with_cirq_simulation()
    test_mixed_state_fidelity_noisy_circuit()
    test_mixed_state_fidelity_dimension_mismatch()
    test_mixed_state_fidelity_non_square_matrix()
    test_mixed_state_fidelity_consistency_with_inline_formula()
    print("All tests passed!")
