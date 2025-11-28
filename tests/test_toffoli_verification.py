"""
Tests for full basis-sweep fidelity verification harness.

This test module validates the Toffoli gate synthesis verification functions:
- computational_basis_state: Creates basis state vectors
- toffoli_truth_table: Generates truth tables for n-controlled NOT gates
- full_basis_fidelity: Computes average fidelity over all basis inputs
- full_basis_fidelity_toffoli: Convenience wrapper for Toffoli family gates

These tests ensure:
1. Perfect implementations achieve fidelity = 1.0
2. Identity/wrong circuits have fidelity < 1.0
3. Noise reduces fidelity appropriately
4. Nontrivial circuits are only accepted if they correctly implement the gate
"""

import numpy as np
import pytest
import cirq

from src.utils.metrics import (
    computational_basis_state,
    toffoli_truth_table,
    full_basis_fidelity,
    full_basis_fidelity_toffoli,
)


class TestComputationalBasisState:
    """Tests for computational_basis_state function."""

    def test_basis_state_zero(self):
        """Test creation of |0> state (1 qubit)."""
        state = computational_basis_state(0, 1)
        expected = np.array([1, 0], dtype=np.complex128)
        np.testing.assert_array_equal(state, expected)

    def test_basis_state_one(self):
        """Test creation of |1> state (1 qubit)."""
        state = computational_basis_state(1, 1)
        expected = np.array([0, 1], dtype=np.complex128)
        np.testing.assert_array_equal(state, expected)

    def test_basis_state_two_qubits(self):
        """Test all basis states for 2 qubits."""
        # |00> = [1,0,0,0]
        np.testing.assert_array_equal(
            computational_basis_state(0, 2),
            np.array([1, 0, 0, 0], dtype=np.complex128)
        )
        # |01> = [0,1,0,0]
        np.testing.assert_array_equal(
            computational_basis_state(1, 2),
            np.array([0, 1, 0, 0], dtype=np.complex128)
        )
        # |10> = [0,0,1,0]
        np.testing.assert_array_equal(
            computational_basis_state(2, 2),
            np.array([0, 0, 1, 0], dtype=np.complex128)
        )
        # |11> = [0,0,0,1]
        np.testing.assert_array_equal(
            computational_basis_state(3, 2),
            np.array([0, 0, 0, 1], dtype=np.complex128)
        )

    def test_basis_state_three_qubits(self):
        """Test a few basis states for 3 qubits."""
        # |000> = [1,0,0,0,0,0,0,0]
        state_0 = computational_basis_state(0, 3)
        assert state_0[0] == 1.0
        assert np.sum(np.abs(state_0)) == 1.0

        # |111> = [0,0,0,0,0,0,0,1]
        state_7 = computational_basis_state(7, 3)
        assert state_7[7] == 1.0
        assert np.sum(np.abs(state_7)) == 1.0

    def test_basis_state_normalization(self):
        """Test that all basis states are normalized."""
        for n_qubits in [1, 2, 3, 4]:
            dim = 2 ** n_qubits
            for i in range(dim):
                state = computational_basis_state(i, n_qubits)
                norm = np.linalg.norm(state)
                assert np.isclose(norm, 1.0), f"State {i} with {n_qubits} qubits is not normalized"


class TestToffoliTruthTable:
    """Tests for toffoli_truth_table function."""

    def test_toffoli_ccnot_truth_table(self):
        """Test truth table for standard Toffoli (2 controls, CCNOT)."""
        truth_fn = toffoli_truth_table(n_controls=2)

        # Expected behavior: flip target (LSB) only when both controls (bits 2,1) are 1
        expected = {
            0b000: 0b000,  # |000> -> |000>
            0b001: 0b001,  # |001> -> |001>
            0b010: 0b010,  # |010> -> |010>
            0b011: 0b011,  # |011> -> |011>
            0b100: 0b100,  # |100> -> |100>
            0b101: 0b101,  # |101> -> |101>
            0b110: 0b111,  # |110> -> |111> (flip!)
            0b111: 0b110,  # |111> -> |110> (flip!)
        }
        for input_idx, expected_output in expected.items():
            assert truth_fn(input_idx) == expected_output, \
                f"Truth table wrong for input {input_idx:03b}"

    def test_toffoli_cccnot_truth_table(self):
        """Test truth table for 3-controlled NOT (CCCNOT)."""
        truth_fn = toffoli_truth_table(n_controls=3)

        # Only flip when all 3 controls are 1
        for i in range(16):
            if i in (0b1110, 0b1111):
                expected = i ^ 1
            else:
                expected = i
            assert truth_fn(i) == expected, \
                f"Truth table wrong for input {i:04b}: got {truth_fn(i):04b}, expected {expected:04b}"

    def test_toffoli_cnot_truth_table(self):
        """Test truth table for 1-controlled NOT (CNOT)."""
        truth_fn = toffoli_truth_table(n_controls=1)

        expected = {
            0b00: 0b00,  # |00> -> |00>
            0b01: 0b01,  # |01> -> |01>
            0b10: 0b11,  # |10> -> |11> (flip!)
            0b11: 0b10,  # |11> -> |10> (flip!)
        }
        for input_idx, expected_output in expected.items():
            assert truth_fn(input_idx) == expected_output

    def test_truth_table_matches_cirq_toffoli(self):
        """Test that truth table matches Cirq's Toffoli implementation."""
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(cirq.TOFFOLI(*qubits))
        unitary = cirq.unitary(circuit)

        truth_fn = toffoli_truth_table(n_controls=2)

        for i in range(8):
            # Find output state from unitary
            cirq_output = np.argmax(np.abs(unitary[i, :]))
            my_output = truth_fn(i)
            assert my_output == cirq_output, \
                f"Mismatch at input {i}: truth_fn={my_output}, cirq={cirq_output}"


class TestFullBasisFidelity:
    """Tests for full_basis_fidelity function."""

    def test_perfect_toffoli_fidelity(self):
        """Test that perfect Toffoli gate gives fidelity 1.0."""
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(cirq.TOFFOLI(*qubits))
        truth_fn = toffoli_truth_table(n_controls=2)

        fidelity = full_basis_fidelity(circuit, qubits, truth_fn)

        assert np.isclose(fidelity, 1.0), f"Perfect Toffoli should have fidelity 1.0, got {fidelity}"

    def test_identity_circuit_fidelity(self):
        """Test that identity circuit has fidelity < 1.0 for Toffoli."""
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit()  # Empty (identity)
        truth_fn = toffoli_truth_table(n_controls=2)

        fidelity = full_basis_fidelity(circuit, qubits, truth_fn)

        # Identity matches Toffoli on 6/8 inputs (those where controls aren't both 1)
        expected_fidelity = 6 / 8
        assert np.isclose(fidelity, expected_fidelity), \
            f"Identity circuit should have fidelity {expected_fidelity}, got {fidelity}"

    def test_wrong_circuit_rejected(self):
        """Test that incorrect circuits have fidelity < 1.0."""
        qubits = cirq.LineQubit.range(3)

        # Test various wrong circuits
        wrong_circuits = [
            cirq.Circuit(cirq.CNOT(qubits[0], qubits[2])),  # Only CNOT, not Toffoli
            cirq.Circuit(cirq.X(qubits[0])),  # Just X on control
            cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[2])),  # Random gates
        ]

        truth_fn = toffoli_truth_table(n_controls=2)

        for circuit in wrong_circuits:
            fidelity = full_basis_fidelity(circuit, qubits, truth_fn)
            assert fidelity < 1.0, f"Wrong circuit should have fidelity < 1.0, got {fidelity}"

    def test_fidelity_with_noise(self):
        """Test that noise reduces fidelity below 1.0."""
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(cirq.TOFFOLI(*qubits))
        truth_fn = toffoli_truth_table(n_controls=2)

        # Apply significant depolarizing noise
        noise_model = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.05))
        fidelity_noisy = full_basis_fidelity(circuit, qubits, truth_fn, noise_model=noise_model)

        assert fidelity_noisy < 1.0, "Noisy circuit should have fidelity < 1.0"
        assert fidelity_noisy > 0.5, "Fidelity shouldn't be too low with p=0.05 noise"


class TestFullBasisFidelityToffoli:
    """Tests for full_basis_fidelity_toffoli convenience function."""

    def test_ccnot_perfect(self):
        """Test CCNOT (2-controlled NOT) with perfect implementation."""
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(cirq.TOFFOLI(*qubits))

        fidelity = full_basis_fidelity_toffoli(circuit, qubits, n_controls=2)

        assert np.isclose(fidelity, 1.0)

    def test_cccnot_perfect(self):
        """Test CCCNOT (3-controlled NOT) with perfect implementation."""
        qubits = cirq.LineQubit.range(4)
        cccnot_gate = cirq.X.controlled(num_controls=3)
        circuit = cirq.Circuit(cccnot_gate(*qubits))

        fidelity = full_basis_fidelity_toffoli(circuit, qubits, n_controls=3)

        assert np.isclose(fidelity, 1.0)

    def test_cnot_perfect(self):
        """Test CNOT (1-controlled NOT) with perfect implementation."""
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.CNOT(*qubits))

        fidelity = full_basis_fidelity_toffoli(circuit, qubits, n_controls=1)

        assert np.isclose(fidelity, 1.0)

    def test_nontrivial_circuit_validation(self):
        """
        Regression test: Ensure nontrivial circuits are only accepted if they
        correctly implement the gate (fidelity = 1.0).

        This is the key requirement from the problem statement.
        """
        qubits = cirq.LineQubit.range(3)

        # A circuit that looks complex but is NOT a correct Toffoli
        nontrivial_wrong_circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.T(qubits[2]),
            cirq.H(qubits[0]),
        ])

        fidelity = full_basis_fidelity_toffoli(nontrivial_wrong_circuit, qubits, n_controls=2)

        # This circuit should NOT have fidelity 1.0
        assert fidelity < 1.0, \
            "Nontrivial incorrect circuit should not be accepted as valid Toffoli"

        # The correct Toffoli should have fidelity 1.0
        correct_circuit = cirq.Circuit(cirq.TOFFOLI(*qubits))
        fidelity_correct = full_basis_fidelity_toffoli(correct_circuit, qubits, n_controls=2)
        assert np.isclose(fidelity_correct, 1.0), \
            "Correct Toffoli should have fidelity 1.0"


class TestRegressionNoZeroInputOnly:
    """
    Regression tests to ensure evaluation uses full basis sweep,
    not just the |000...0> input.
    """

    def test_swap_detection(self):
        """
        Test that we can distinguish Toffoli from circuits that only
        differ on non-zero inputs.

        A circuit that is identity on |000> but wrong elsewhere should
        NOT have fidelity 1.0.
        """
        qubits = cirq.LineQubit.range(3)

        # Circuit that is identity on |000> but applies X to target always
        # (This would pass a |000>-only test but fail full basis sweep)
        always_flip_circuit = cirq.Circuit(cirq.X(qubits[2]))

        fidelity = full_basis_fidelity_toffoli(always_flip_circuit, qubits, n_controls=2)

        # This should NOT be accepted as Toffoli
        assert fidelity < 1.0, \
            "Circuit that only matches on some inputs should not have fidelity 1.0"

    def test_partial_implementation(self):
        """
        Test detection of circuits that partially implement Toffoli.

        A circuit that flips the target on more states than just |110> and |111>
        should be rejected.
        """
        qubits = cirq.LineQubit.range(3)

        # Controlled-X with only one control (flips when qubit 0 is 1, ignoring qubit 1)
        # This flips on |100>, |101>, |110>, |111> - more than just |110>, |111>
        partial_circuit = cirq.Circuit(
            cirq.CNOT(qubits[0], qubits[2])  # Control on q0, target on q2
        )

        fidelity = full_basis_fidelity_toffoli(partial_circuit, qubits, n_controls=2)

        # Should not be 1.0 because it flips on wrong states
        assert fidelity < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
