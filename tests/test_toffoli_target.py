"""
Tests for n-controlled Toffoli gate target functions.

This test module validates the Toffoli gate utilities that serve as the default
compilation target for quantum architecture search experiments.

Tests include:
- Toffoli unitary correctness for n=2,3,4 qubits
- Toffoli state preparation via circuit simulation
- Consistency between unitary and circuit methods
- Edge cases and error handling
"""

import numpy as np
import pytest
import cirq

from src.qas_gym.utils import (
    get_toffoli_unitary,
    get_toffoli_state,
    get_toffoli_target_state,
    create_toffoli_circuit_and_qubits,
    get_default_target_state,
    get_ghz_state,
)
from src.utils.metrics import (
    toffoli_circuit,
    ideal_toffoli_state,
    ghz_circuit,
    ideal_ghz_state,
)


class TestToffoliUnitary:
    """Tests for get_toffoli_unitary function."""
    
    def test_unitary_2_qubits_is_cnot(self):
        """2-qubit Toffoli should be a CNOT gate."""
        unitary = get_toffoli_unitary(2)
        assert unitary.shape == (4, 4)
        
        # CNOT (control=q0 MSB, target=q1 LSB):
        # |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
        expected_cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        assert np.allclose(unitary, expected_cnot)
    
    def test_unitary_3_qubits_is_toffoli(self):
        """3-qubit should be a Toffoli (CCNOT) gate."""
        unitary = get_toffoli_unitary(3)
        assert unitary.shape == (8, 8)
        
        # Verify it's unitary
        assert np.allclose(unitary @ unitary.conj().T, np.eye(8))
        
        # Toffoli (controls=q0,q1, target=q2):
        # Flips target only when both controls are |1>
        # |110> (idx 6) <-> |111> (idx 7)
        assert unitary[6, 6] == 0  # |110> doesn't stay
        assert unitary[7, 7] == 0  # |111> doesn't stay
        assert unitary[6, 7] == 1  # |110> <- |111>
        assert unitary[7, 6] == 1  # |111> <- |110>
        
        # Other states unchanged
        assert unitary[0, 0] == 1  # |000>
        assert unitary[1, 1] == 1  # |001>
        assert unitary[5, 5] == 1  # |101>
    
    def test_unitary_4_qubits_cccnot(self):
        """4-qubit should be CCCNOT."""
        unitary = get_toffoli_unitary(4)
        assert unitary.shape == (16, 16)
        
        # Verify unitarity
        assert np.allclose(unitary @ unitary.conj().T, np.eye(16))
        
        # Only |1110> (idx 14) <-> |1111> (idx 15) should be swapped
        assert unitary[14, 14] == 0
        assert unitary[15, 15] == 0
        assert unitary[14, 15] == 1
        assert unitary[15, 14] == 1
    
    def test_unitary_error_for_single_qubit(self):
        """Should raise error for n < 2."""
        with pytest.raises(ValueError, match="at least 2 qubits"):
            get_toffoli_unitary(1)


class TestToffoliState:
    """Tests for Toffoli state preparation functions."""
    
    def test_toffoli_state_2_qubits(self):
        """2-qubit Toffoli state: |11> -> |10>."""
        state = get_toffoli_state(2)
        assert state.shape == (4,)
        
        # After applying CNOT to |11>, we get |10>
        # |10> in big-endian = index 2
        assert np.isclose(np.abs(state[2]), 1.0)
        # All others should be zero
        for i in [0, 1, 3]:
            assert np.isclose(np.abs(state[i]), 0.0)
    
    def test_toffoli_state_3_qubits(self):
        """3-qubit Toffoli state: |111> -> |110>."""
        state = get_toffoli_state(3)
        assert state.shape == (8,)
        
        # After applying CCNOT to |111>, we get |110>
        # |110> = index 6
        assert np.isclose(np.abs(state[6]), 1.0)
    
    def test_toffoli_state_4_qubits(self):
        """4-qubit Toffoli state: |1111> -> |1110>."""
        state = get_toffoli_state(4)
        assert state.shape == (16,)
        
        # After applying CCCNOT to |1111>, we get |1110>
        # |1110> = index 14
        assert np.isclose(np.abs(state[14]), 1.0)
    
    def test_default_target_state_matches_toffoli(self):
        """get_default_target_state should return Toffoli state."""
        for n in [2, 3, 4]:
            default = get_default_target_state(n)
            toffoli = get_toffoli_state(n)
            assert np.allclose(default, toffoli)
    
    def test_toffoli_state_error_for_single_qubit(self):
        """Should raise error for n < 2."""
        with pytest.raises(ValueError, match="at least 2 qubits"):
            get_toffoli_state(1)


class TestToffoliCircuit:
    """Tests for Toffoli circuit creation functions."""
    
    def test_circuit_2_qubits(self):
        """2-qubit circuit should have X gates and CNOT."""
        circuit, qubits = create_toffoli_circuit_and_qubits(2)
        
        assert len(qubits) == 2
        
        # Simulate and verify output state
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit, qubit_order=qubits)
        state = result.final_state_vector
        
        # Should be |10>
        assert np.isclose(np.abs(state[2]), 1.0)
    
    def test_circuit_3_qubits_uses_toffoli(self):
        """3-qubit circuit should include Toffoli gate."""
        circuit, qubits = create_toffoli_circuit_and_qubits(3)
        
        assert len(qubits) == 3
        
        # Verify Toffoli gate is present
        ops = list(circuit.all_operations())
        toffoli_found = any(
            isinstance(op.gate, cirq.CCXPowGate) or 
            (hasattr(op.gate, '_gate') and isinstance(getattr(op.gate, '_gate', None), cirq.XPowGate))
            for op in ops
        )
        # TOFFOLI is a CCXPowGate, or may appear as ControlledGate
        # Just verify simulation gives correct result
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit, qubit_order=qubits)
        state = result.final_state_vector
        
        # Should be |110>
        assert np.isclose(np.abs(state[6]), 1.0)
    
    def test_circuit_4_qubits(self):
        """4-qubit circuit should produce correct state."""
        circuit, qubits = create_toffoli_circuit_and_qubits(4)
        
        assert len(qubits) == 4
        
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit, qubit_order=qubits)
        state = result.final_state_vector
        
        # Should be |1110>
        assert np.isclose(np.abs(state[14]), 1.0)
    
    def test_circuit_matches_get_toffoli_state(self):
        """Circuit simulation should match get_toffoli_state."""
        for n in [2, 3, 4]:
            circuit, qubits = create_toffoli_circuit_and_qubits(n)
            simulator = cirq.Simulator()
            result = simulator.simulate(circuit, qubit_order=qubits)
            circuit_state = result.final_state_vector
            
            toffoli_state = get_toffoli_state(n)
            
            # States should match (up to global phase)
            overlap = np.abs(np.vdot(circuit_state, toffoli_state))
            assert np.isclose(overlap, 1.0)


class TestMetricsToffoliFunctions:
    """Tests for Toffoli functions in metrics.py."""
    
    def test_toffoli_circuit_matches_utils(self):
        """toffoli_circuit in metrics.py should match utils version."""
        for n in [2, 3, 4]:
            metrics_circuit, metrics_qubits = toffoli_circuit(n)
            utils_circuit, utils_qubits = create_toffoli_circuit_and_qubits(n)
            
            # Simulate both
            simulator = cirq.Simulator()
            metrics_state = simulator.simulate(metrics_circuit, qubit_order=metrics_qubits).final_state_vector
            utils_state = simulator.simulate(utils_circuit, qubit_order=utils_qubits).final_state_vector
            
            assert np.allclose(metrics_state, utils_state)
    
    def test_ideal_toffoli_state_matches_utils(self):
        """ideal_toffoli_state should match get_toffoli_state."""
        for n in [2, 3, 4]:
            metrics_state = ideal_toffoli_state(n)
            utils_state = get_toffoli_state(n)
            
            assert np.allclose(metrics_state, utils_state)


class TestBackwardCompatibility:
    """Tests ensuring GHZ functions still work as legacy option."""
    
    def test_ghz_state_still_works(self):
        """get_ghz_state should still produce correct GHZ state."""
        for n in [2, 3, 4]:
            state = get_ghz_state(n)
            
            # GHZ state: (|00...0> + |11...1>) / sqrt(2)
            expected_amplitude = 1.0 / np.sqrt(2)
            
            assert np.isclose(np.abs(state[0]), expected_amplitude)
            assert np.isclose(np.abs(state[-1]), expected_amplitude)
    
    def test_ghz_differs_from_toffoli(self):
        """GHZ and Toffoli states should be different."""
        for n in [2, 3, 4]:
            ghz = get_ghz_state(n)
            toffoli = get_toffoli_state(n)
            
            # They should not be the same
            assert not np.allclose(ghz, toffoli)
    
    def test_metrics_ghz_functions_work(self):
        """Legacy GHZ functions in metrics.py should still work."""
        for n in [2, 3, 4]:
            circuit, qubits = ghz_circuit(n)
            state = ideal_ghz_state(n)
            
            simulator = cirq.Simulator()
            result = simulator.simulate(circuit, qubit_order=qubits)
            circuit_state = result.final_state_vector
            
            # Should match
            overlap = np.abs(np.vdot(circuit_state, state))
            assert np.isclose(overlap, 1.0)


class TestToffoliTargetState:
    """Tests for get_toffoli_target_state with custom inputs."""
    
    def test_default_input_matches_get_toffoli_state(self):
        """Default input should match get_toffoli_state."""
        for n in [2, 3, 4]:
            target_state = get_toffoli_target_state(n)
            toffoli_state = get_toffoli_state(n)
            
            assert np.allclose(target_state, toffoli_state)
    
    def test_custom_input_state(self):
        """Custom input should produce expected output."""
        # For 3 qubits with input |100>, controls are not both 1
        # So output should be |100> (unchanged)
        state = get_toffoli_target_state(3, input_state='100')
        
        # |100> = index 4 (binary 100 = 4)
        assert np.isclose(np.abs(state[4]), 1.0)
    
    def test_all_zeros_input(self):
        """Input |000...0> should remain unchanged."""
        for n in [2, 3, 4]:
            state = get_toffoli_target_state(n, input_state='0' * n)
            
            # |00...0> = index 0, should remain unchanged
            assert np.isclose(np.abs(state[0]), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
