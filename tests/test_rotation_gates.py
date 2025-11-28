"""
Tests for parameterized rotation gate support in quantum architecture search environments.

These tests verify:
- Rotation gate creation and identification
- Action space expansion with rotation gates
- Circuit serialization with rotation parameters
- Gate counting and metrics for rotation gates
- GHZ state recovery with rotation gates available
"""

import numpy as np
import pytest
import cirq
import tempfile
import os
import json

from src.qas_gym.envs import QuantumArchSearchEnv, ArchitectEnv
from src.qas_gym.utils import (
    get_default_gates, get_gates_by_name, create_rotation_gate,
    is_rotation_gate, get_rotation_gate_info, serialize_circuit_with_rotations,
    count_rotation_gates, get_ghz_state
)
from src.utils.metrics import count_rotation_gates as metrics_count_rotation_gates
from src.utils.metrics import get_rotation_angles, evaluate_circuit


class TestRotationGateUtilities:
    """Tests for rotation gate utility functions."""

    def test_create_rotation_gate_rx(self):
        """Test Rx gate creation."""
        qubit = cirq.LineQubit(0)
        angle = np.pi / 4
        gate = create_rotation_gate('Rx', qubit, angle)
        
        assert gate.qubits == (qubit,)
        assert isinstance(gate.gate, cirq.Rx)
        # cirq uses exponent = angle/pi
        assert abs(gate.gate.exponent - 0.25) < 1e-10

    def test_create_rotation_gate_ry(self):
        """Test Ry gate creation."""
        qubit = cirq.LineQubit(1)
        angle = np.pi / 2
        gate = create_rotation_gate('Ry', qubit, angle)
        
        assert gate.qubits == (qubit,)
        assert isinstance(gate.gate, cirq.Ry)
        assert abs(gate.gate.exponent - 0.5) < 1e-10

    def test_create_rotation_gate_rz(self):
        """Test Rz gate creation."""
        qubit = cirq.LineQubit(2)
        angle = np.pi
        gate = create_rotation_gate('Rz', qubit, angle)
        
        assert gate.qubits == (qubit,)
        assert isinstance(gate.gate, cirq.Rz)
        assert abs(gate.gate.exponent - 1.0) < 1e-10

    def test_create_rotation_gate_invalid_type(self):
        """Test that invalid gate type raises error."""
        qubit = cirq.LineQubit(0)
        with pytest.raises(ValueError):
            create_rotation_gate('InvalidGate', qubit, np.pi)

    def test_is_rotation_gate(self):
        """Test rotation gate identification."""
        qubit = cirq.LineQubit(0)
        
        # Rotation gates
        assert is_rotation_gate(cirq.rx(np.pi).on(qubit).gate) is True
        assert is_rotation_gate(cirq.ry(np.pi).on(qubit).gate) is True
        assert is_rotation_gate(cirq.rz(np.pi).on(qubit).gate) is True
        
        # Non-rotation gates
        assert is_rotation_gate(cirq.X.on(qubit).gate) is False
        assert is_rotation_gate(cirq.H.on(qubit).gate) is False
        assert is_rotation_gate(cirq.T.on(qubit).gate) is False

    def test_get_rotation_gate_info(self):
        """Test extraction of rotation gate parameters."""
        qubit = cirq.LineQubit(0)
        angle = np.pi / 3
        
        rx_op = cirq.rx(angle).on(qubit)
        info = get_rotation_gate_info(rx_op)
        
        assert info is not None
        assert info['type'] == 'Rx'
        assert info['qubit'] == qubit
        assert abs(info['angle'] - angle) < 1e-10

    def test_get_rotation_gate_info_non_rotation(self):
        """Test that non-rotation gates return None."""
        qubit = cirq.LineQubit(0)
        x_op = cirq.X.on(qubit)
        
        info = get_rotation_gate_info(x_op)
        assert info is None


class TestRotationGatesInActionSpace:
    """Tests for rotation gates in environment action spaces."""

    def test_get_gates_by_name_without_rotations(self):
        """Test that default gates don't include rotations."""
        qubits = cirq.LineQubit.range(2)
        gates = get_gates_by_name(qubits, ['X', 'Y', 'H'], include_rotations=False)
        
        # Should have X, Y, H on each qubit + CNOTs
        # 3 gates * 2 qubits + 2 CNOTs = 8
        assert len(gates) == 8
        
        # No rotation gates
        for gate_op in gates:
            assert not is_rotation_gate(gate_op.gate)

    def test_get_gates_by_name_with_rotations(self):
        """Test that include_rotations adds Rx, Ry, Rz gates."""
        qubits = cirq.LineQubit.range(2)
        gates = get_gates_by_name(qubits, ['X', 'Y', 'H'], include_rotations=True)
        
        # Should have X, Y, H + Rx, Ry, Rz on each qubit + CNOTs
        # (3 + 3) gates * 2 qubits + 2 CNOTs = 14
        assert len(gates) == 14
        
        # Count rotation gates
        rotation_count = sum(1 for g in gates if is_rotation_gate(g.gate))
        assert rotation_count == 6  # 3 rotation types * 2 qubits

    def test_get_default_gates_without_rotations(self):
        """Test default gates without rotations."""
        qubits = cirq.LineQubit.range(2)
        gates = get_default_gates(qubits, include_rotations=False)
        
        rotation_count = sum(1 for g in gates if is_rotation_gate(g.gate))
        assert rotation_count == 0

    def test_get_default_gates_with_rotations(self):
        """Test default gates with rotations included."""
        qubits = cirq.LineQubit.range(2)
        gates = get_default_gates(qubits, include_rotations=True)
        
        rotation_count = sum(1 for g in gates if is_rotation_gate(g.gate))
        assert rotation_count == 6  # 3 rotation types * 2 qubits


class TestQuantumArchSearchEnvWithRotations:
    """Tests for QuantumArchSearchEnv with rotation gates."""

    def test_env_creation_with_rotations(self):
        """Test environment creation with rotation gates enabled."""
        target = get_ghz_state(2)
        env = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=10,
            include_rotations=True
        )
        
        assert env.include_rotations is True
        # Check that rotation gates are in action space
        rotation_count = sum(1 for g in env.action_gates if is_rotation_gate(g.gate))
        assert rotation_count > 0

    def test_env_creation_without_rotations(self):
        """Test environment creation without rotation gates (backward compatible)."""
        target = get_ghz_state(2)
        env = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=10,
            include_rotations=False
        )
        
        assert env.include_rotations is False
        # No rotation gates in action space
        rotation_count = sum(1 for g in env.action_gates if is_rotation_gate(g.gate))
        assert rotation_count == 0

    def test_env_default_no_rotations(self):
        """Test that rotations are disabled by default for backward compatibility."""
        target = get_ghz_state(2)
        env = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=10
        )
        
        # Default should be no rotations for backward compatibility
        assert env.include_rotations is False

    def test_step_with_rotation_gates(self):
        """Test stepping with rotation gate actions."""
        target = get_ghz_state(2)
        env = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=10,
            include_rotations=True
        )
        
        env.reset(seed=42)
        
        # Find a rotation gate action
        rotation_action = None
        for i, gate_op in enumerate(env.action_gates):
            if is_rotation_gate(gate_op.gate):
                rotation_action = i
                break
        
        assert rotation_action is not None
        
        # Take the rotation action
        obs, reward, terminated, truncated, info = env.step(rotation_action)
        
        # Verify rotation gate was added
        assert 'rotation_counts' in info
        assert info['rotation_counts']['total_rotations'] >= 1

    def test_rotation_angles_tracked(self):
        """Test that rotation angles are properly tracked."""
        target = get_ghz_state(2)
        env = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=10,
            include_rotations=True,
            default_rotation_angle=np.pi/4
        )
        
        env.reset(seed=42)
        
        # Find and execute a rotation action
        for i, gate_op in enumerate(env.action_gates):
            if is_rotation_gate(gate_op.gate):
                env.step(i)
                break
        
        # Check rotation angles are tracked
        angles = env.get_rotation_angles()
        assert len(angles) >= 1
        assert 0 in angles  # First gate index

    def test_set_rotation_angle(self):
        """Test setting rotation angle for a gate."""
        target = get_ghz_state(2)
        env = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=10,
            include_rotations=True,
            default_rotation_angle=np.pi/4
        )
        
        env.reset(seed=42)
        
        # Add a rotation gate
        for i, gate_op in enumerate(env.action_gates):
            if is_rotation_gate(gate_op.gate):
                env.step(i)
                break
        
        # Set new angle
        new_angle = np.pi / 2
        env.set_rotation_angle(0, new_angle)
        
        # Verify angle was updated
        angles = env.get_rotation_angles()
        assert abs(angles[0] - new_angle) < 1e-10

    def test_get_circuit_info(self):
        """Test circuit info retrieval with rotation gates."""
        target = get_ghz_state(2)
        env = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=10,
            include_rotations=True
        )
        
        env.reset(seed=42)
        
        # Add some gates
        for i, gate_op in enumerate(env.action_gates):
            if is_rotation_gate(gate_op.gate):
                env.step(i)
                if env.get_circuit_info()['rotation_counts']['total_rotations'] >= 2:
                    break
        
        info = env.get_circuit_info()
        
        assert 'total_gates' in info
        assert 'rotation_counts' in info
        assert 'cnot_count' in info
        assert 'rotation_angles' in info
        assert 'serialized' in info


class TestCircuitSerialization:
    """Tests for circuit serialization with rotation gates."""

    def test_serialize_circuit_with_rotations(self):
        """Test serialization includes rotation angles."""
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.rx(np.pi/4).on(qubits[0]),
            cirq.ry(np.pi/3).on(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.rz(np.pi/2).on(qubits[0])
        ])
        
        serialized = serialize_circuit_with_rotations(circuit)
        
        assert len(serialized) == 4
        
        # Check rotation gate serialization
        assert serialized[0]['gate_type'] == 'Rx'
        assert 'angle' in serialized[0]
        assert abs(serialized[0]['angle'] - np.pi/4) < 1e-10
        
        assert serialized[1]['gate_type'] == 'Ry'
        assert abs(serialized[1]['angle'] - np.pi/3) < 1e-10
        
        assert serialized[2]['gate_type'] == 'CNOT'
        
        assert serialized[3]['gate_type'] == 'Rz'
        assert abs(serialized[3]['angle'] - np.pi/2) < 1e-10

    def test_count_rotation_gates_circuit(self):
        """Test rotation gate counting in circuits."""
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.rx(np.pi).on(qubits[0]),
            cirq.rx(np.pi/2).on(qubits[1]),
            cirq.ry(np.pi).on(qubits[0]),
            cirq.rz(np.pi).on(qubits[1]),
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1])
        ])
        
        counts = count_rotation_gates(circuit)
        
        assert counts['Rx'] == 2
        assert counts['Ry'] == 1
        assert counts['Rz'] == 1
        assert counts['total_rotations'] == 4


class TestMetricsWithRotations:
    """Tests for metrics module with rotation gates."""

    def test_metrics_count_rotation_gates(self):
        """Test metrics module rotation gate counting."""
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.rx(np.pi).on(qubits[0]),
            cirq.ry(np.pi/2).on(qubits[1]),
            cirq.rz(np.pi/4).on(qubits[0])
        ])
        
        counts = metrics_count_rotation_gates(circuit)
        
        assert counts['rx_count'] == 1
        assert counts['ry_count'] == 1
        assert counts['rz_count'] == 1
        assert counts['total_rotations'] == 3

    def test_get_rotation_angles_metrics(self):
        """Test rotation angle extraction from metrics."""
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.rx(np.pi/4).on(qubits[0]),
            cirq.H(qubits[1]),
            cirq.ry(np.pi/3).on(qubits[0])
        ])
        
        angles = get_rotation_angles(circuit)
        
        assert len(angles) == 2  # Only rotation gates
        assert angles[0]['gate_type'] == 'Rx'
        assert abs(angles[0]['angle'] - np.pi/4) < 1e-10
        assert angles[1]['gate_type'] == 'Ry'
        assert abs(angles[1]['angle'] - np.pi/3) < 1e-10

    def test_evaluate_circuit_with_rotations(self):
        """Test comprehensive circuit evaluation with rotations."""
        qubits = cirq.LineQubit.range(2)
        target = np.array([1, 0, 0, 0], dtype=complex)  # |00>
        
        circuit = cirq.Circuit([
            cirq.rx(0).on(qubits[0]),  # Identity rotation
            cirq.ry(0).on(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1])
        ])
        
        metrics = evaluate_circuit(circuit, target, qubits)
        
        assert 'fidelity' in metrics
        assert 'rotation_counts' in metrics
        assert 'rotation_angles' in metrics
        assert metrics['rotation_counts']['total_rotations'] == 2


class TestGHZStateRecoveryWithRotations:
    """Tests for GHZ state recovery using rotation gates."""

    def test_ghz_state_achievable_with_rotations(self):
        """Test that GHZ state can be achieved with rotation gates available.
        
        The standard GHZ circuit (H + CNOTs) should still work when
        rotation gates are available in the action space.
        """
        target = get_ghz_state(3)
        env = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=20,
            include_rotations=True
        )
        
        env.reset(seed=42)
        
        # Find H gate action (first qubit)
        h_action = None
        for i, gate_op in enumerate(env.action_gates):
            if isinstance(gate_op.gate, cirq.HPowGate) and gate_op.qubits[0] == env.qubits[0]:
                h_action = i
                break
        
        assert h_action is not None, "H gate on qubit 0 not found in action space"
        
        # Apply H gate on first qubit
        obs, reward, terminated, truncated, info = env.step(h_action)
        
        # Find and apply CNOTs for GHZ state
        for target_qubit_idx in range(1, 3):
            for i, gate_op in enumerate(env.action_gates):
                if (isinstance(gate_op.gate, cirq.CNotPowGate) and 
                    gate_op.qubits[0] == env.qubits[0] and 
                    gate_op.qubits[1] == env.qubits[target_qubit_idx]):
                    obs, reward, terminated, truncated, info = env.step(i)
                    break
        
        # Verify high fidelity was achieved
        assert info['fidelity'] > 0.99, f"GHZ fidelity {info['fidelity']} too low"

    def test_rotations_dont_break_existing_functionality(self):
        """Test that enabling rotations doesn't break existing circuit building."""
        target = get_ghz_state(2)
        
        # Create environment with rotations
        env_with_rot = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=10,
            include_rotations=True
        )
        
        # Create environment without rotations
        env_without_rot = QuantumArchSearchEnv(
            target=target,
            fidelity_threshold=0.99,
            reward_penalty=0.01,
            max_timesteps=10,
            include_rotations=False
        )
        
        # Both should be able to reset and step
        env_with_rot.reset(seed=42)
        env_without_rot.reset(seed=42)
        
        # Find a common action (e.g., H gate)
        h_action_with = None
        h_action_without = None
        
        for i, gate_op in enumerate(env_with_rot.action_gates):
            if isinstance(gate_op.gate, cirq.HPowGate):
                h_action_with = i
                break
        
        for i, gate_op in enumerate(env_without_rot.action_gates):
            if isinstance(gate_op.gate, cirq.HPowGate):
                h_action_without = i
                break
        
        # Both should have H gate
        assert h_action_with is not None
        assert h_action_without is not None
        
        # Step with H gate
        obs1, _, _, _, info1 = env_with_rot.step(h_action_with)
        obs2, _, _, _, info2 = env_without_rot.step(h_action_without)
        
        # Both should produce valid results
        assert 'fidelity' in info1
        assert 'fidelity' in info2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
