"""
Integration tests for VQEArchitectEnv.

These tests verify the complete VQE workflow including:
- Environment reset and step mechanics
- Action space encoding and decoding
- Classical optimization of rotation parameters
- Reward computation based on optimized energy
- Logging and circuit saving functionality
- Edge cases and sanity checks

See ExpPlan.md, Part 7.2 (VQE physics check) for background.
"""

import numpy as np
import pytest
import tempfile
import os
import json

from src.qas_gym.envs import VQEArchitectEnv


# Constants
CHEMICAL_ACCURACY = 0.0016  # 1.6 mHa in Hartree


class TestVQEEnvBasics:
    """Basic tests for VQEArchitectEnv initialization and spaces."""

    def test_h2_env_creation(self):
        """Test H2 environment creation."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        assert env.n_qubits == 2
        assert env.molecule == "H2"
        assert env.max_timesteps == 10

    def test_h4_env_creation(self):
        """Test H4 environment creation."""
        env = VQEArchitectEnv(molecule="H4", max_timesteps=15)
        assert env.n_qubits == 4
        assert env.molecule == "H4"
        assert env.max_timesteps == 15

    def test_action_space_h2(self):
        """Test action space for H2 (2 qubits)."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        # 2 qubits: 3*2 = 6 rotation actions + 2*1 = 2 CNOTs + 1 DONE = 9
        assert env.action_space.n == 9
        assert env.n_rotation_actions == 6
        assert env.n_cnot_actions == 2

    def test_action_space_h4(self):
        """Test action space for H4 (4 qubits)."""
        env = VQEArchitectEnv(molecule="H4", max_timesteps=15)
        # 4 qubits: 3*4 = 12 rotation actions + 4*3 = 12 CNOTs + 1 DONE = 25
        assert env.action_space.n == 25
        assert env.n_rotation_actions == 12
        assert env.n_cnot_actions == 12

    def test_observation_space(self):
        """Test observation space shape."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_unsupported_molecule(self):
        """Test that unsupported molecules raise an error."""
        with pytest.raises(NotImplementedError):
            VQEArchitectEnv(molecule="H6", max_timesteps=10)


class TestVQEEnvReset:
    """Tests for environment reset functionality."""

    def test_reset_returns_observation_and_info(self):
        """Test that reset returns proper observation and info."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        obs, info = env.reset(seed=42)

        assert obs is not None
        assert isinstance(info, dict)
        assert 'initial_energy' in info
        assert 'n_gates' in info
        assert info['n_gates'] == 0

    def test_reset_clears_circuit(self):
        """Test that reset clears the circuit."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        # Add some gates
        env.step(0)  # Rx on qubit 0
        env.step(1)  # Rx on qubit 1
        assert len(env.circuit_gates) == 2

        # Reset should clear
        env.reset()
        assert len(env.circuit_gates) == 0

    def test_reset_with_seed_reproducibility(self):
        """Test that reset with same seed gives reproducible results."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)

        obs1, _ = env.reset(seed=42)
        env.step(2)  # Add Ry gate (has random initial angle)
        angle1 = env.rotation_params.get(0)

        obs2, _ = env.reset(seed=42)
        env.step(2)  # Same action
        angle2 = env.rotation_params.get(0)

        np.testing.assert_array_equal(obs1, obs2)
        assert angle1 == angle2


class TestVQEEnvStep:
    """Tests for environment step functionality."""

    def test_step_adds_gate(self):
        """Test that step adds gates to the circuit."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(0)  # Rx on qubit 0
        assert info['n_gates'] == 1
        assert env.circuit_gates[0]['type'] == 'Rx'
        assert env.circuit_gates[0]['qubit'] == 0

    def test_step_rotation_gate_types(self):
        """Test that all rotation gate types are correctly decoded."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        # For 2 qubits: actions 0-1 are Rx, 2-3 are Ry, 4-5 are Rz
        env.step(0)  # Rx(q0)
        env.step(3)  # Ry(q1)
        env.step(4)  # Rz(q0)

        assert env.circuit_gates[0]['type'] == 'Rx'
        assert env.circuit_gates[0]['qubit'] == 0
        assert env.circuit_gates[1]['type'] == 'Ry'
        assert env.circuit_gates[1]['qubit'] == 1
        assert env.circuit_gates[2]['type'] == 'Rz'
        assert env.circuit_gates[2]['qubit'] == 0

    def test_step_cnot_gates(self):
        """Test that CNOT gates are correctly decoded."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        # For 2 qubits: CNOT actions start at 6
        # Action 6: CNOT(0, 1), Action 7: CNOT(1, 0)
        env.step(6)  # CNOT(0, 1)
        env.step(7)  # CNOT(1, 0)

        assert env.circuit_gates[0]['type'] == 'CNOT'
        assert env.circuit_gates[0]['control'] == 0
        assert env.circuit_gates[0]['target'] == 1
        assert env.circuit_gates[1]['type'] == 'CNOT'
        assert env.circuit_gates[1]['control'] == 1
        assert env.circuit_gates[1]['target'] == 0

    def test_done_action_terminates(self):
        """Test that DONE action terminates episode."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        env.step(0)  # Add a gate first
        obs, reward, terminated, truncated, info = env.step(8)  # DONE action

        assert terminated is True
        assert truncated is False

    def test_max_timesteps_terminates(self):
        """Test that reaching max_timesteps terminates episode."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=3)
        env.reset(seed=42)

        env.step(0)  # Gate 1
        env.step(1)  # Gate 2
        obs, reward, terminated, truncated, info = env.step(2)  # Gate 3

        assert terminated is True


class TestVQEEnvOptimization:
    """Tests for parameter optimization functionality."""

    def test_optimization_improves_energy(self):
        """Test that optimization improves energy."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        # Build a simple ansatz
        env.step(2)  # Ry(q0)
        env.step(3)  # Ry(q1)
        env.step(6)  # CNOT(0, 1)
        obs, reward, terminated, truncated, info = env.step(8)  # DONE

        # Optimized energy should be better than or equal to initial
        assert info['optimized_energy'] <= info['initial_energy']

    def test_optimization_reaches_chemical_accuracy_h2(self):
        """Test that a good ansatz can reach chemical accuracy for H2."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        # Build ansatz that can reach ground state
        env.step(2)  # Ry(q0)
        env.step(3)  # Ry(q1)
        env.step(6)  # CNOT(0, 1)
        obs, reward, terminated, truncated, info = env.step(8)  # DONE

        # Check chemical accuracy
        energy_error = abs(info['optimized_energy'] - env.fci_energy)
        assert energy_error < CHEMICAL_ACCURACY, \
            f"Energy error {energy_error*1000:.2f} mHa > {CHEMICAL_ACCURACY*1000:.1f} mHa"

    def test_optimization_with_no_rotation_gates(self):
        """Test optimization when only CNOT gates are present."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        env.step(6)  # CNOT(0, 1)
        obs, reward, terminated, truncated, info = env.step(8)  # DONE

        # Should complete without error
        assert 'optimized_energy' in info
        # Energy should match initial since no parameters to optimize
        assert info['initial_energy'] == info['optimized_energy']

    def test_optimization_with_empty_circuit(self):
        """Test optimization with empty circuit (immediate DONE)."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(8)  # Immediate DONE

        # Should return HF energy (|00> state)
        assert abs(info['optimized_energy'] - env.reference_energy) < 0.01


class TestVQEEnvReward:
    """Tests for reward computation."""

    def test_reward_is_negative_energy_error(self):
        """Test that reward is based on energy error."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10, cnot_penalty=0.0)
        env.reset(seed=42)

        env.step(2)  # Ry(q0)
        env.step(3)  # Ry(q1)
        env.step(6)  # CNOT(0, 1)
        obs, reward, terminated, truncated, info = env.step(8)  # DONE

        # Reward should be -(optimized_energy - fci_energy)
        expected_reward = -(info['optimized_energy'] - env.fci_energy)
        assert abs(reward - expected_reward) < 1e-6

    def test_cnot_penalty_reduces_reward(self):
        """Test that CNOT penalty reduces reward."""
        penalty = 0.01

        env = VQEArchitectEnv(molecule="H2", max_timesteps=10, cnot_penalty=penalty)
        env.reset(seed=42)

        env.step(6)  # CNOT(0, 1)
        obs, reward, terminated, truncated, info = env.step(8)  # DONE

        # Reward should include CNOT penalty
        energy_reward = -(info['optimized_energy'] - env.fci_energy)
        expected_reward = energy_reward - penalty * 1  # 1 CNOT
        assert abs(reward - expected_reward) < 1e-6


class TestVQEEnvLogging:
    """Tests for logging functionality."""

    def test_logging_creates_files(self):
        """Test that logging creates episode log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = VQEArchitectEnv(molecule="H2", max_timesteps=10, log_dir=tmpdir)
            env.reset(seed=42)

            env.step(2)  # Ry(q0)
            env.step(6)  # CNOT(0, 1)
            env.step(8)  # DONE

            # Check log file exists
            log_file = os.path.join(tmpdir, 'episode_logs.json')
            assert os.path.exists(log_file)

            # Check log content
            with open(log_file) as f:
                logs = json.load(f)
            assert len(logs) == 1
            assert logs[0]['n_gates'] == 2
            assert logs[0]['n_cnots'] == 1

    def test_logging_contains_required_fields(self):
        """Test that logs contain all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = VQEArchitectEnv(molecule="H2", max_timesteps=10, log_dir=tmpdir)
            env.reset(seed=42)

            env.step(2)  # Ry(q0)
            env.step(8)  # DONE

            log_file = os.path.join(tmpdir, 'episode_logs.json')
            with open(log_file) as f:
                logs = json.load(f)

            required_fields = [
                'timestamp', 'episode', 'molecule', 'bond_distance',
                'circuit_architecture', 'n_gates', 'n_cnots',
                'initial_params', 'optimized_params',
                'initial_energy', 'optimized_energy',
                'optimization_success', 'optimization_iterations',
                'reference_energies'
            ]
            for field in required_fields:
                assert field in logs[0], f"Missing field: {field}"


class TestVQEEnvHartreeFockRecovery:
    """Tests for Hartree-Fock energy recovery (Stage 7.2 verification)."""

    def test_empty_circuit_gives_hf_energy_h2(self):
        """Test that empty circuit (identity) gives HF energy for H2."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        # Immediate DONE - empty circuit
        obs, reward, terminated, truncated, info = env.step(8)

        # Energy should be close to HF
        hf_energy = env.reference_energy
        assert abs(info['optimized_energy'] - hf_energy) < 0.01, \
            f"Empty circuit energy {info['optimized_energy']:.4f} != HF {hf_energy:.4f}"

    def test_empty_circuit_gives_hf_energy_h4(self):
        """Test that empty circuit (identity) gives HF energy for H4."""
        env = VQEArchitectEnv(molecule="H4", max_timesteps=15)
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(24)  # DONE for H4

        hf_energy = env.reference_energy
        assert abs(info['optimized_energy'] - hf_energy) < 0.01, \
            f"Empty circuit energy {info['optimized_energy']:.4f} != HF {hf_energy:.4f}"

    def test_fci_is_lower_than_hf(self):
        """Test that FCI energy is lower than HF (correlation energy is positive)."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        assert env.fci_energy < env.reference_energy

        env_h4 = VQEArchitectEnv(molecule="H4", max_timesteps=15)
        assert env_h4.fci_energy < env_h4.reference_energy


class TestVQEEnvEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_gate_episode(self):
        """Test episode with single gate."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        env.step(0)  # Rx(q0)
        obs, reward, terminated, truncated, info = env.step(8)  # DONE

        assert info['n_gates'] == 1
        assert terminated is True

    def test_all_rotation_types_in_one_episode(self):
        """Test episode with all rotation types."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        env.step(0)  # Rx
        env.step(2)  # Ry
        env.step(4)  # Rz
        obs, reward, terminated, truncated, info = env.step(8)  # DONE

        assert info['n_gates'] == 3
        gate_types = [g['type'] for g in env.circuit_gates]
        assert 'Rx' in gate_types
        assert 'Ry' in gate_types
        assert 'Rz' in gate_types

    def test_get_best_circuit(self):
        """Test retrieval of best circuit."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)
        env.reset(seed=42)

        # Run first episode
        env.step(2)
        env.step(6)
        env.step(8)

        circuit, energy = env.get_best_circuit()
        assert circuit is not None
        assert energy < float('inf')

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        env = VQEArchitectEnv(molecule="H2", max_timesteps=10)

        for ep in range(3):
            env.reset(seed=ep)
            env.step(2)
            env.step(6)
            obs, reward, terminated, truncated, info = env.step(8)
            assert terminated is True
            assert 'optimized_energy' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
