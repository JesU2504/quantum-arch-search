"""
Tests for architect RL training circuit saving functionality.

This test module validates that the architect training script correctly
saves champion circuits during training.

Tests include:
- Baseline circuit saving regression test (any circuit with fidelity >= 0 saved)
- Config logging at training start
- Champion callback final report generation
"""

import os
import tempfile
import pytest
import numpy as np
import cirq

# Import test utilities
import sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src_root = os.path.join(_repo_root, 'src')
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from experiments.train_architect import ChampionCircuitCallback, train_baseline_architect
from experiments import config
from qas_gym.utils import save_circuit, load_circuit, verify_toffoli_unitary


class TestChampionCircuitCallback:
    """Tests for ChampionCircuitCallback functionality."""
    
    def test_callback_initialization(self):
        """Callback should initialize with correct default values."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            callback = ChampionCircuitCallback(circuit_filename=f.name)
            assert callback.best_fidelity == -1.0
            assert callback.best_saved_fidelity == -1.0
            assert callback.circuit_saved is False
            assert callback.circuits_evaluated == 0
            assert callback.fidelities == []
            assert callback.steps == []
            os.unlink(f.name)
    
    def test_callback_final_report_no_circuit(self):
        """Final report should explain why no circuit was saved."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            callback = ChampionCircuitCallback(circuit_filename=f.name)
            report = callback.get_final_report()
            assert "Circuit saved: No" in report
            assert "CHAMPION CIRCUIT CALLBACK FINAL REPORT" in report
            os.unlink(f.name)
    
    def test_callback_tracks_evaluations(self):
        """Callback should track number of circuits evaluated."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            callback = ChampionCircuitCallback(circuit_filename=f.name)
            # Manually set as if circuits were evaluated
            callback.circuits_evaluated = 5
            callback.circuit_saved = True
            callback.best_fidelity = 0.5
            report = callback.get_final_report()
            assert "Circuits evaluated: 5" in report
            assert "Circuit saved: Yes" in report
            os.unlink(f.name)


class TestVerifyToffoliUnitary:
    """Tests for the unitary verification with correct bit ordering."""
    
    def test_perfect_toffoli_process_fidelity(self):
        """Perfect Toffoli circuit should achieve 1.0 process fidelity."""
        n_qubits = 3
        qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
        circuit = cirq.Circuit(cirq.TOFFOLI(qubits[0], qubits[1], qubits[2]))
        accuracy, process_fid = verify_toffoli_unitary(circuit, n_qubits, silent=True)
        assert accuracy == 1.0, f"Expected accuracy 1.0, got {accuracy}"
        assert abs(process_fid - 1.0) < 1e-6, f"Expected process fidelity 1.0, got {process_fid}"
    
    def test_perfect_cnot_process_fidelity(self):
        """Perfect CNOT circuit should achieve 1.0 process fidelity for 2 qubits."""
        n_qubits = 2
        qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
        circuit = cirq.Circuit(cirq.CNOT(qubits[0], qubits[1]))
        accuracy, process_fid = verify_toffoli_unitary(circuit, n_qubits, silent=True)
        assert accuracy == 1.0, f"Expected accuracy 1.0, got {accuracy}"
        assert abs(process_fid - 1.0) < 1e-6, f"Expected process fidelity 1.0, got {process_fid}"
    
    def test_identity_circuit_low_fidelity(self):
        """Empty/identity circuit should have low process fidelity."""
        n_qubits = 3
        circuit = cirq.Circuit()
        accuracy, process_fid = verify_toffoli_unitary(circuit, n_qubits, silent=True)
        # Identity preserves 6/8 inputs for Toffoli (only |110> and |111> differ)
        assert accuracy == 0.75, f"Expected accuracy 0.75, got {accuracy}"
        # Process fidelity for identity vs Toffoli is 0.5625
        assert abs(process_fid - 0.5625) < 1e-6, f"Expected process fidelity 0.5625, got {process_fid}"


class TestBaselineCircuitSavingRegression:
    """
    Regression tests to ensure baseline circuits are always saved.
    
    This test ensures that even a simple baseline (like CNOT or partial implementation)
    gets saved if the RL agent finds any circuit with fidelity >= 0.
    """
    
    @pytest.mark.timeout(120)
    def test_short_training_saves_circuit_state_mode(self):
        """Short training in state_preparation mode should save at least one circuit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline_architect(
                results_dir=tmpdir,
                n_qubits=3,
                architect_steps=200,  # Very short training
                n_steps=50,
                include_rotations=True,
                target_type='toffoli',
                task_mode='state_preparation'
            )
            circuit_path = os.path.join(tmpdir, 'circuit_vanilla.json')
            assert os.path.exists(circuit_path), (
                "No circuit was saved during training. "
                "The champion callback should save at least one circuit if any fidelity >= 0."
            )
    
    @pytest.mark.timeout(120)
    def test_short_training_saves_circuit_unitary_mode(self):
        """Short training in unitary_preparation mode should save at least one circuit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline_architect(
                results_dir=tmpdir,
                n_qubits=3,
                architect_steps=200,  # Very short training
                n_steps=50,
                include_rotations=True,
                target_type='toffoli',
                task_mode='unitary_preparation'
            )
            circuit_path = os.path.join(tmpdir, 'circuit_vanilla.json')
            assert os.path.exists(circuit_path), (
                "No circuit was saved during training. "
                "The champion callback should save at least one circuit if any fidelity >= 0."
            )
    
    @pytest.mark.timeout(120)
    def test_saved_circuit_can_be_loaded(self):
        """Saved circuit should be loadable and valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline_architect(
                results_dir=tmpdir,
                n_qubits=3,
                architect_steps=200,
                n_steps=50,
                include_rotations=True,
                target_type='toffoli',
                task_mode='state_preparation'
            )
            circuit_path = os.path.join(tmpdir, 'circuit_vanilla.json')
            if os.path.exists(circuit_path):
                circuit = load_circuit(circuit_path)
                assert circuit is not None
                assert isinstance(circuit, cirq.Circuit)


class TestConfigLogging:
    """Tests for config logging at training start."""
    
    def test_config_has_required_attributes(self):
        """Config module should have all required attributes."""
        required_attrs = [
            'TARGET_TYPE',
            'TASK_MODE',
            'MAX_CIRCUIT_TIMESTEPS',
            'EXPERIMENT_PARAMS',
            'AGENT_PARAMS',
            'get_target_state',
            'get_target_circuit',
            'get_experiment_label',
            'get_action_gates',
        ]
        for attr in required_attrs:
            assert hasattr(config, attr), f"Config missing required attribute: {attr}"
    
    def test_get_experiment_label(self):
        """get_experiment_label should return expected format."""
        label = config.get_experiment_label('toffoli', 'unitary_preparation')
        assert label == 'toffoli_unitary_preparation'
        
        label = config.get_experiment_label('ghz', 'state_preparation')
        assert label == 'ghz_state_preparation'
