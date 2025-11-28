"""
Central configuration for Quantum Architecture Search experiments.

This module provides centralized configuration for:
- Target type selection (GHZ, Toffoli, etc.)
- Task mode selection (state_preparation vs unitary_preparation)
- Experiment hyperparameters
- Statistical reporting settings

Configuration Options:
----------------------
TARGET_TYPE: str
    The target circuit/state to learn. Options:
    - 'toffoli': n-controlled NOT gate (CNOT for n=2, CCNOT for n=3, etc.)
    - 'ghz': GHZ state preparation (|00...0> + |11...1>) / sqrt(2)

TASK_MODE: str
    The fidelity evaluation mode. Options:
    - 'state_preparation': Single-state fidelity. The circuit is evaluated on
      a fixed input state (typically |0...0> for GHZ or |1...1> for Toffoli).
      This is faster but may allow "trivially high fidelity with wrong circuit"
      if the circuit happens to produce the correct output for one input.
    - 'unitary_preparation': Full basis-sweep fidelity. The circuit is evaluated
      on ALL 2^n computational basis inputs and the fidelity is averaged.
      This guarantees that fidelity=1.0 requires correct behavior on all inputs,
      solving the "trivial fidelity" issue.

Usage Example:
--------------
    from experiments import config

    # Use the configured target and mode
    target_state = config.get_target_state(n_qubits=3)
    fidelity = config.compute_fidelity(circuit, qubits, n_qubits=3)

    # Override for a specific experiment
    target_state = config.get_target_state(n_qubits=3, target_type='ghz')
    fidelity = config.compute_fidelity(circuit, qubits, task_mode='unitary_preparation')
"""

import os
import sys
from typing import Optional, Callable

# Add repository root to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import numpy as np
import cirq

# =============================================================================
# Target Type and Task Mode Configuration
# =============================================================================

# TARGET_TYPE: The type of quantum circuit/state to learn
# Options: 'toffoli', 'ghz'
# Default: 'toffoli' (n-controlled NOT gate)
TARGET_TYPE = 'toffoli'

# TASK_MODE: How to evaluate circuit fidelity
# Options: 'state_preparation', 'unitary_preparation'
# Default: 'state_preparation' (legacy mode, faster but less rigorous)
TASK_MODE = 'state_preparation'

# Valid options for validation
VALID_TARGET_TYPES = ['toffoli', 'ghz']
VALID_TASK_MODES = ['state_preparation', 'unitary_preparation']


def get_target_state(
    n_qubits: int,
    target_type: Optional[str] = None
) -> np.ndarray:
    """
    Get the target state vector for the specified target type.

    Args:
        n_qubits: Number of qubits.
        target_type: Override for TARGET_TYPE. If None, uses global config.

    Returns:
        Target state vector as numpy array.

    Raises:
        ValueError: If target_type is not recognized.
    """
    from qas_gym.utils import get_toffoli_state, get_ghz_state

    effective_target = target_type if target_type is not None else TARGET_TYPE

    if effective_target not in VALID_TARGET_TYPES:
        raise ValueError(
            f"Invalid target_type '{effective_target}'. "
            f"Valid options: {VALID_TARGET_TYPES}"
        )

    if effective_target == 'toffoli':
        return get_toffoli_state(n_qubits)
    elif effective_target == 'ghz':
        return get_ghz_state(n_qubits)
    else:
        raise ValueError(f"Unknown target type: {effective_target}")


def get_target_circuit(
    n_qubits: int,
    target_type: Optional[str] = None
) -> tuple:
    """
    Get the target circuit and qubits for the specified target type.

    Args:
        n_qubits: Number of qubits.
        target_type: Override for TARGET_TYPE. If None, uses global config.

    Returns:
        Tuple of (circuit, qubits).

    Raises:
        ValueError: If target_type is not recognized.
    """
    from qas_gym.utils import create_toffoli_circuit_and_qubits, create_ghz_circuit_and_qubits

    effective_target = target_type if target_type is not None else TARGET_TYPE

    if effective_target not in VALID_TARGET_TYPES:
        raise ValueError(
            f"Invalid target_type '{effective_target}'. "
            f"Valid options: {VALID_TARGET_TYPES}"
        )

    if effective_target == 'toffoli':
        return create_toffoli_circuit_and_qubits(n_qubits)
    elif effective_target == 'ghz':
        return create_ghz_circuit_and_qubits(n_qubits)
    else:
        raise ValueError(f"Unknown target type: {effective_target}")


def compute_fidelity(
    circuit: cirq.Circuit,
    qubits: list,
    n_qubits: Optional[int] = None,
    target_type: Optional[str] = None,
    task_mode: Optional[str] = None,
    noise_model: Optional[cirq.NoiseModel] = None
) -> float:
    """
    Compute fidelity using the configured task mode.

    For 'state_preparation' mode:
        Computes F = <target|rho|target> for a single target state.

    For 'unitary_preparation' mode:
        Computes average fidelity over all 2^n computational basis inputs.
        This ensures fidelity=1.0 only when the circuit correctly implements
        the full truth table of the target gate.

    Args:
        circuit: The circuit to evaluate.
        qubits: Qubits in the circuit.
        n_qubits: Number of qubits. If None, inferred from qubits list.
        target_type: Override for TARGET_TYPE. If None, uses global config.
        task_mode: Override for TASK_MODE. If None, uses global config.
        noise_model: Optional noise model for simulation.

    Returns:
        Fidelity value in [0, 1].

    Raises:
        ValueError: If task_mode or target_type is not recognized.
    """
    from qas_gym.utils import fidelity_pure_target
    from utils.metrics import full_basis_fidelity_toffoli, full_basis_fidelity, toffoli_truth_table

    effective_mode = task_mode if task_mode is not None else TASK_MODE
    effective_target = target_type if target_type is not None else TARGET_TYPE
    effective_n_qubits = n_qubits if n_qubits is not None else len(qubits)

    if effective_mode not in VALID_TASK_MODES:
        raise ValueError(
            f"Invalid task_mode '{effective_mode}'. "
            f"Valid options: {VALID_TASK_MODES}"
        )

    if effective_target not in VALID_TARGET_TYPES:
        raise ValueError(
            f"Invalid target_type '{effective_target}'. "
            f"Valid options: {VALID_TARGET_TYPES}"
        )

    if effective_mode == 'state_preparation':
        # Single-state fidelity (legacy mode)
        target_state = get_target_state(effective_n_qubits, effective_target)
        return fidelity_pure_target(circuit, target_state, qubits)

    elif effective_mode == 'unitary_preparation':
        # Full basis-sweep fidelity
        if effective_target == 'toffoli':
            n_controls = effective_n_qubits - 1
            return full_basis_fidelity_toffoli(
                circuit, qubits, n_controls, noise_model
            )
        elif effective_target == 'ghz':
            # For GHZ, we use state preparation mode since GHZ is about
            # preparing a specific entangled state, not implementing a gate
            # Fall back to state_preparation for GHZ
            target_state = get_target_state(effective_n_qubits, effective_target)
            return fidelity_pure_target(circuit, target_state, qubits)
        else:
            raise ValueError(f"Unknown target type: {effective_target}")

    else:
        raise ValueError(f"Unknown task mode: {effective_mode}")


def get_experiment_label(
    target_type: Optional[str] = None,
    task_mode: Optional[str] = None
) -> str:
    """
    Get a label string for experiment outputs based on current configuration.

    Args:
        target_type: Override for TARGET_TYPE. If None, uses global config.
        task_mode: Override for TASK_MODE. If None, uses global config.

    Returns:
        A string label like 'toffoli_state_prep' or 'ghz_unitary'.
    """
    effective_target = target_type if target_type is not None else TARGET_TYPE
    effective_mode = task_mode if task_mode is not None else TASK_MODE

    mode_short = 'state_prep' if effective_mode == 'state_preparation' else 'unitary'
    return f"{effective_target}_{mode_short}"


def get_config_summary() -> dict:
    """
    Get a summary of the current configuration for logging.

    Returns:
        Dictionary with configuration settings.
    """
    return {
        'target_type': TARGET_TYPE,
        'task_mode': TASK_MODE,
        'valid_target_types': VALID_TARGET_TYPES,
        'valid_task_modes': VALID_TASK_MODES,
    }


# --- Statistical Reporting Configuration ---
# Number of random seeds per experimental setting for statistical validity.
# Minimum recommended: 5, ideal: 10 for publication-quality results.
# Can be overridden via CLI --n-seeds argument in run_experiments.py.
N_SEEDS = 5  # Default number of seeds per experiment setting
N_SEEDS_MIN = 5  # Minimum recommended for statistical validity
N_SEEDS_RECOMMENDED = 10  # Recommended for publication-quality results

# --- General Experiment Parameters ---
N_RUNS = 5 
RESULTS_DIR = "results"
# CRITICAL: This determines the padding size for the Saboteur's input.
# Must be consistent across env creation in train_adversarial.py
MAX_CIRCUIT_TIMESTEPS = 20 

# --- Per-Qubit Hyperparameter Configurations ---
EXPERIMENT_PARAMS = {
    3: {
        "ARCHITECT_N_STEPS": 4096,
        "ARCHITECT_STEPS": 4096 * 10,
        "N_GENERATIONS": 40,
        "ARCHITECT_STEPS_PER_GENERATION": 4096,
        "SABOTEUR_STEPS_PER_GENERATION": 2048,
        "SABOTEUR_N_STEPS": 2048,
        "SABOTEUR_STEPS": 2048 * 8,
    },
    4: {
        "ARCHITECT_N_STEPS": 4096,
        "ARCHITECT_STEPS": 4096 * 40,
        "N_GENERATIONS": 100, 
        "ARCHITECT_STEPS_PER_GENERATION": 4096, 
        "SABOTEUR_STEPS_PER_GENERATION": 2048,
        "SABOTEUR_N_STEPS": 2048,
        "SABOTEUR_STEPS": 2048 * 10, 
    },
}

# --- Agent Hyperparameters ---
AGENT_PARAMS = {
    "n_steps": 1000,
    "batch_size": 100,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    "ent_coef": 0.01,
    "verbose": 0,
    "device": "cpu", 
}

# Saboteur-only Training
SABOTEUR_STEPS = 20000 

# --- Analysis Parameters ---
NOISE_LEVELS = np.linspace(0, 0.01, 20)
N_PREDICTIONS = 100 

from qas_gym.utils import get_gates_by_name

def get_action_gates(qubits: list[cirq.LineQubit]) -> list[cirq.Operation]:
    single_qubit_gate_names = ['X', 'Y', 'Z', 'H', 'T', 'S']
    return get_gates_by_name(qubits, single_qubit_gate_names)