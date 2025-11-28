# --- Target State and Circuit Utilities ---
def get_target_state(n_qubits, target_type=None):
    """
    Returns the target state vector for the given number of qubits and target type.
    Supported target types: 'ghz', 'toffoli'.
    """
    import numpy as np
    if target_type is None:
        target_type = TARGET_TYPE
    if target_type.lower() == "ghz":
        # GHZ state: (|00...0> + |11...1>)/sqrt(2)
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1/np.sqrt(2)
        state[-1] = 1/np.sqrt(2)
        return state
    elif target_type.lower() == "toffoli":
        # Use utility to get correct Toffoli target state for any n_qubits
        from qas_gym.utils import get_toffoli_target_state
        return get_toffoli_target_state(n_qubits)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

def get_target_circuit(n_qubits, target_type=None, include_input_prep=True):
    """
    Returns a Cirq circuit that prepares the target state for the given number of qubits and target type.
    Supported target types: 'ghz', 'toffoli'.
    """
    import cirq
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    if target_type is None:
        target_type = TARGET_TYPE
    if target_type.lower() == "ghz":
        # Prepare GHZ state: H on q0, CNOT chain
        circuit.append(cirq.H(qubits[0]))
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
        return circuit, qubits
    elif target_type.lower() == "toffoli":
        # Use utility to get correct Toffoli preparation circuit for any n_qubits
        from qas_gym.utils import create_toffoli_circuit_and_qubits
        return create_toffoli_circuit_and_qubits(n_qubits)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
# --- Utility: Experiment Label Helper ---
def get_experiment_label(target_type, task_mode):
    """
    Returns a string label for the experiment based on target type and task mode.
    """
    return f"{target_type}_{task_mode}"
import os
import sys

# Add repository root to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import numpy as np
import cirq

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

# --- Default Target Type for Experiments ---
TARGET_TYPE = "toffoli"  # or "toffoli" as needed

# --- Default Task Mode for Experiments ---
TASK_MODE = "unitary_preparation"  # or "unitary_preparation" as needed

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
    5: {
        "ARCHITECT_N_STEPS": 4096,
        "ARCHITECT_STEPS": 4096 * 80,
        "N_GENERATIONS": 200,
        "ARCHITECT_STEPS_PER_GENERATION": 4096,
        "SABOTEUR_STEPS_PER_GENERATION": 2048,
        "SABOTEUR_N_STEPS": 2048,
        "SABOTEUR_STEPS": 2048 * 20,
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

# --- Rotation Gate Configuration ---
# Default setting for including parameterized rotation gates (Rx, Ry, Rz) in experiments.
# Set to True to enable VQE-style variational circuits with more expressive action space.
# Set to False (default) for backward compatibility with Clifford+T gate set.
INCLUDE_ROTATIONS = True


def get_action_gates(
    qubits: list[cirq.LineQubit],
    include_rotations: bool = INCLUDE_ROTATIONS
) -> list[cirq.Operation]:
    """
    Get the action gates for quantum architecture search experiments.
    
    Args:
        qubits: List of qubits to apply gates to.
        include_rotations: If True, include parameterized rotation gates (Rx, Ry, Rz)
            in addition to the Clifford+T gates. When True, the action space becomes
            more expressive (suitable for VQE-style variational circuits).
            Defaults to INCLUDE_ROTATIONS config value.
    
    Returns:
        List of gate operations including single-qubit gates, optionally rotation
        gates, and two-qubit CNOT gates for all ordered qubit pairs.
    
    Note:
        The default gate set is Clifford+T: X, Y, Z, H, T, S (plus CNOT).
        When include_rotations=True, Rx, Ry, Rz gates are added for each qubit.
    """
    single_qubit_gate_names = ['H'] #['X', 'Y', 'Z', 'H', 'T', 'S']
    # Define allowed rotation angles
    allowed_angles = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi]
    return get_gates_by_name(
        qubits,
        single_qubit_gate_names,
        include_rotations=include_rotations,
        default_rotation_angle=allowed_angles
    )