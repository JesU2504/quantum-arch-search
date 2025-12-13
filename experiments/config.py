# --- Target State and Circuit Utilities ---
def get_target_state(n_qubits, target_type=None):
    """Delegate to qas_gym.utils.get_target_state for central logic."""
    from qas_gym.utils import get_target_state as _util_get_target_state
    tt = target_type if target_type is not None else TARGET_TYPE
    return _util_get_target_state(n_qubits, tt)

def get_target_circuit(n_qubits, target_type=None, include_input_prep=True):
    """Delegate to qas_gym.utils.get_target_circuit for central logic."""
    from qas_gym.utils import get_target_circuit as _util_get_target_circuit
    tt = target_type if target_type is not None else TARGET_TYPE
    return _util_get_target_circuit(n_qubits, tt, include_input_prep)

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
# Bumped to allow deeper circuits during robust training.
MAX_CIRCUIT_TIMESTEPS = 15 

# --- Default Target Type for Experiments ---
TARGET_TYPE = "ghz"  # or "toffoli" as needed

# --- Default Task Mode for Experiments ---
TASK_MODE = "state_preparation"  # or "unitary_preparation" as needed

# --- Task Mode to Metric Key Mapping ---
# Maps supported task modes to their corresponding evaluation metric keys.
_METRIC_FOR_TASK_MODE = {
    "state_preparation": "state_preparation_metric",
    "unitary_preparation": "unitary_preparation_metric",
}


def get_metric_key_for_task_mode(task_mode: str | None = None) -> str:
    """
    Return the metric key string for a given task mode.

    Use this helper to look up the evaluation metric key corresponding to a
    task mode instead of hard-coding metric names elsewhere.

    Args:
        task_mode: The task mode string. If None, defaults to the module-level
            TASK_MODE value.

    Returns:
        The metric key string (e.g., 'state_preparation_metric').

    Raises:
        ValueError: If task_mode is not a supported value.

    Example:
        >>> from experiments.config import get_metric_key_for_task_mode
        >>> metric_key = get_metric_key_for_task_mode()
    """
    # Use module default if caller passes None
    mode = task_mode if task_mode is not None else TASK_MODE
    if mode not in _METRIC_FOR_TASK_MODE:
        raise ValueError(
            f"Unknown task_mode '{mode}'. Supported modes: "
            f"{', '.join(_METRIC_FOR_TASK_MODE.keys())}"
        )
    return _METRIC_FOR_TASK_MODE[mode]


# --- Per-Qubit Hyperparameter Configurations ---
EXPERIMENT_PARAMS = {
    # "Quick" Implementation (3 Qubits) - For debugging and rapid testing
    # MODIFIED (Publication): Increased saboteur by 25% (4096 → 5120) for better attack sophistication
    3: {
        # More rollout per update to help PPO escape the 0.69 plateau
        "ARCHITECT_N_STEPS": 2048,
        # Baseline Total = Steps/Gen * Generations (short run with cushion to hit ~1.0)
        "ARCHITECT_STEPS": 12000 * 5,      
        
        "N_GENERATIONS": 5,
        "ARCHITECT_STEPS_PER_GENERATION": 12000,
        
        # Give the saboteur enough budget to learn, scaled to shorter run
        # INCREASED: 4096 → 5120 (+25%) for publication-quality adversarial training
        "SABOTEUR_STEPS_PER_GENERATION": 5120,
        "SABOTEUR_N_STEPS": 5120,
        # Saboteur Total
        "SABOTEUR_STEPS": 5120 * 5,      # = 25,600 steps (was 20,480, +25%)
    },
    
    # "Full" Implementation (4 Qubits) - Standard Experiment (ExpPlan.md)
    # MODIFIED (Publication): Increased architect steps by 17% and saboteur steps by 80%
    # Rationale: Better architect exploration + ultra-trained saboteur for publication-grade adversarial training
    4: {
        # More rollout per update to help PPO escape the 0.69 plateau
        "ARCHITECT_N_STEPS": 2048,
        # Baseline Total = Steps/Gen * Generations (increased for better exploration)
        "ARCHITECT_STEPS": 14000 * 10,      # = 140,000 steps (was 120,000, +17%)
        
        "N_GENERATIONS": 10,
        "ARCHITECT_STEPS_PER_GENERATION": 14000,  # was 12000
        
        # Give the saboteur enough budget to learn extremely well (increased by 80% total)
        # INCREASED: 6144 → 7360 (+80% from original 4096) for publication-quality attacks
        "SABOTEUR_STEPS_PER_GENERATION": 7360,     # was 4096, now +80%
        "SABOTEUR_N_STEPS": 7360,
        # Saboteur Total
        "SABOTEUR_STEPS": 7360 * 10,      # = 73,600 steps (was 40,960, +80%)
    },
    
    # "Long" Implementation (5 Qubits) - Scalability / Wall Test
    # MODIFIED (Publication): Increased saboteur steps by 87.5% to produce ultra-trained, sophisticated attacks
    # Architect steps kept at current level (180K is already substantial for 5Q)
    # Rationale: Publication-quality adversarial training requires heavily-trained saboteur at 5 qubits
    5: {
        # More rollout per update to help PPO escape the 0.69 plateau
        "ARCHITECT_N_STEPS": 2048,
        # Baseline Total = Steps/Gen * Generations (kept as-is, already substantial)
        "ARCHITECT_STEPS": 12000 * 15,      # = 180,000 steps (unchanged)
        
        "N_GENERATIONS": 15,
        "ARCHITECT_STEPS_PER_GENERATION": 12000,
        
        # Give the saboteur enough budget to learn extremely well (increased by 87.5% total)
        # INCREASED: 6144 → 7680 (+87.5% from original 4096) for publication-grade adversarial training
        "SABOTEUR_STEPS_PER_GENERATION": 7680,     # was 4096, now +87.5%
        "SABOTEUR_N_STEPS": 7680,
        # Saboteur Total
        "SABOTEUR_STEPS": 7680 * 15,      # = 115,200 steps (was 61,440, +87.5%)
    }
}

# Default helpers for scripts that need a quick, module-level fallback.
# Use the 4-qubit entry when available (the "standard" experiment), otherwise
# fall back to the first configured qubit count.
def get_params_for_qubits(n_qubits: int) -> dict:
    """Return the parameter set for the given qubit count or raise if missing."""
    if n_qubits not in EXPERIMENT_PARAMS:
        raise ValueError(
            f"No experiment parameters defined for n_qubits={n_qubits}. "
            f"Available: {sorted(EXPERIMENT_PARAMS.keys())}"
        )
    return EXPERIMENT_PARAMS[n_qubits]


DEFAULT_N_QUBITS = 4 if 4 in EXPERIMENT_PARAMS else next(iter(EXPERIMENT_PARAMS))
_DEFAULT_PARAMS = get_params_for_qubits(DEFAULT_N_QUBITS)

# Script-level defaults (used as argparse defaults in training scripts)
ADVERSARIAL_GENS = _DEFAULT_PARAMS["N_GENERATIONS"]
ARCHITECT_N_STEPS = _DEFAULT_PARAMS["ARCHITECT_STEPS_PER_GENERATION"]
SABOTEUR_N_STEPS = _DEFAULT_PARAMS["SABOTEUR_STEPS_PER_GENERATION"]

# --- Agent Hyperparameters ---
AGENT_PARAMS = {
    # PPO stability / exploration tweaks to improve small-GHZ convergence
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.92,
    "learning_rate": 3e-4,
    "ent_coef": 0.02,
    "verbose": 0,
    "device": "cpu", 
}

# Saboteur-only Training
SABOTEUR_STEPS = 20000 

# --- Verbosity controls for experiment logs ---
# Control how chatty the saboteur-only training callback is.
# Set SABOTEUR_VERBOSE=0 to silence periodic distributions; 1 to enable.
SABOTEUR_VERBOSE = 0
# How often (in timesteps) the saboteur callback reports action distribution
# and average reward. Set very large to effectively disable.
SABOTEUR_LOG_INTERVAL = 10_000

# --- Analysis Parameters ---
NOISE_LEVELS = np.linspace(0, 0.01, 20)
N_PREDICTIONS = 100 

from qas_gym.utils import get_gates_by_name

# --- Rotation Gate Configuration ---
# Default setting for including parameterized rotation gates (Rx, Ry, Rz) in experiments.
# Set to True to enable VQE-style variational circuits with more expressive action space.
# Set to False (default) for backward compatibility with Clifford+T gate set.
INCLUDE_ROTATIONS = True
# Limit rotation gate types to reduce action space while preserving Toffoli synthesis.
# Rz(π/4) acts as the T gate (up to global phase). Rx is helpful for GHZ prep.
ROTATION_TYPES = ['Rz', 'Rx']


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
    # Keep a minimal discrete primitive set for GHZ: H is sufficient
    # (optionally add 'X' if explicit bit-flips are desired).
    single_qubit_gate_names = ['H']
    # Use a discrete set of allowed rotation angles (radians). Expanded from a
    # single π/4 value to a small grid so the agent can choose finer rotations.
    # Keep the grid reasonably small to avoid exploding the action space.
    # Symmetric discrete grid of rotation angles (radians). We include
    # negative angles so the agent can select clockwise vs counter-clockwise
    # rotations. The grid is intentionally small to limit action-space growth.
    # Smallest symmetric grid: ±{π/16, π/8}. Keeps action-space small but
    # provides sign and fine-grained control for Rx/Rz.
    allowed_angles = np.array([
        -np.pi / 8,
        -np.pi / 16,
        np.pi / 16,
        np.pi / 8,
    ])
    return get_gates_by_name(
        qubits,
        single_qubit_gate_names,
        include_rotations=include_rotations,
        default_rotation_angle=allowed_angles,
        rotation_types=ROTATION_TYPES
    )
