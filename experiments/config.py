import numpy as np
import cirq

# --- General Experiment Parameters ---
N_RUNS = 5 # Use 3-5 runs for statistically meaningful averages
RESULTS_DIR = "results"
MAX_CIRCUIT_TIMESTEPS = 20

# --- Per-Qubit Hyperparameter Configurations ---
# This dictionary allows us to define specific training parameters for each system size.
EXPERIMENT_PARAMS = {
    3: {
        "ARCHITECT_N_STEPS": 4096,
        "ARCHITECT_STEPS": 4096 * 10, # 40,960
        "N_GENERATIONS": 40,
        "ARCHITECT_STEPS_PER_GENERATION": 4096,
        "SABOTEUR_STEPS_PER_GENERATION": 2048,
        "SABOTEUR_N_STEPS": 2048,
        "SABOTEUR_STEPS": 2048 * 8, # 16,384
        "SABOTEUR_MAX_ERROR_LEVEL": 4, # Match the max level in co-evolution curriculum
    },
    4: {
        "ARCHITECT_N_STEPS": 4096,
        "ARCHITECT_STEPS": 4096 * 40, # Increased from 20x to give more time for baseline
        "N_GENERATIONS": 100,         # Increased from 50 to double co-evolution time
        "ARCHITECT_STEPS_PER_GENERATION": 4096, # Kept the same for generation-by-generation updates
        "SABOTEUR_STEPS_PER_GENERATION": 2048,
        "SABOTEUR_N_STEPS": 2048,
        "SABOTEUR_STEPS": 2048 * 10, # 
        "SABOTEUR_MAX_ERROR_LEVEL": 4, # Match the max level in co-evolution curriculum
    },
    # You can easily add a 5-qubit configuration here in the future
}

# --- Agent Hyperparameters ---
# A single source of truth for PPO agent configuration to ensure fair comparisons.
AGENT_PARAMS = {
    "n_steps": 1000,  # For quick runs, use 1000
    "batch_size": 100,  # For quick runs, use 100 (factor of 1000)
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    "ent_coef": 0.01,
    "verbose": 0,
    "device": "cpu", # Explicitly set device for consistent performance
}

# Saboteur-only Training
SABOTEUR_STEPS = 20000 # Default, will be overridden by EXPERIMENT_PARAMS
SABOTEUR_MAX_ERROR_LEVEL = 2 # Default, will be overridden by EXPERIMENT_PARAMS

# --- Analysis Parameters ---
# Noise resilience comparison
NOISE_LEVELS = np.linspace(0, 0.01, 20)  # Test from 0% to 1% error rate

# Saboteur policy analysis
N_PREDICTIONS = 100 # Reduce predictions for faster analysis

# Import should be at the top, but placing here to avoid breaking other logic if this file is imported early.
from qas_gym.utils import get_gates_by_name

def get_action_gates(qubits: list[cirq.LineQubit]) -> list[cirq.Operation]:
    """
    Generates the list of possible gate operations for the architect.
    This function calls a central utility to ensure a single source of truth for the action space.
    """
    # Define the set of single-qubit gates to be used in the experiments.
    single_qubit_gate_names = ['X', 'Y', 'Z', 'H', 'T', 'S']
    return get_gates_by_name(qubits, single_qubit_gate_names)