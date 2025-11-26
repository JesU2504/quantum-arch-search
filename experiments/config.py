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