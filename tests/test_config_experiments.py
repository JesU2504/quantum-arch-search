import pytest
from experiments import config

# List of expected config attributes used by experiments
EXPECTED_CONFIG_ATTRS = [
    'N_SEEDS',
    'N_SEEDS_MIN',
    'N_SEEDS_RECOMMENDED',
    'N_RUNS',
    'RESULTS_DIR',
    'MAX_CIRCUIT_TIMESTEPS',
    'EXPERIMENT_PARAMS',
    'AGENT_PARAMS',
    'SABOTEUR_STEPS',
    'NOISE_LEVELS',
    'N_PREDICTIONS',
    'INCLUDE_ROTATIONS',
]

def test_config_has_all_expected_attrs():
    for attr in EXPECTED_CONFIG_ATTRS:
        assert hasattr(config, attr), f"config.py missing attribute: {attr}"

# Optionally, test that EXPERIMENT_PARAMS contains expected keys (e.g., for 3, 4, 5 qubits)
def test_experiment_params_keys():
    assert isinstance(config.EXPERIMENT_PARAMS, dict)
    for n_qubits in [3, 4, 5]:
        assert n_qubits in config.EXPERIMENT_PARAMS, f"EXPERIMENT_PARAMS missing key: {n_qubits}"
