import os
import sys
import tempfile

# Ensure repository root is on sys.path so tests can import local modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.train_architect_ghz import ChampionCircuitCallback
from qas_gym.utils import create_ghz_circuit_and_qubits


def test_champion_callback_records_and_saves():
    with tempfile.TemporaryDirectory() as td:
        circuit_file = os.path.join(td, 'champ.json')
        # Create a small GHZ circuit to act as champion
        circuit, _ = create_ghz_circuit_and_qubits(3)

        cb = ChampionCircuitCallback(circuit_filename=circuit_file)
        # Simulate internal SB3 state
        cb.num_timesteps = 123
        cb.last_printed_step = 0
        cb.print_freq = 1_000_000  # avoid printing in test

        # Make training_env provide the champion via get_attr
        class DummyEnv:
            def __init__(self, circuit):
                self._circuit = circuit

            def get_attr(self, name):
                if name == 'champion_circuit':
                    return [self._circuit]
                if name == 'best_fidelity':
                    return [1.0]
                raise KeyError(name)

        # BaseCallback.training_env is a property without a setter; monkeypatch
        # the class property temporarily so it returns our DummyEnv for this
        # instance.
        orig_prop = type(cb).training_env
        try:
            type(cb).training_env = property(lambda self: DummyEnv(circuit))
            # Simulate episode end with lower fidelity in info (so champion is only visible via env)
            cb.locals = {'dones': [True], 'infos': [{'fidelity': 0.2}]}

            # Initially no fidelities
            assert cb.fidelities == []
            # Call _on_step which should detect champion via env.get_attr and save it
            ok = cb._on_step()
            assert ok
            # File should be created
            assert os.path.exists(circuit_file)
            # Callback should have recorded the champion fidelity (1.0)
            assert cb.fidelities[-1] == 1.0
        finally:
            # Restore original property
            type(cb).training_env = orig_prop
        
