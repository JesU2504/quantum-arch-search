#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from qas_gym.utils import create_ghz_circuit_and_qubits, apply_noise, get_ghz_state
import cirq


def main():
    # Basic smoke checks for utils
    circuit, qubits = create_ghz_circuit_and_qubits(3)
    assert len(qubits) == 3, "Expected 3 qubits in GHZ circuit"
    assert isinstance(circuit, cirq.Circuit), "Expected a cirq.Circuit"

    state = get_ghz_state(3)
    assert state.shape[0] == 2 ** 3, "GHZ state vector has incorrect size"

    noisy = apply_noise(circuit, gate_index=0, error_rate=0.01)
    assert isinstance(noisy, cirq.Circuit), "apply_noise should return a cirq.Circuit"

    print("SMOKE_OK")


if __name__ == '__main__':
    main()
