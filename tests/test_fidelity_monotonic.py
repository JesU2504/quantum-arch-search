import cirq
import numpy as np
from qas_gym.utils import fidelity_pure_target

def test_fidelity_monotonic_decrease():
    n_qubits = 2
    qubits = list(cirq.LineQubit.range(n_qubits))
    # Prepare a Bell state
    circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    target_state = result.final_state_vector

    # Apply increasing depolarizing noise to both qubits
    fidelities = []
    for p in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]:
        noisy_circuit = circuit.with_noise(cirq.depolarize(p))
        f = fidelity_pure_target(noisy_circuit, target_state, qubits)
        fidelities.append(f)
    # Assert monotonic non-increasing
    for i in range(1, len(fidelities)):
        assert fidelities[i] <= fidelities[i-1] + 1e-10, f"Fidelity increased: {fidelities}"
