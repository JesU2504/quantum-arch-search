import numpy as np
import cirq
from qas_gym.utils import (
    create_ghz_circuit_and_qubits,
    fidelity_pure_target,
    build_frame_twirled_noisy_circuit,
)


def one_qubit_plus_state():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))
    qubits = [q]
    sim = cirq.Simulator()
    target = sim.simulate(circuit, qubit_order=qubits).final_state_vector
    return circuit, qubits, target


def sanity_no_noise(n_samples=50):
    circuit, qubits, target = one_qubit_plus_state()
    rng = np.random.default_rng(42)
    fids = []
    for _ in range(n_samples):
        noisy, frame = build_frame_twirled_noisy_circuit(
            circuit, rng, attack_mode='over_rotation', epsilon_overrot=0.0
        )
        fids.append(fidelity_pure_target(noisy, target, qubits, frame))
    return float(np.mean(fids)), float(np.std(fids))


def sanity_small_overrot(epsilon=0.05, n_samples=200):
    circuit, qubits, target = one_qubit_plus_state()
    # Baseline: untwirled add Rx after op
    untwirled_ops = []
    for op in circuit.all_operations():
        untwirled_ops.append(op)
        for q in op.qubits:
            untwirled_ops.append(cirq.rx(epsilon).on(q))
    untwirled = cirq.Circuit(untwirled_ops)
    f_untwirled = fidelity_pure_target(untwirled, target, qubits)

    # Twirled Monte Carlo
    rng = np.random.default_rng(7)
    fids = []
    for _ in range(n_samples):
        noisy, frame = build_frame_twirled_noisy_circuit(
            circuit, rng, attack_mode='over_rotation', epsilon_overrot=epsilon
        )
        fids.append(fidelity_pure_target(noisy, target, qubits, frame))
    return float(f_untwirled), float(np.mean(fids)), float(np.std(fids))


if __name__ == "__main__":
    mn, sd = sanity_no_noise()
    print({"no_noise_mean": mn, "no_noise_std": sd})
    fu, tm, ts = sanity_small_overrot()
    print({"untwirled": fu, "twirled_mean": tm, "twirled_std": ts})
