import numpy as np
import cirq
from qas_gym.utils import (
    fidelity_pure_target,
    create_ghz_circuit_and_qubits,
    build_frame_twirled_noisy_circuit,
)


def build_simple_3q_circuit():
    circuit, qubits = create_ghz_circuit_and_qubits(3)
    return circuit, qubits


def apply_over_rotation(circuit: cirq.Circuit, epsilon: float) -> cirq.Circuit:
    new_ops = []
    for op in circuit.all_operations():
        new_ops.append(op)
        for q in op.qubits:
            new_ops.append(cirq.rx(epsilon).on(q))
    return cirq.Circuit(new_ops)


def run_ab(n_qubits=3, epsilon=0.1, n_samples=200):
    circuit, qubits = create_ghz_circuit_and_qubits(n_qubits)
    # Target GHZ state
    sim = cirq.Simulator()
    target = sim.simulate(circuit, qubit_order=qubits).final_state_vector

    # Untwirled baseline
    untwirled_noisy = apply_over_rotation(cirq.Circuit(circuit.all_operations()), epsilon)
    f_untwirled = fidelity_pure_target(untwirled_noisy, target, qubits)

    # Twirled average over multiple frames
    rng = np.random.default_rng(1234)
    fids = []
    for _ in range(n_samples):
        twirled_noisy, frame = build_frame_twirled_noisy_circuit(
            circuit,
            rng,
            attack_mode='over_rotation',
            epsilon_overrot=epsilon,
        )
        fids.append(fidelity_pure_target(twirled_noisy, target, qubits, frame))

    return {
        "epsilon": epsilon,
        "n_samples": n_samples,
        "fidelity_untwirled": float(f_untwirled),
        "fidelity_twirled_mean": float(np.mean(fids)),
        "fidelity_twirled_std": float(np.std(fids)),
    }


def main():
    out = run_ab(n_qubits=3, epsilon=0.1, n_samples=200)
    print(out)


if __name__ == "__main__":
    main()
