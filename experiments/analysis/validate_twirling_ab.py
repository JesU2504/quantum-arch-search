import numpy as np
import cirq
from qas_gym.utils import fidelity_pure_target, randomized_compile, create_ghz_circuit_and_qubits, is_twirl_op


def build_simple_3q_circuit():
    # Use GHZ as a simple 3-qubit coherent target circuit
    circuit, qubits = create_ghz_circuit_and_qubits(3)
    return circuit, qubits


def apply_over_rotation_skip_twirl(circuit: cirq.Circuit, epsilon: float) -> cirq.Circuit:
    new_ops = []
    for op in circuit.all_operations():
        new_ops.append(op)
        if is_twirl_op(op):
            continue
        for q in op.qubits:
            new_ops.append(cirq.rx(epsilon).on(q))
    return cirq.Circuit(new_ops)


def main():
    rng = np.random.default_rng(123)
    circuit, qubits = build_simple_3q_circuit()
    # Target GHZ state
    sim = cirq.Simulator()
    res = sim.simulate(circuit, qubit_order=qubits)
    target = res.final_state_vector

    epsilon = 0.1

    # Untwirled
    untwirled_noisy = apply_over_rotation_skip_twirl(cirq.Circuit(circuit.all_operations()), epsilon)
    f_untwirled = fidelity_pure_target(untwirled_noisy, target, qubits)

    # Twirled (current randomized_compile implementation with tags)
    twirled = randomized_compile(circuit, rng)
    twirled_noisy = apply_over_rotation_skip_twirl(twirled, epsilon)
    f_twirled = fidelity_pure_target(twirled_noisy, target, qubits)

    print({
        "epsilon": epsilon,
        "fidelity_untwirled": f_untwirled,
        "fidelity_twirled": f_twirled,
        "uplift": f_twirled - f_untwirled,
    })


if __name__ == "__main__":
    main()
