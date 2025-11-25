import cirq
import numpy as np
from typing import Sequence


def get_gates_by_name(qubits, gate_names):
    gates = {
        'X': cirq.X,
        'Y': cirq.Y,
        'Z': cirq.Z,
        'H': cirq.H,
        'T': cirq.T,
        'S': cirq.S,
    }
    action_gates = []
    # Add single-qubit gates
    if gate_names:
        for q in qubits:
            for gate_name in gate_names:
                if gate_name in gates:
                    action_gates.append(gates[gate_name](q))

    # Add two-qubit CNOT gates for all ordered pairs
    for q1 in qubits:
        for q2 in qubits:
            if q1 != q2:
                action_gates.append(cirq.CNOT(q1, q2))
    return action_gates


def get_observables_by_name(qubits, observable_names):
    observables = {
        'X': cirq.X,
        'Y': cirq.Y,
        'Z': cirq.Z,
    }
    state_observables = []
    for i in range(len(qubits)):
        for observable_name in observable_names:
            state_observables.append(observables[observable_name](qubits[i]))
    return state_observables


def apply_noise(circuit, gate_index, error_rate):
    ops = list(circuit.all_operations())
    if not ops:
        return circuit

    gate_index %= len(ops)
    gate_to_attack = ops[gate_index]
    qubits_to_attack = gate_to_attack.qubits
    noise_ops = [cirq.depolarize(p=error_rate).on(q) for q in qubits_to_attack]
    new_ops = []
    for i, op in enumerate(ops):
        new_ops.append(op)
        if i == gate_index:
            new_ops.extend(noise_ops)
    return cirq.Circuit(new_ops)


def get_bell_state():
    return np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


def get_ghz_state(n_qubits):
    """
    Generates the GHZ state vector by simulating the canonical preparation circuit.
    This ensures the target state is perfectly consistent with the circuit logic.
    The state is (|0...0> + |1...1>) / sqrt(2).
    """
    circuit, _ = create_ghz_circuit_and_qubits(n_qubits)
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    # The state vector is guaranteed to be real for the canonical GHZ circuit.
    return result.final_state_vector


def fidelity_pure_target(circuit: cirq.Circuit, target_state: np.ndarray, qubits: Sequence[cirq.Qid]) -> float:
    """Compute fidelity F = <psi| rho |psi> for a pure target state |psi>.

    Args:
        circuit: Circuit to simulate.
        target_state: Target state vector (normalized) as a 1D numpy array.
        qubits: Qubit ordering for simulation.
    Returns:
        Fidelity as a float in [0,1]. Returns 0.0 for empty circuit.
    """
    if circuit is None or not circuit.all_operations():
        return 0.0
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.simulate(circuit, qubit_order=qubits)
    rho = result.final_density_matrix
    # Ensure Hermiticity numerically (mitigate tiny simulation asymmetries)
    rho = 0.5 * (rho + np.conj(rho).T)
    return float(np.real(np.vdot(target_state, rho @ target_state)))


def get_default_observables(qubits):
    return get_observables_by_name(qubits, ['X', 'Y'])


def get_default_gates(qubits):
    return get_gates_by_name(qubits, ['X', 'Y', 'Z', 'H', 'T', 'S'])


def create_ghz_circuit_and_qubits(n_qubits: int) -> tuple[cirq.Circuit, list[cirq.LineQubit]]:
    """
    Creates the canonical GHZ state preparation circuit and the corresponding qubits.

    Args:
        n_qubits (int): The number of qubits.

    Returns:
        A tuple containing the cirq.Circuit and the list of cirq.LineQubits.
    """
    qubits = list(cirq.LineQubit.range(n_qubits))
    circuit = cirq.Circuit()
    if n_qubits > 0:
        # Apply Hadamard to the first qubit
        circuit.append(cirq.H(qubits[0]))
        # Append explicit CNOTs to avoid generator-expression ambiguity
        if n_qubits > 1:
            cnot_ops = [cirq.CNOT(qubits[0], qubits[i]) for i in range(1, n_qubits)]
            circuit.append(cnot_ops)
    return circuit, qubits


def save_circuit(path: str, circuit: cirq.Circuit) -> None:
    """Save a Cirq circuit to a JSON file.

    Args:
        path: Path to write the circuit JSON to.
        circuit: The Cirq Circuit to save.
    """
    json_str = cirq.to_json(circuit)
    with open(path, "w") as f:
        f.write(json_str)


def load_circuit(path: str) -> cirq.Circuit:
    """Load a Cirq circuit from a JSON file.

    Args:
        path: Path to the circuit JSON file.

    Returns:
        A Cirq Circuit object.
    """
    return cirq.read_json(path)
