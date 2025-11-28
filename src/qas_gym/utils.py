import cirq
import numpy as np
from typing import Sequence, List, Dict, Any, Optional, Tuple


# Gate type constants for rotation gates
ROTATION_GATE_TYPES = ['Rx', 'Ry', 'Rz']


def get_gates_by_name(qubits, gate_names, include_rotations=False, default_rotation_angle=np.pi/4):
    """
    Get a list of gate operations for the given qubits and gate names.
    
    Args:
        qubits: List of qubits to apply gates to.
        gate_names: List of gate names (e.g., ['X', 'Y', 'H', 'T', 'S']).
        include_rotations: If True, include parameterized rotation gates (Rx, Ry, Rz)
            with angles specified in radians.
        default_rotation_angle: Default angle for rotation gates in radians (used for
            initial gate operations, can be modified later). Default is π/4.
    
    Returns:
        List of gate operations.
    """
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

    # Add parameterized rotation gates if requested
    if include_rotations:
        for q in qubits:
            # Add Rx, Ry, Rz gates with default angle
            action_gates.append(cirq.rx(default_rotation_angle).on(q))
            action_gates.append(cirq.ry(default_rotation_angle).on(q))
            action_gates.append(cirq.rz(default_rotation_angle).on(q))

    # Add two-qubit CNOT gates for all ordered pairs
    for q1 in qubits:
        for q2 in qubits:
            if q1 != q2:
                action_gates.append(cirq.CNOT(q1, q2))
    return action_gates


def create_rotation_gate(gate_type: str, qubit: cirq.Qid, angle: float) -> cirq.GateOperation:
    """
    Create a parameterized rotation gate.
    
    Args:
        gate_type: Type of rotation gate ('Rx', 'Ry', or 'Rz').
        qubit: Qubit to apply the gate to.
        angle: Rotation angle in radians.
    
    Returns:
        A cirq gate operation.
    
    Raises:
        ValueError: If gate_type is not a valid rotation gate type.
    """
    if gate_type == 'Rx':
        return cirq.rx(angle).on(qubit)
    elif gate_type == 'Ry':
        return cirq.ry(angle).on(qubit)
    elif gate_type == 'Rz':
        return cirq.rz(angle).on(qubit)
    else:
        raise ValueError(f"Invalid rotation gate type: {gate_type}. Must be one of {ROTATION_GATE_TYPES}")


def is_rotation_gate(gate: cirq.Gate) -> bool:
    """
    Check if a gate is a parameterized rotation gate.
    
    Args:
        gate: The gate to check.
    
    Returns:
        True if the gate is Rx, Ry, or Rz.
    """
    return isinstance(gate, (cirq.Rx, cirq.Ry, cirq.Rz))


def get_rotation_gate_info(op: cirq.GateOperation) -> Optional[Dict[str, Any]]:
    """
    Extract information about a rotation gate operation.
    
    Args:
        op: A gate operation.
    
    Returns:
        Dictionary with 'type', 'qubit', and 'angle' if it's a rotation gate,
        None otherwise.
    """
    gate = op.gate
    if isinstance(gate, cirq.Rx):
        return {
            'type': 'Rx',
            'qubit': op.qubits[0],
            'angle': float(gate.exponent * np.pi)
        }
    elif isinstance(gate, cirq.Ry):
        return {
            'type': 'Ry',
            'qubit': op.qubits[0],
            'angle': float(gate.exponent * np.pi)
        }
    elif isinstance(gate, cirq.Rz):
        return {
            'type': 'Rz',
            'qubit': op.qubits[0],
            'angle': float(gate.exponent * np.pi)
        }
    return None


def serialize_circuit_with_rotations(circuit: cirq.Circuit) -> List[Dict[str, Any]]:
    """
    Serialize a circuit including rotation gate parameters.
    
    Args:
        circuit: The cirq circuit to serialize.
    
    Returns:
        List of dictionaries representing each gate operation, including
        rotation angles for parameterized gates.
    """
    serialized = []
    for op in circuit.all_operations():
        gate_info = get_rotation_gate_info(op)
        if gate_info is not None:
            serialized.append({
                'gate_type': gate_info['type'],
                'qubit': str(gate_info['qubit']),
                'angle': gate_info['angle']
            })
        elif isinstance(op.gate, cirq.CNotPowGate):
            serialized.append({
                'gate_type': 'CNOT',
                'control': str(op.qubits[0]),
                'target': str(op.qubits[1])
            })
        else:
            # Other gates
            serialized.append({
                'gate_type': type(op.gate).__name__,
                'qubits': [str(q) for q in op.qubits]
            })
    return serialized


def count_rotation_gates(circuit: cirq.Circuit) -> Dict[str, int]:
    """
    Count rotation gates by type in a circuit.
    
    Args:
        circuit: The cirq circuit to analyze.
    
    Returns:
        Dictionary with counts for 'Rx', 'Ry', 'Rz', and 'total_rotations'.
    """
    counts = {'Rx': 0, 'Ry': 0, 'Rz': 0, 'total_rotations': 0}
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.Rx):
            counts['Rx'] += 1
            counts['total_rotations'] += 1
        elif isinstance(op.gate, cirq.Ry):
            counts['Ry'] += 1
            counts['total_rotations'] += 1
        elif isinstance(op.gate, cirq.Rz):
            counts['Rz'] += 1
            counts['total_rotations'] += 1
    return counts


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


def get_default_gates(qubits, include_rotations=False):
    """
    Get the default set of action gates for quantum architecture search.
    
    Args:
        qubits: List of qubits.
        include_rotations: If True, include parameterized rotation gates (Rx, Ry, Rz).
            This increases circuit expressiveness but also increases action space
            complexity.
    
    Returns:
        List of gate operations including Clifford gates, T gate, and optionally
        rotation gates.
    """
    return get_gates_by_name(qubits, ['X', 'Y', 'Z', 'H', 'T', 'S'], include_rotations=include_rotations)


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
