import cirq
import numpy as np
from typing import Sequence, List, Dict, Any, Optional, Tuple


# Gate type constants for rotation gates
ROTATION_GATE_TYPES = ['Rx', 'Ry', 'Rz']


from typing import Sequence, Union

def get_gates_by_name(
    qubits,
    gate_names,
    include_rotations=False,
    default_rotation_angle: Union[float, Sequence[float]] = np.pi/4
):

    """
    Get a list of gate operations for the given qubits and gate names.
    
    Args:
        qubits: List of qubits to apply gates to.
        gate_names: List of gate names (e.g., ['X', 'Y', 'H', 'T', 'S']).
        include_rotations: If True, include parameterized rotation gates (Rx, Ry, Rz)
            with angles specified in radians.
        default_rotation_angle: (float or list/tuple/np.ndarray of floats) Angle(s) for rotation gates in radians.
            If a sequence is provided, all angles will be included for each rotation type/qubit.
            Default is π/4.
    
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
        # Support a list of angles
        if isinstance(default_rotation_angle, (list, tuple, np.ndarray)):
            angles = default_rotation_angle
        else:
            angles = [default_rotation_angle]
        for q in qubits:
            for angle in angles:
                a = float(angle)  # type: ignore
                action_gates.append(cirq.rx(a).on(q))
                action_gates.append(cirq.ry(a).on(q))
                action_gates.append(cirq.rz(a).on(q))

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


def get_toffoli_unitary(n_qubits: int) -> np.ndarray:
    """
    Generate the unitary matrix for an n-controlled Toffoli (multi-controlled NOT) gate.
    
    For n qubits, this creates an (n-1)-controlled NOT gate where qubits 0 to n-2 are 
    controls and qubit n-1 is the target. The gate flips the target qubit if and only 
    if all control qubits are in state |1>.
    
    Uses Cirq's big-endian convention where qubit 0 is MSB in state indexing.
    
    - n=2: CNOT (1 control, 1 target)
    - n=3: Toffoli/CCNOT (2 controls, 1 target)
    - n=4: CCCNOT (3 controls, 1 target)
    - etc.
    
    Args:
        n_qubits: Number of qubits (>= 2). Control qubits are 0 to n-2, target is n-1.
        
    Returns:
        Unitary matrix of shape (2^n, 2^n) representing the n-controlled Toffoli gate.
    """
    if n_qubits < 2:
        raise ValueError("n-controlled Toffoli requires at least 2 qubits (n >= 2)")
    
    dim = 2 ** n_qubits
    unitary = np.eye(dim, dtype=complex)
    
    # Cirq uses big-endian: qubit 0 is MSB
    # For state index i, bit at position j (from MSB) corresponds to qubit j
    # Controls: qubits 0 to n-2 (bits at positions 0 to n-2 from MSB)
    # Target: qubit n-1 (LSB position)
    
    # Find all indices where all control qubits are |1>
    # Control bits are in positions n_qubits-1 down to 1 in the index
    # We need: bits at positions n_qubits-1, n_qubits-2, ..., 1 all equal to 1
    # This means the index has pattern: 11...1? where ? is the target bit
    
    # Control mask: bits n-2 to 0 (the first n-1 qubits in big-endian)
    # In the index, these are the upper n-1 bits
    for i in range(dim):
        # Check if all control bits are 1
        # Control bits are at positions n_qubits-1 down to 1 (shifted right by 1)
        control_bits = i >> 1  # Remove target bit
        all_controls_one = control_bits == (dim // 2 - 1)  # All n-1 bits are 1
        
        if all_controls_one:
            # Flip target bit (bit 0)
            target_idx = i ^ 1  # XOR with 1 flips the LSB (target)
            
            # Swap in the unitary
            unitary[i, i] = 0
            unitary[i, target_idx] = 1
    
    return unitary


from typing import Optional
def get_toffoli_target_state(n_qubits: int, input_state: Optional[str] = None) -> np.ndarray:
    """
    Generate the output state of an n-controlled Toffoli gate for a given input.
    
    This computes what the n-controlled NOT gate produces when applied to an input state.
    The default input state is |11...1> (all qubits in |1>). When the Toffoli gate is
    applied, all control qubits are satisfied (all |1>), so the target qubit is flipped.
    
    The result is |11...10> - the target qubit (last qubit) flips from |1> to |0>.
    
    Args:
        n_qubits: Number of qubits (>= 2).
        input_state: Optional input state as binary string (e.g., '111' for |111>).
                    If None, defaults to all-ones |11...1>.
    
    Returns:
        Output state vector after applying the n-controlled Toffoli gate.
    """
    if n_qubits < 2:
        raise ValueError("n-controlled Toffoli requires at least 2 qubits (n >= 2)")
    
    dim = 2 ** n_qubits
    
    # Default input: all ones |11...1>
    if input_state is None:
        input_idx = dim - 1  # All bits set = 2^n - 1
    else:
        input_idx = int(input_state, 2)
    
    # Create input state vector
    input_vec = np.zeros(dim, dtype=complex)
    input_vec[input_idx] = 1.0
    
    # Apply Toffoli unitary
    unitary = get_toffoli_unitary(n_qubits)
    output_vec = unitary @ input_vec
    
    return output_vec


def create_toffoli_circuit_and_qubits(n_qubits: int) -> tuple[cirq.Circuit, list[cirq.LineQubit]]:
    """
    Creates a circuit implementing the n-controlled Toffoli (multi-controlled NOT) gate.
    
    For n qubits, this implements an (n-1)-controlled NOT gate where:
    - Qubits 0 to n-2 are control qubits
    - Qubit n-1 is the target qubit
    
    The circuit prepares all qubits in |1> (via X gates) and then applies the 
    multi-controlled NOT, resulting in the target state |11...10> (target flipped).
    
    - n=2: CNOT preparation circuit
    - n=3: Toffoli (CCNOT) preparation circuit 
    - n=4: CCCNOT preparation circuit (using decomposition)
    - etc.
    
    Args:
        n_qubits: Number of qubits (>= 2).
        
    Returns:
        Tuple of (circuit, qubits) where circuit prepares the Toffoli target state.
    """
    if n_qubits < 2:
        raise ValueError("n-controlled Toffoli requires at least 2 qubits (n >= 2)")
    
    qubits = list(cirq.LineQubit.range(n_qubits))
    circuit = cirq.Circuit()
    
    # Step 1: Prepare all qubits in |1> state (apply X to all qubits)
    for q in qubits:
        circuit.append(cirq.X(q))
    
    # Step 2: Apply the n-controlled NOT gate
    # Controls: qubits[0:-1], Target: qubits[-1]
    controls = qubits[:-1]
    target = qubits[-1]
    
    if n_qubits == 2:
        # Simple CNOT
        circuit.append(cirq.CNOT(controls[0], target))
    elif n_qubits == 3:
        # Toffoli (CCNOT) gate - built into Cirq
        circuit.append(cirq.TOFFOLI(controls[0], controls[1], target))
    else:
        # For n >= 4: Use ControlledGate to create (n-1)-controlled X
        # Cirq supports multi-controlled gates via controlled_by()
        controlled_x = cirq.X(target).controlled_by(*controls)
        circuit.append(controlled_x)
    
    return circuit, qubits


def get_toffoli_state(n_qubits: int) -> np.ndarray:
    """
    Generates the Toffoli target state by simulating the preparation circuit.
    
    This is the primary function for obtaining the target state for n-qubit
    Toffoli-based experiments. It simulates the canonical Toffoli preparation
    circuit to ensure perfect consistency between the target and the circuit logic.
    
    The resulting state is |11...10> for n >= 2 qubits, which is the result of
    applying an n-controlled NOT to the |11...1> input state.
    
    Args:
        n_qubits: Number of qubits (>= 2).
        
    Returns:
        Target state vector for Toffoli gate output.
    """
    if n_qubits < 2:
        raise ValueError("n-controlled Toffoli requires at least 2 qubits (n >= 2)")
    
    circuit, qubits = create_toffoli_circuit_and_qubits(n_qubits)
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit, qubit_order=qubits)
    return result.final_state_vector


def get_default_target_state(n_qubits: int) -> np.ndarray:
    """
    Get the default target state for quantum architecture search experiments.
    
    This is the primary entry point for obtaining target states in experiments.
    Uses n-controlled Toffoli gates as the default target:
    - 2 qubits: CNOT target
    - 3 qubits: Toffoli (CCNOT) target  
    - 4 qubits: CCCNOT target
    - etc.
    
    Note: For backward compatibility with legacy experiments, use get_ghz_state()
    explicitly if GHZ state preparation is needed.
    
    Args:
        n_qubits: Number of qubits (>= 2).
        
    Returns:
        Target state vector for the default compilation target.
    """
    return get_toffoli_state(n_qubits)


def get_ghz_state(n_qubits):
    """
    Generates the GHZ state vector by simulating the canonical preparation circuit.
    
    LEGACY/OPTIONAL: This function is retained for backward compatibility with
    existing experiments. For new experiments, use get_toffoli_state() or
    get_default_target_state() instead, which use n-controlled Toffoli gates
    as the default compilation target.
    
    This ensures the target state is perfectly consistent with the circuit logic.
    The state is (|0...0> + |1...1>) / sqrt(2).
    
    Args:
        n_qubits: Number of qubits.
        
    Returns:
        GHZ state vector.
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
