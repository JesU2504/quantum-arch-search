"""
Metrics utilities for Quantum Architecture Search.

See ExpPlan.md, Implementation notes:
  - Logging: Log `cnot_count` explicitly (not just `len(circuit)`)
  - Include it in env info, e.g., `{ 'cnot_count': ... }`

This module provides standardized metric computation for:
  - Fidelity: State overlap with target
  - Gate counts: Total gates, CNOT count
  - Evaluation: Combined circuit quality metrics
  - Full basis-sweep fidelity for gate synthesis verification

TODO: Implement the following:
  - CNOT gate counting
  - Circuit depth analysis
  - Noise robustness metrics

DONE:
  - Fidelity computation using density matrix simulation (mixed_state_fidelity)
  - Full basis-sweep fidelity for Toffoli/n-controlled-NOT gate synthesis
"""

import numpy as np
import cirq
from typing import Callable, Optional, Sequence


def ghz_circuit(n_qubits):
    """
    Create a perfect GHZ circuit for n qubits.

    Creates the state (|00...0> + |11...1>) / sqrt(2)
    using a Hadamard on the first qubit followed by CNOTs.

    Args:
        n_qubits: Number of qubits for the GHZ state.

    Returns:
        Tuple of (circuit, qubits) where circuit is a Cirq circuit
        and qubits is the list of qubits used.
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    # Apply Hadamard to first qubit
    circuit.append(cirq.H(qubits[0]))
    # Apply CNOTs in chain
    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit, qubits


def ideal_ghz_state(n_qubits):
    """
    Compute the ideal GHZ state vector.

    Returns (|00...0> + |11...1>) / sqrt(2)

    Args:
        n_qubits: Number of qubits.

    Returns:
        State vector as numpy array of shape (2^n_qubits,).
    """
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2)  # |00...0>
    state[-1] = 1.0 / np.sqrt(2)  # |11...1>
    return state


def state_fidelity(state1, state2):
    """
    Compute fidelity between two pure state vectors.

    Uses F = |<state1|state2>|^2

    Args:
        state1: First state vector (numpy array).
        state2: Second state vector (numpy array).

    Returns:
        Fidelity value in [0, 1].
    """
    overlap = np.vdot(state1, state2)
    return float(np.abs(overlap) ** 2)


def mixed_state_fidelity(pure_state, density_matrix):
    """
    Compute fidelity between a pure state and a density matrix.

    Uses F = <psi|rho|psi> for a pure state |psi> and density matrix rho.
    This is the correct fidelity metric for evaluating noisy quantum
    circuit outputs against an ideal target state.

    Args:
        pure_state: Target pure state vector (numpy array of shape (d,)).
        density_matrix: Noisy output density matrix (numpy array of shape (d, d)).

    Returns:
        Fidelity value in [0, 1].

    Raises:
        ValueError: If dimensions of pure_state and density_matrix don't match.

    Example:
        >>> # Create a simple 2-qubit pure state |00>
        >>> pure_state = np.array([1, 0, 0, 0], dtype=complex)
        >>> # Create a noisy density matrix (mixed state)
        >>> rho = np.diag([0.9, 0.05, 0.03, 0.02])
        >>> fidelity = mixed_state_fidelity(pure_state, rho)
        >>> print(f"Fidelity: {fidelity:.3f}")  # Should be 0.9
    """
    pure_state = np.asarray(pure_state, dtype=np.complex128)
    density_matrix = np.asarray(density_matrix, dtype=np.complex128)

    # Validate dimensions
    if pure_state.shape[0] != density_matrix.shape[0]:
        raise ValueError(
            f"Dimension mismatch: pure_state has dimension {pure_state.shape[0]}, "
            f"but density_matrix has shape {density_matrix.shape}"
        )
    if density_matrix.ndim != 2 or density_matrix.shape[0] != density_matrix.shape[1]:
        raise ValueError(
            f"density_matrix must be a square matrix, got shape {density_matrix.shape}"
        )

    # Compute F = <psi|rho|psi>
    # rho @ pure_state gives rho|psi>
    # np.vdot(pure_state, ...) gives <psi|...>
    fidelity = np.real(np.vdot(pure_state, density_matrix @ pure_state))

    return float(fidelity)


def simulate_circuit(circuit, qubits):
    """
    Simulate a circuit and return the final state vector.

    Args:
        circuit: Cirq circuit to simulate.
        qubits: Qubits used in the circuit (for ordering).

    Returns:
        Final state vector as numpy array.
    """
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit, qubit_order=qubits)
    return result.final_state_vector


def compute_fidelity(circuit, target_state, qubits=None):
    """
    Compute fidelity between circuit output and target state.

    Uses F = |<target|output>|^2 for pure states.

    Args:
        circuit: Cirq circuit to evaluate.
        target_state: Target state vector (numpy array).
        qubits: Qubit ordering for simulation.

    Returns:
        Fidelity value in [0, 1].
    """
    if qubits is None:
        qubits = sorted(circuit.all_qubits())
    output_state = simulate_circuit(circuit, qubits)
    return state_fidelity(output_state, target_state)


def count_cnots(circuit):
    """
    Count the number of CNOT gates in a circuit.

    CNOT count is a key metric for circuit complexity and
    noise sensitivity on NISQ devices.

    Args:
        circuit: Cirq circuit to analyze.

    Returns:
        Number of CNOT gates.
    """
    count = 0
    for op in circuit.all_operations():
        if isinstance(op.gate, (cirq.CNotPowGate, cirq.CXPowGate)):
            count += 1
        elif isinstance(op.gate, cirq.ControlledGate):
            # Check if it's a controlled X (CNOT)
            if isinstance(op.gate.sub_gate, cirq.XPowGate):
                count += 1
    return count


def count_rotation_gates(circuit):
    """
    Count parameterized rotation gates (Rx, Ry, Rz) in a circuit.
    
    Rotation gates are key for expressiveness in VQE and other
    variational quantum algorithms.
    
    Args:
        circuit: Cirq circuit to analyze.
    
    Returns:
        Dictionary with counts:
          - rx_count: Number of Rx gates
          - ry_count: Number of Ry gates  
          - rz_count: Number of Rz gates
          - total_rotations: Total rotation gate count
    """
    counts = {'rx_count': 0, 'ry_count': 0, 'rz_count': 0, 'total_rotations': 0}
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.Rx):
            counts['rx_count'] += 1
            counts['total_rotations'] += 1
        elif isinstance(op.gate, cirq.Ry):
            counts['ry_count'] += 1
            counts['total_rotations'] += 1
        elif isinstance(op.gate, cirq.Rz):
            counts['rz_count'] += 1
            counts['total_rotations'] += 1
    return counts


def get_rotation_angles(circuit):
    """
    Extract rotation angles from all rotation gates in a circuit.
    
    Args:
        circuit: Cirq circuit to analyze.
    
    Returns:
        List of dictionaries with gate info:
          - gate_type: 'Rx', 'Ry', or 'Rz'
          - qubit: The qubit the gate acts on
          - angle: The rotation angle in radians
          - index: Position in the circuit
    """
    angles = []
    for i, op in enumerate(circuit.all_operations()):
        gate = op.gate
        if isinstance(gate, cirq.Rx):
            angles.append({
                'gate_type': 'Rx',
                'qubit': str(op.qubits[0]),
                'angle': float(gate.exponent * np.pi),
                'index': i
            })
        elif isinstance(gate, cirq.Ry):
            angles.append({
                'gate_type': 'Ry',
                'qubit': str(op.qubits[0]),
                'angle': float(gate.exponent * np.pi),
                'index': i
            })
        elif isinstance(gate, cirq.Rz):
            angles.append({
                'gate_type': 'Rz',
                'qubit': str(op.qubits[0]),
                'angle': float(gate.exponent * np.pi),
                'index': i
            })
    return angles


def count_gates(circuit):
    """
    Count total number of gates in a circuit.

    Args:
        circuit: Cirq circuit to analyze.

    Returns:
        Total gate count.
    """
    return len(list(circuit.all_operations()))


def get_circuit_depth(circuit):
    """
    Get the depth (number of moments) of a circuit.

    Args:
        circuit: Cirq circuit to analyze.

    Returns:
        Circuit depth.
    """
    return len(circuit)


def evaluate_circuit(circuit, target_state, qubits=None):
    """
    Comprehensive circuit evaluation including rotation gate metrics.

    Args:
        circuit: Cirq circuit to evaluate.
        target_state: Target state vector.
        qubits: Qubit ordering for simulation.

    Returns:
        Dict with metrics:
          - fidelity: State fidelity
          - cnot_count: Number of CNOTs
          - total_gates: Total gate count
          - depth: Circuit depth
          - rotation_counts: Dict with rx_count, ry_count, rz_count, total_rotations
          - rotation_angles: List of rotation angle info dicts
    """
    if qubits is None:
        qubits = sorted(circuit.all_qubits())
    
    rotation_counts = count_rotation_gates(circuit)
    
    return {
        "fidelity": compute_fidelity(circuit, target_state, qubits),
        "cnot_count": count_cnots(circuit),
        "total_gates": count_gates(circuit),
        "depth": get_circuit_depth(circuit),
        "rotation_counts": rotation_counts,
        "rotation_angles": get_rotation_angles(circuit),
    }


def fidelity_retention_ratio(fidelity_noisy, fidelity_clean):
    """
    Compute fidelity retention ratio under noise.

    See ExpPlan.md, Part 2 (Cross-noise evaluation):
      Metric: Fidelity retention ratio F_noisy / F_clean

    Args:
        fidelity_noisy: Fidelity under noise.
        fidelity_clean: Fidelity without noise.

    Returns:
        Retention ratio in [0, 1].
    """
    if fidelity_clean == 0:
        return 0.0
    return fidelity_noisy / fidelity_clean


def state_energy(state_vector, hamiltonian_matrix):
    """
    Compute the energy expectation value of a quantum state.

    Computes <psi|H|psi> for a given state vector and Hamiltonian matrix.

    Args:
        state_vector: Complex numpy array of shape (2^n,) representing the
            quantum state |psi>.
        hamiltonian_matrix: Complex numpy array of shape (2^n, 2^n)
            representing the Hamiltonian operator H.

    Returns:
        Real energy expectation value in Hartree.
    """
    # Normalize the state vector
    state_vector = np.asarray(state_vector, dtype=complex)
    norm = np.linalg.norm(state_vector)
    if norm > 0:
        state_vector = state_vector / norm

    # Compute <psi|H|psi>
    h_psi = hamiltonian_matrix @ state_vector
    energy = np.real(np.vdot(state_vector, h_psi))
    return float(energy)


# =============================================================================
# Full Basis-Sweep Fidelity for Gate Synthesis Verification
# =============================================================================

def computational_basis_state(index: int, n_qubits: int) -> np.ndarray:
    """
    Create a computational basis state |index> for n qubits.

    Args:
        index: Integer index representing the basis state (0 to 2^n - 1).
        n_qubits: Number of qubits.

    Returns:
        State vector as numpy array of shape (2^n_qubits,).

    Example:
        >>> computational_basis_state(0, 2)  # |00>
        array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
        >>> computational_basis_state(3, 2)  # |11>
        array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])
    """
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[index] = 1.0
    return state


def toffoli_truth_table(n_controls: int) -> Callable[[int], int]:
    """
    Generate the truth table function for an n-controlled NOT gate (Toffoli generalization).

    The n-controlled NOT gate flips the target qubit if and only if all control
    qubits are in state |1>. Qubit ordering: control qubits first, target qubit last.

    For example, a 2-controlled NOT (Toffoli/CCNOT) with 3 qubits:
        |000> -> |000>  (controls 00, no flip)
        |001> -> |001>  (controls 00, no flip)
        |010> -> |010>  (controls 01, no flip)
        |011> -> |011>  (controls 01, no flip)
        |100> -> |100>  (controls 10, no flip)
        |101> -> |101>  (controls 10, no flip)
        |110> -> |111>  (controls 11, flip!)
        |111> -> |110>  (controls 11, flip!)

    Args:
        n_controls: Number of control qubits. Total qubits = n_controls + 1.

    Returns:
        A function that maps input basis state index to output basis state index.
    """
    def truth_fn(input_index: int) -> int:
        # Check if all control bits are 1 (ignore target bit which is LSB)
        controls_value = input_index >> 1
        if controls_value == (1 << n_controls) - 1:
            # All controls are on, flip the target (LSB)
            return input_index ^ 1
        return input_index

    return truth_fn


def full_basis_fidelity(
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.Qid],
    truth_table_fn: Callable[[int], int],
    noise_model: Optional[cirq.NoiseModel] = None,
) -> float:
    """
    Compute average fidelity over all computational basis inputs.

    This function evaluates a candidate circuit for gate synthesis by:
    1. Enumerating all 2^n computational basis input states
    2. For each input: preparing the input state, simulating the circuit
    3. Computing fidelity between simulated output and expected output
    4. Averaging fidelities over all inputs

    Args:
        circuit: The candidate circuit to evaluate.
        qubits: Qubit ordering for simulation.
        truth_table_fn: Function mapping input basis state index to expected
            output basis state index (encodes the gate's classical truth table).
        noise_model: Optional noise model to apply during simulation.

    Returns:
        Average fidelity over all basis inputs, in range [0, 1].

    Example:
        >>> # Evaluate a circuit meant to implement Toffoli
        >>> from src.utils.metrics import full_basis_fidelity, toffoli_truth_table
        >>> qubits = cirq.LineQubit.range(3)
        >>> circuit = cirq.Circuit(cirq.TOFFOLI(*qubits))
        >>> truth_fn = toffoli_truth_table(n_controls=2)
        >>> fidelity = full_basis_fidelity(circuit, qubits, truth_fn)
        >>> print(f"Fidelity: {fidelity:.4f}")  # Should be 1.0
    """
    n_qubits = len(qubits)
    dim = 2 ** n_qubits
    simulator = cirq.DensityMatrixSimulator()

    fidelities = []
    for input_idx in range(dim):
        # Create input basis state
        input_state = computational_basis_state(input_idx, n_qubits)

        # Create preparation circuit for input state
        prep_circuit = _prepare_basis_state_circuit(input_idx, qubits)

        # Combine preparation + candidate circuit
        full_circuit = prep_circuit + circuit

        # Apply noise if specified
        if noise_model is not None:
            full_circuit = full_circuit.with_noise(noise_model)

        # Simulate to get output density matrix
        result = simulator.simulate(full_circuit, qubit_order=qubits)
        output_rho = result.final_density_matrix
        # Ensure Hermiticity
        output_rho = 0.5 * (output_rho + np.conj(output_rho).T)

        # Get expected output state from truth table
        expected_output_idx = truth_table_fn(input_idx)
        expected_output_state = computational_basis_state(expected_output_idx, n_qubits)

        # Compute fidelity F = <psi|rho|psi>
        fidelity = float(np.real(np.vdot(
            expected_output_state, output_rho @ expected_output_state
        )))
        fidelities.append(fidelity)

    return float(np.mean(fidelities))


def _prepare_basis_state_circuit(
    index: int, qubits: Sequence[cirq.Qid]
) -> cirq.Circuit:
    """
    Create a circuit to prepare computational basis state |index>.

    Applies X gates to qubits that should be |1> in the basis state.
    Uses Cirq's convention where qubits[0] is the MSB of the state index.

    Args:
        index: Integer index representing the desired basis state.
        qubits: List of qubits to prepare.

    Returns:
        Cirq circuit that prepares |index> from |0...0>.
    """
    n_qubits = len(qubits)
    ops = []
    for i in range(n_qubits):
        # In Cirq, qubits[i] corresponds to bit (n_qubits - 1 - i) of the index
        # i.e., qubits[0] is the MSB of the state index
        bit_position = n_qubits - 1 - i
        if (index >> bit_position) & 1:
            ops.append(cirq.X(qubits[i]))
    return cirq.Circuit(ops)


def full_basis_fidelity_toffoli(
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.Qid],
    n_controls: int,
    noise_model: Optional[cirq.NoiseModel] = None,
) -> float:
    """
    Compute average fidelity for an n-controlled NOT (Toffoli) gate synthesis.

    Convenience wrapper around full_basis_fidelity for Toffoli-family gates.

    Args:
        circuit: The candidate circuit to evaluate.
        qubits: Qubit ordering for simulation (controls first, target last).
        n_controls: Number of control qubits (2 for standard Toffoli/CCNOT).
        noise_model: Optional noise model to apply during simulation.

    Returns:
        Average fidelity over all basis inputs, in range [0, 1].

    Example:
        >>> qubits = cirq.LineQubit.range(3)
        >>> circuit = cirq.Circuit(cirq.TOFFOLI(*qubits))
        >>> fidelity = full_basis_fidelity_toffoli(circuit, qubits, n_controls=2)
        >>> print(f"Fidelity: {fidelity:.4f}")  # Should be 1.0
    """
    truth_fn = toffoli_truth_table(n_controls)
    return full_basis_fidelity(circuit, qubits, truth_fn, noise_model)
