"""
Metrics utilities for Quantum Architecture Search.

See ExpPlan.md, Implementation notes:
  - Logging: Log `cnot_count` explicitly (not just `len(circuit)`)
  - Include it in env info, e.g., `{ 'cnot_count': ... }`

This module provides standardized metric computation for:
  - Fidelity: State overlap with target
  - Gate counts: Total gates, CNOT count
  - Evaluation: Combined circuit quality metrics

TODO: Implement the following:
  - CNOT gate counting
  - Circuit depth analysis
  - Noise robustness metrics

DONE:
  - Fidelity computation using density matrix simulation (mixed_state_fidelity)
"""

import numpy as np
import cirq


def ghz_circuit(n_qubits):
    """
    Create a perfect GHZ circuit for n qubits.
    
    LEGACY/OPTIONAL: This function is retained for backward compatibility.
    For new experiments, use toffoli_circuit() instead.

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


def toffoli_circuit(n_qubits):
    """
    Create a circuit implementing the n-controlled Toffoli gate.
    
    This is the default circuit for n-qubit experiments. For n qubits,
    this implements an (n-1)-controlled NOT gate where:
    - Qubits 0 to n-2 are control qubits
    - Qubit n-1 is the target qubit
    
    The circuit prepares all qubits in |1> (via X gates) and then applies
    the multi-controlled NOT, resulting in the target state |11...10>.
    
    - n=2: CNOT preparation circuit
    - n=3: Toffoli (CCNOT) preparation circuit
    - n=4: CCCNOT preparation circuit
    - etc.

    Args:
        n_qubits: Number of qubits (>= 2).

    Returns:
        Tuple of (circuit, qubits) where circuit is a Cirq circuit
        and qubits is the list of qubits used.
    """
    if n_qubits < 2:
        raise ValueError("n-controlled Toffoli requires at least 2 qubits")
    
    qubits = list(cirq.LineQubit.range(n_qubits))
    circuit = cirq.Circuit()
    
    # Step 1: Prepare all qubits in |1> state
    for q in qubits:
        circuit.append(cirq.X(q))
    
    # Step 2: Apply the n-controlled NOT gate
    controls = qubits[:-1]
    target = qubits[-1]
    
    if n_qubits == 2:
        circuit.append(cirq.CNOT(controls[0], target))
    elif n_qubits == 3:
        circuit.append(cirq.TOFFOLI(controls[0], controls[1], target))
    else:
        controlled_x = cirq.X(target).controlled_by(*controls)
        circuit.append(controlled_x)
    
    return circuit, qubits


def ideal_toffoli_state(n_qubits):
    """
    Compute the ideal n-controlled Toffoli output state.
    
    This is the default target state for n-qubit experiments.
    Returns the output of applying an n-controlled NOT to |11...1>,
    which is |11...10> (target qubit flips from 1 to 0 when all controls are 1).

    Args:
        n_qubits: Number of qubits (>= 2).

    Returns:
        State vector as numpy array of shape (2^n_qubits,).
    """
    if n_qubits < 2:
        raise ValueError("n-controlled Toffoli requires at least 2 qubits")
    
    circuit, qubits = toffoli_circuit(n_qubits)
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit, qubit_order=qubits)
    return result.final_state_vector


def ideal_ghz_state(n_qubits):
    """
    Compute the ideal GHZ state vector.
    
    LEGACY/OPTIONAL: This function is retained for backward compatibility.
    For new experiments, use ideal_toffoli_state() instead.

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
    Comprehensive circuit evaluation.

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
    """
    if qubits is None:
        qubits = sorted(circuit.all_qubits())
    return {
        "fidelity": compute_fidelity(circuit, target_state, qubits),
        "cnot_count": count_cnots(circuit),
        "total_gates": count_gates(circuit),
        "depth": get_circuit_depth(circuit),
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
