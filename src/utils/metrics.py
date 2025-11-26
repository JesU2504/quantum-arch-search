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
  - Fidelity computation using density matrix simulation
  - CNOT gate counting
  - Circuit depth analysis
  - Noise robustness metrics
"""

import numpy as np
import cirq


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
    return np.abs(overlap) ** 2


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
    # TODO: Simulate circuit to get output state
    # TODO: Compute overlap with target state
    # TODO: Return fidelity
    pass


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
    # TODO: Iterate through circuit operations
    # TODO: Count CNotPowGate instances
    # TODO: Return count
    pass


def count_gates(circuit):
    """
    Count total number of gates in a circuit.

    Args:
        circuit: Cirq circuit to analyze.

    Returns:
        Total gate count.
    """
    # TODO: Return len(list(circuit.all_operations()))
    pass


def get_circuit_depth(circuit):
    """
    Get the depth (number of moments) of a circuit.

    Args:
        circuit: Cirq circuit to analyze.

    Returns:
        Circuit depth.
    """
    # TODO: Return len(circuit)
    pass


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
    # TODO: Compute all metrics
    # TODO: Return as dictionary
    pass


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
