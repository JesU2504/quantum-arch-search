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
