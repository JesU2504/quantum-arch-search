#!/usr/bin/env python3
"""
Diagnostic utilities for unitary fidelity troubleshooting.

Provides programmatic functions for unit tests and a CLI for manual debugging.
This module is designed for the comparison workspace to help analyze and compare
fidelity metrics between DRL-based (arXiv:2407.20147) and coevolutionary agents.
"""

import json
import numpy as np


def toffoli_matrix():
    """
    Generate the 3-qubit Toffoli (CCNOT) matrix.

    Returns:
        np.ndarray: 8x8 unitary matrix in computational basis order
        |000>,|001>,...,|111>. Toffoli flips target qubit (LSB) when both
        control qubits are 1: basis index 110 (6) <-> 111 (7) are swapped.
    """
    d = 8
    U = np.eye(d, dtype=complex)
    # Toffoli flips target qubit (LSB here) when both control qubits are 1:
    # basis index 110 (6) <-> 111 (7) are swapped.
    U[6, 6] = 0
    U[7, 7] = 0
    U[6, 7] = 1
    U[7, 6] = 1
    return U


def trace_overlap_fidelity(U_target, U):
    """
    Compute trace overlap fidelity between two unitaries.

    Args:
        U_target: Target unitary matrix
        U: Candidate unitary matrix

    Returns:
        tuple: (normalized_trace_fidelity, complex_trace_value)
    """
    d = U_target.shape[0]
    val = np.trace(U_target.conj().T @ U)
    return np.abs(val) / d, val


def entanglement_fidelity_from_trace(trace_val, d):
    """
    Compute entanglement fidelity from trace value.

    Args:
        trace_val: Complex trace value from trace_overlap_fidelity
        d: Dimension of the Hilbert space

    Returns:
        float: Entanglement fidelity Fe = |Tr(U†V)|² / d²
    """
    return (np.abs(trace_val) ** 2) / (d ** 2)


def average_gate_fidelity_from_entanglement(Fe, d):
    """
    Compute average gate fidelity from entanglement fidelity.

    Args:
        Fe: Entanglement fidelity
        d: Dimension of the Hilbert space

    Returns:
        float: Average gate fidelity Favg = (d*Fe + 1) / (d + 1)
    """
    return (d * Fe + 1) / (d + 1)


def remove_global_phase(U_target, U):
    """
    Remove global phase from candidate unitary to align with target.

    Args:
        U_target: Target unitary matrix
        U: Candidate unitary matrix

    Returns:
        tuple: (phase_corrected_unitary, phase_angle_alpha)
    """
    d = U_target.shape[0]
    V = U_target.conj().T @ U
    detV = np.linalg.det(V)
    # handle possible numerical issues if detV is zero-ish
    alpha = np.angle(detV) / d if detV != 0 else 0.0
    U_corrected = U * np.exp(-1j * alpha)
    return U_corrected, alpha


def compute_fidelities(U_target, U):
    """
    Compute comprehensive fidelity metrics for unitary comparison.

    This function computes raw and phase-corrected fidelity metrics including
    trace overlap fidelity, entanglement fidelity, and average gate fidelity.

    Args:
        U_target: Target unitary matrix
        U: Candidate unitary matrix

    Returns:
        dict: Dictionary containing:
            - d: Hilbert space dimension
            - raw_trace_f: Raw trace overlap fidelity
            - trace_complex: Complex trace value
            - Fe: Entanglement fidelity
            - Favg: Average gate fidelity
            - global_phase_alpha: Phase correction angle
            - phase_corrected_trace_f: Phase-corrected trace fidelity
            - phase_corrected_Fe: Phase-corrected entanglement fidelity
            - phase_corrected_Favg: Phase-corrected average gate fidelity
    """
    d = U_target.shape[0]
    f_trace, trace_val = trace_overlap_fidelity(U_target, U)
    Fe = entanglement_fidelity_from_trace(trace_val, d)
    Favg = average_gate_fidelity_from_entanglement(Fe, d)

    U_corr, alpha = remove_global_phase(U_target, U)
    f_trace_corr, trace_val_corr = trace_overlap_fidelity(U_target, U_corr)
    Fe_corr = entanglement_fidelity_from_trace(trace_val_corr, d)
    Favg_corr = average_gate_fidelity_from_entanglement(Fe_corr, d)

    return {
        'd': d,
        'raw_trace_f': float(f_trace),
        'trace_complex': complex(trace_val),
        'Fe': float(Fe),
        'Favg': float(Favg),
        'global_phase_alpha': float(alpha),
        'phase_corrected_trace_f': float(f_trace_corr),
        'phase_corrected_Fe': float(Fe_corr),
        'phase_corrected_Favg': float(Favg_corr),
    }


def run_basic_sanity_checks():
    """
    Run basic sanity checks on fidelity computations.

    Tests include:
        - identity vs identity (should be perfect fidelity ~1.0)
        - toffoli vs toffoli (should be perfect fidelity ~1.0)
        - toffoli vs identity (should show mismatch)
        - toffoli vs phase-shifted toffoli (phase correction should yield ~1.0)

    Returns:
        dict: Results dictionary with test names as keys and fidelity dicts as values
    """
    U_t = toffoli_matrix()
    identity = np.eye(U_t.shape[0], dtype=complex)

    results = {}
    results['identity_identity'] = compute_fidelities(identity, identity)
    results['toffoli_toffoli'] = compute_fidelities(U_t, U_t)
    results['toffoli_identity'] = compute_fidelities(U_t, identity)

    phi = 0.7
    U_phase = U_t * np.exp(1j * phi)
    results['toffoli_toffoli_phase'] = compute_fidelities(U_t, U_phase)

    return results


if __name__ == '__main__':
    res = run_basic_sanity_checks()
    # Output as JSON for programmatic consumption
    output = {}
    for k, v in res.items():
        # Convert complex to string for JSON serialization
        v_serializable = {}
        for kk, vv in v.items():
            if isinstance(vv, complex):
                v_serializable[kk] = {'real': vv.real, 'imag': vv.imag}
            else:
                v_serializable[kk] = vv
        output[k] = v_serializable

    print(json.dumps(output, indent=2))
