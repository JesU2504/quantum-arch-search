"""
Tests for comparison diagnostics module.

These tests verify that the fidelity computation functions work correctly
for the DRL vs EA comparison workspace.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add comparison package to path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from comparison.diagnostics import (
    toffoli_matrix,
    trace_overlap_fidelity,
    entanglement_fidelity_from_trace,
    average_gate_fidelity_from_entanglement,
    remove_global_phase,
    compute_fidelities,
    run_basic_sanity_checks,
)

# Tolerances for numerical comparisons
RTOL = 1e-12
ATOL = 1e-12


class TestToffoliMatrix:
    """Tests for the toffoli_matrix function."""

    def test_shape(self):
        """Toffoli matrix should be 8x8."""
        U = toffoli_matrix()
        assert U.shape == (8, 8)

    def test_unitarity(self):
        """Toffoli matrix should be unitary."""
        U = toffoli_matrix()
        identity = np.eye(8, dtype=complex)
        product = U @ U.conj().T
        assert np.allclose(product, identity, rtol=RTOL, atol=ATOL)

    def test_swaps_correct_indices(self):
        """Toffoli should swap |110> and |111>."""
        U = toffoli_matrix()
        # Index 6 = |110>, Index 7 = |111>
        assert U[6, 7] == 1
        assert U[7, 6] == 1
        assert U[6, 6] == 0
        assert U[7, 7] == 0


class TestTraceOverlapFidelity:
    """Tests for trace_overlap_fidelity function."""

    def test_identity_with_self(self):
        """Identity vs identity should give fidelity 1.0."""
        I = np.eye(4, dtype=complex)
        f, trace = trace_overlap_fidelity(I, I)
        assert np.isclose(f, 1.0, rtol=RTOL, atol=ATOL)

    def test_orthogonal_unitaries(self):
        """Orthogonal unitaries should give fidelity < 1."""
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
        f, trace = trace_overlap_fidelity(I, X)
        # Trace of X is 0, so fidelity should be 0
        assert np.isclose(f, 0.0, rtol=RTOL, atol=ATOL)


class TestEntanglementFidelity:
    """Tests for entanglement fidelity calculation."""

    def test_perfect_fidelity(self):
        """Perfect overlap should give Fe = 1.0."""
        d = 4
        trace_val = d  # Perfect trace
        Fe = entanglement_fidelity_from_trace(trace_val, d)
        assert np.isclose(Fe, 1.0, rtol=RTOL, atol=ATOL)

    def test_zero_fidelity(self):
        """Zero trace should give Fe = 0.0."""
        d = 4
        trace_val = 0
        Fe = entanglement_fidelity_from_trace(trace_val, d)
        assert np.isclose(Fe, 0.0, rtol=RTOL, atol=ATOL)


class TestAverageGateFidelity:
    """Tests for average gate fidelity calculation."""

    def test_perfect_entanglement_fidelity(self):
        """Fe = 1.0 should give Favg = 1.0."""
        d = 8
        Favg = average_gate_fidelity_from_entanglement(1.0, d)
        assert np.isclose(Favg, 1.0, rtol=RTOL, atol=ATOL)

    def test_formula(self):
        """Check Favg = (d*Fe + 1)/(d + 1) formula."""
        d = 4
        Fe = 0.5
        expected = (d * Fe + 1) / (d + 1)  # (4*0.5 + 1)/5 = 3/5 = 0.6
        Favg = average_gate_fidelity_from_entanglement(Fe, d)
        assert np.isclose(Favg, expected, rtol=RTOL, atol=ATOL)


class TestRemoveGlobalPhase:
    """Tests for global phase removal."""

    def test_phase_removal(self):
        """Should correctly remove global phase."""
        U = np.eye(4, dtype=complex)
        phi = 0.5
        U_phased = U * np.exp(1j * phi)

        U_corrected, alpha = remove_global_phase(U, U_phased)

        # After correction, should be close to identity
        assert np.allclose(U_corrected, U, rtol=1e-6, atol=1e-6)


class TestComputeFidelities:
    """Tests for the comprehensive compute_fidelities function."""

    def test_returns_dict(self):
        """Should return a dictionary with expected keys."""
        U = np.eye(4, dtype=complex)
        result = compute_fidelities(U, U)

        expected_keys = [
            'd', 'raw_trace_f', 'trace_complex', 'Fe', 'Favg',
            'global_phase_alpha', 'phase_corrected_trace_f',
            'phase_corrected_Fe', 'phase_corrected_Favg'
        ]
        for key in expected_keys:
            assert key in result

    def test_perfect_match(self):
        """Identical unitaries should give all fidelities = 1.0."""
        U = toffoli_matrix()
        result = compute_fidelities(U, U)

        assert np.isclose(result['raw_trace_f'], 1.0, rtol=RTOL, atol=ATOL)
        assert np.isclose(result['phase_corrected_trace_f'], 1.0, rtol=RTOL, atol=ATOL)
        assert np.isclose(result['Fe'], 1.0, rtol=RTOL, atol=ATOL)
        assert np.isclose(result['Favg'], 1.0, rtol=RTOL, atol=ATOL)


class TestSanityChecks:
    """Tests using the run_basic_sanity_checks function."""

    def test_identity_identity_is_perfect(self):
        """Identity vs identity should be perfect fidelity."""
        res = run_basic_sanity_checks()['identity_identity']
        assert np.isclose(res['raw_trace_f'], 1.0, rtol=RTOL, atol=ATOL)
        assert np.isclose(res['phase_corrected_trace_f'], 1.0, rtol=RTOL, atol=ATOL)

    def test_toffoli_toffoli_is_perfect(self):
        """Toffoli vs toffoli should be perfect fidelity."""
        res = run_basic_sanity_checks()['toffoli_toffoli']
        assert np.isclose(res['raw_trace_f'], 1.0, rtol=RTOL, atol=ATOL)
        assert np.isclose(res['phase_corrected_trace_f'], 1.0, rtol=RTOL, atol=ATOL)

    def test_toffoli_vs_phase_equivalent_is_phase_corrected(self):
        """Toffoli vs phase-shifted toffoli should be corrected to ~1.0."""
        res = run_basic_sanity_checks()['toffoli_toffoli_phase']
        # Raw trace may be < 1 due to phase, but phase-corrected must be ~1
        assert res['raw_trace_f'] < 1.0
        assert np.isclose(res['phase_corrected_trace_f'], 1.0, rtol=RTOL, atol=ATOL)

    def test_toffoli_identity_mismatch(self):
        """Toffoli vs identity should show mismatch."""
        res = run_basic_sanity_checks()['toffoli_identity']
        # Identity is not toffoli, so fidelity should be < 1
        assert res['raw_trace_f'] < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
