import numpy as np
from diagnostics import run_basic_sanity_checks, compute_fidelities

RTOL = 1e-12
ATOL = 1e-12

def test_identity_identity_is_perfect():
    res = run_basic_sanity_checks()['identity_identity']
    assert np.isclose(res['raw_trace_f'], 1.0, rtol=RTOL, atol=ATOL)
    assert np.isclose(res['phase_corrected_trace_f'], 1.0, rtol=RTOL, atol=ATOL)

def test_toffoli_toffoli_is_perfect():
    res = run_basic_sanity_checks()['toffoli_toffoli']
    assert np.isclose(res['raw_trace_f'], 1.0, rtol=RTOL, atol=ATOL)
    assert np.isclose(res['phase_corrected_trace_f'], 1.0, rtol=RTOL, atol=ATOL)

def test_toffoli_vs_phase_equivalent_is_phase_corrected():
    res = run_basic_sanity_checks()['toffoli_toffoli_phase']
    # raw trace may be <1 due to phase, but phase-corrected must be ~1
    assert res['raw_trace_f'] < 1.0
    assert np.isclose(res['phase_corrected_trace_f'], 1.0, rtol=RTOL, atol=ATOL)
