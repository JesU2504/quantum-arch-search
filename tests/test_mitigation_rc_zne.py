import numpy as np
import cirq

from experiments.analysis.compare_circuits import (
    RC_ZNE_DEFAULT_SCALES,
    MITIGATION_NONE,
    MITIGATION_RC_ZNE,
    _richardson_extrapolate,
    _zero_noise_extrapolate,
    _scale_noise_kwargs,
    evaluate_multi_gate_attacks,
)


def _build_target_state(circuit: cirq.Circuit, qubits: list[cirq.Qid]) -> np.ndarray:
    """Utility helper to obtain the ideal state vector for a small circuit."""
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit, qubit_order=qubits)
    return result.final_state_vector


def test_richardson_extrapolate_linear_recovery():
    scales = [1.0, 2.0, 3.0]
    # Linear model: f(s) = 1 - 0.2 * s, so zero-noise intercept should be exactly 1.0
    values = [0.8, 0.6, 0.4]
    estimate = _richardson_extrapolate(scales, values)
    assert np.isclose(estimate, 1.0, atol=1e-8)


def test_zero_noise_extrapolate_quadratic():
    scales = [1.0, 1.5, 2.5]
    values = [1 - 0.15 * s + 0.03 * (s ** 2) for s in scales]
    estimate = _zero_noise_extrapolate(scales, values, fit="quadratic")
    assert np.isclose(estimate, 1.0, atol=1e-8)


def test_scale_noise_kwargs_amplitude_damping_growth():
    base_gamma = 0.1
    scaled = _scale_noise_kwargs(
        attack_mode="amplitude_damping",
        scale=2.0,
        epsilon_overrot=0.1,
        p_x=0.0,
        p_y=0.0,
        p_z=0.0,
        gamma_amp=base_gamma,
        gamma_phase=0.0,
        p_readout=0.0,
    )
    assert 0.0 < scaled["gamma_amp"] < 1.0
    assert scaled["gamma_amp"] > base_gamma


def test_rc_zne_mitigation_improves_over_rotation_mean():
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.rx(np.pi / 4).on(qubit))
    target_state = _build_target_state(circuit, [qubit])
    epsilon = 0.2
    samples = 12
    rng_seed = 1234

    baseline = evaluate_multi_gate_attacks(
        circuit,
        saboteur_agent=None,
        target_state=target_state,
        n_qubits=1,
        samples=samples,
        saboteur_budget=1,
        rng=np.random.default_rng(rng_seed),
        attack_mode="over_rotation",
        epsilon_overrot=epsilon,
        mitigation_mode=MITIGATION_NONE,
    )

    mitigated = evaluate_multi_gate_attacks(
        circuit,
        saboteur_agent=None,
        target_state=target_state,
        n_qubits=1,
        samples=samples,
        saboteur_budget=1,
        rng=np.random.default_rng(rng_seed),
        attack_mode="over_rotation",
        epsilon_overrot=epsilon,
        mitigation_mode=MITIGATION_RC_ZNE,
        rc_zne_scales=RC_ZNE_DEFAULT_SCALES,
    )

    clean = baseline["clean_fidelity"]
    assert 0.0 <= clean <= 1.0
    assert 0.0 <= baseline["mean_attacked"] <= 1.0
    assert 0.0 <= mitigated["mean_attacked"] <= 1.0
    # Mitigated fidelity should not be farther from the clean value than the baseline
    baseline_gap = abs(clean - baseline["mean_attacked"])
    mitigated_gap = abs(clean - mitigated["mean_attacked"])
    assert mitigated_gap <= baseline_gap + 1e-6


def test_rc_zne_reps_report_stats():
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.rx(np.pi / 4).on(qubit))
    target_state = _build_target_state(circuit, [qubit])

    mitigated = evaluate_multi_gate_attacks(
        circuit,
        saboteur_agent=None,
        target_state=target_state,
        n_qubits=1,
        samples=4,
        saboteur_budget=1,
        rng=np.random.default_rng(42),
        attack_mode="over_rotation",
        epsilon_overrot=0.15,
        mitigation_mode=MITIGATION_RC_ZNE,
        rc_zne_scales=RC_ZNE_DEFAULT_SCALES,
        rc_zne_reps=3,
    )

    assert mitigated.get("rc_zne_reps") == 3
    assert mitigated.get("rc_zne_scales_used") == list(RC_ZNE_DEFAULT_SCALES)
    assert len(mitigated.get("rc_zne_scale_mean", [])) == len(RC_ZNE_DEFAULT_SCALES)
    assert len(mitigated.get("rc_zne_scale_std", [])) == len(RC_ZNE_DEFAULT_SCALES)
