"""
Compare VQE circuits (baseline/robust/quantumnas) under deterministic noise by
reporting energy expectations. Mirrors the state-prep robustness compare but
uses energy instead of fidelity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import cirq

from experiments import config
from qas_gym.utils import (
    is_twirl_op,
    build_frame_twirled_noisy_circuit,
)
from experiments.analysis.compare_circuits import (
    _scale_noise_kwargs,
    _zero_noise_extrapolate,
    MITIGATION_NONE,
    MITIGATION_RC_ZNE,
    RC_ZNE_DEFAULT_SCALES,
)
from src.utils.metrics import state_energy
from src.utils.standard_hamiltonians import get_standard_hamiltonian


DETERMINISTIC_MODES = {"over_rotation", "asymmetric_noise", "amplitude_damping", "phase_damping", "readout"}


def _state_energy(circuit: cirq.Circuit, qubits: Sequence[cirq.Qid], hamiltonian: np.ndarray, frame: dict | None = None) -> float:
    sim = cirq.Simulator()
    res = sim.simulate(circuit, qubit_order=qubits)
    vec = res.final_state_vector
    return state_energy(vec, hamiltonian)


def evaluate_energy_under_noise(
    circuit: cirq.Circuit,
    hamiltonian: np.ndarray,
    *,
    attack_mode: str,
    epsilon_overrot: float = 0.1,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0,
    gamma_amp: float = 0.05,
    gamma_phase: float = 0.05,
    p_readout: float = 0.03,
    samples: int = 16,
    rng: np.random.Generator | None = None,
    mitigation_mode: str = MITIGATION_NONE,
    rc_zne_scales: Sequence[float] = RC_ZNE_DEFAULT_SCALES,
    rc_zne_fit: str = "linear",
    rc_zne_reps: int = 1,
) -> dict:
    qubits = sorted(list(circuit.all_qubits()))
    clean_e = _state_energy(circuit, qubits, hamiltonian)
    attacked = []
    rc_scales = tuple(float(s) for s in rc_zne_scales if s is not None and float(s) >= 0.0)
    if mitigation_mode == MITIGATION_RC_ZNE and len(rc_scales) < 2:
        rc_scales = RC_ZNE_DEFAULT_SCALES
    rc_zne_reps = max(1, int(rc_zne_reps) if rc_zne_reps is not None else 1)
    rng = rng or np.random.default_rng()

    for _ in range(samples):
        if attack_mode not in DETERMINISTIC_MODES:
            # Non-deterministic saboteur modes not supported here; skip
            attacked.append(clean_e)
            continue
        if mitigation_mode == MITIGATION_RC_ZNE:
            seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
            zne_vals = []
            for scale in rc_scales:
                reps = []
                for rep in range(rc_zne_reps):
                    local_rng = np.random.default_rng(seed + rep)
                    scaled = _scale_noise_kwargs(
                        attack_mode,
                        scale,
                        epsilon_overrot=epsilon_overrot,
                        p_x=p_x,
                        p_y=p_y,
                        p_z=p_z,
                        gamma_amp=gamma_amp,
                        gamma_phase=gamma_phase,
                        p_readout=p_readout,
                    )
                    noisy, frame = build_frame_twirled_noisy_circuit(
                        circuit,
                        local_rng,
                        attack_mode,
                        epsilon_overrot=scaled["epsilon_overrot"],
                        p_x=scaled["p_x"],
                        p_y=scaled["p_y"],
                        p_z=scaled["p_z"],
                        gamma_amp=scaled["gamma_amp"],
                        gamma_phase=scaled["gamma_phase"],
                        p_readout=scaled["p_readout"],
                    )
                    reps.append(_state_energy(noisy, qubits, hamiltonian, frame))
                zne_vals.append(float(np.mean(reps)))
            est = _zero_noise_extrapolate(rc_scales, zne_vals, fit=rc_zne_fit)
            attacked.append(float(est))
        else:
            noisy_ops = []
            for op in circuit.all_operations():
                noisy_ops.append(op)
                if is_twirl_op(op):
                    continue
                for q in op.qubits:
                    if attack_mode == "over_rotation":
                        noisy_ops.append(cirq.rx(epsilon_overrot).on(q))
                    elif attack_mode == "asymmetric_noise":
                        noisy_ops.append(cirq.asymmetric_depolarize(p_x=p_x, p_y=p_y, p_z=p_z).on(q))
                    elif attack_mode == "amplitude_damping":
                        noisy_ops.append(cirq.amplitude_damp(gamma_amp).on(q))
                    elif attack_mode == "phase_damping":
                        noisy_ops.append(cirq.phase_damp(gamma_phase).on(q))
                    elif attack_mode == "readout":
                        noisy_ops.append(cirq.bit_flip(p_readout).on(q))
            noisy = cirq.Circuit(noisy_ops)
            attacked.append(_state_energy(noisy, qubits, hamiltonian))

    arr = np.asarray(attacked, dtype=float)
    return {
        "clean_energy": float(clean_e),
        "mean_attacked": float(arr.mean()),
        "std_attacked": float(arr.std()),
        "samples": attacked,
    }


def _zz_chain_hamiltonian(n_qubits: int) -> np.ndarray:
    """Simple ZZ chain Hamiltonian sum_i Z_i Z_{i+1}."""
    import functools

    paulis = [np.array([[1, 0], [0, -1]], dtype=float)]
    I = np.eye(2, dtype=float)

    def kron_all(mats):
        return functools.reduce(np.kron, mats)

    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=float)
    for i in range(n_qubits - 1):
        mats = []
        for j in range(n_qubits):
            if j == i or j == i + 1:
                mats.append(paulis[0])
            else:
                mats.append(I)
        H += kron_all(mats)
    return H


def compare_vqe(
    base_results_dir: str,
    num_runs: int,
    molecule: str | None,
    attack_modes: list[str],
    samples: int = 16,
    mitigation_mode: str = MITIGATION_NONE,
    rc_zne_scales: Sequence[float] = RC_ZNE_DEFAULT_SCALES,
    rc_zne_fit: str = "linear",
    rc_zne_reps: int = 1,
    hamiltonian_json: str | None = None,
):
    base_results_dir = os.path.abspath(base_results_dir)
    per_mode = {m: {"vanilla": [], "robust": [], "quantumnas": []} for m in attack_modes}
    aggregated = {}
    h_mat = None
    # Load circuits first to infer qubit count
    run_dirs = [os.path.join(base_results_dir, f"run_{i}") for i in range(num_runs)]
    circuits_cache = []
    from qas_gym.utils import load_circuit
    for run_dir in run_dirs:
        vanilla = os.path.join(run_dir, "circuit_vanilla.json")
        robust = os.path.join(run_dir, "circuit_robust.json")
        qnas = os.path.join(run_dir, "circuit_quantumnas.json")
        if not (os.path.exists(vanilla) and os.path.exists(robust)):
            continue
        circ_v = load_circuit(vanilla)
        circ_r = load_circuit(robust)
        circ_q = load_circuit(qnas) if os.path.exists(qnas) else None
        circuits_cache.append((run_dir, circ_v, circ_r, circ_q))
    if not circuits_cache:
        raise SystemExit("No circuits found under compare dir.")

    # Hamiltonian selection
    if hamiltonian_json:
        with open(hamiltonian_json, "r") as f:
            h_mat = np.array(json.load(f), dtype=complex)
    else:
        if molecule:
            try:
                ham_info = get_standard_hamiltonian(molecule)
                h_mat = ham_info["matrix"]
            except Exception:
                h_mat = None
        if h_mat is None:
            # Fallback: ZZ chain Hamiltonian
            n_qubits = len(circuits_cache[0][1].all_qubits())
            h_mat = _zz_chain_hamiltonian(n_qubits)

    for i in range(num_runs):
        # find cached circuits
        match = next((c for c in circuits_cache if c[0].endswith(f"run_{i}")), None)
        if not match:
            continue
        run_dir, circ_v, circ_r, circ_q = match
        for mode in attack_modes:
            for label, circ in (("vanilla", circ_v), ("robust", circ_r), ("quantumnas", circ_q)):
                if circ is None:
                    continue
                res = evaluate_energy_under_noise(
                    circ,
                    h_mat,
                    attack_mode=mode,
                    samples=samples,
                    mitigation_mode=mitigation_mode if label == "robust" or label == "vanilla" or label == "quantumnas" else MITIGATION_NONE,
                    rc_zne_scales=rc_zne_scales,
                    rc_zne_fit=rc_zne_fit,
                    rc_zne_reps=rc_zne_reps,
                )
                res["variant"] = "mitigated" if mitigation_mode == MITIGATION_RC_ZNE else "untwirled"
                res["seed"] = i
                per_mode[mode][label].append(res)
    # Aggregate
    for mode, buckets in per_mode.items():
        agg_mode = {}
        for label in ("vanilla", "robust", "quantumnas"):
            entries = buckets[label]
            if not entries:
                continue
            vals = [e["mean_attacked"] for e in entries]
            agg_mode[label] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        aggregated[mode] = agg_mode
    out = {
        "metadata": {
            "molecule": molecule,
            "hamiltonian_source": hamiltonian_json or (molecule or "zz_chain"),
            "attack_modes": attack_modes,
            "samples_per_circuit": samples,
            "mitigation_mode": mitigation_mode,
            "rc_zne_scales": list(rc_zne_scales),
            "rc_zne_fit": rc_zne_fit,
            "rc_zne_reps": rc_zne_reps,
        },
        "per_mode": per_mode,
        "aggregated": aggregated,
    }
    out_path = os.path.join(base_results_dir, "vqe_robust_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[compare_vqe_energy] Saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Compare VQE circuit robustness by energy under noise.")
    p.add_argument("--base-results-dir", required=True)
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--molecule", type=str, default="H2")
    p.add_argument(
        "--attack-modes",
        nargs="+",
        default=["asymmetric_noise", "over_rotation", "amplitude_damping", "phase_damping", "readout"],
    )
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--mitigation-mode", choices=[MITIGATION_NONE, MITIGATION_RC_ZNE], default=MITIGATION_RC_ZNE)
    p.add_argument("--rc-zne-scales", type=float, nargs="+", default=RC_ZNE_DEFAULT_SCALES)
    p.add_argument("--rc-zne-fit", type=str, default="linear", choices=["linear", "quadratic"])
    p.add_argument("--rc-zne-reps", type=int, default=1)
    args = p.parse_args()

    compare_vqe(
        base_results_dir=args.base_results_dir,
        num_runs=args.num_runs,
        molecule=args.molecule,
        attack_modes=args.attack_modes,
        samples=args.samples,
        mitigation_mode=args.mitigation_mode,
        rc_zne_scales=args.rc_zne_scales,
        rc_zne_fit=args.rc_zne_fit,
        rc_zne_reps=args.rc_zne_reps,
    )


if __name__ == "__main__":
    main()
