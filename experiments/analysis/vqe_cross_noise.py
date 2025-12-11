#!/usr/bin/env python3
"""
Cross-noise energy robustness for VQE circuits.

Evaluates baseline/robust/quantumnas circuits under multiple noise families/rates
and reports energy statistics. Mirrors state-prep cross-noise but uses energy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import cirq

from qas_gym.utils import is_twirl_op, build_frame_twirled_noisy_circuit, load_circuit
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


def _energy(circuit: cirq.Circuit, qubits, hamiltonian: np.ndarray, frame=None) -> float:
    sim = cirq.Simulator()
    res = sim.simulate(circuit, qubit_order=qubits)
    return state_energy(res.final_state_vector, hamiltonian)


def eval_energy(
    circuit: cirq.Circuit,
    hamiltonian: np.ndarray,
    attack_mode: str,
    *,
    epsilon_overrot: float = 0.1,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0,
    gamma_amp: float = 0.05,
    gamma_phase: float = 0.05,
    p_readout: float = 0.03,
    samples: int = 16,
    mitigation_mode: str = MITIGATION_NONE,
    rc_zne_scales: Sequence[float] = RC_ZNE_DEFAULT_SCALES,
    rc_zne_fit: str = "linear",
    rc_zne_reps: int = 1,
    rng=None,
) -> dict:
    qubits = sorted(list(circuit.all_qubits()))
    clean = _energy(circuit, qubits, hamiltonian)
    rc_scales = tuple(float(s) for s in rc_zne_scales if s is not None and float(s) >= 0.0)
    if mitigation_mode == MITIGATION_RC_ZNE and len(rc_scales) < 2:
        rc_scales = RC_ZNE_DEFAULT_SCALES
    rc_zne_reps = max(1, int(rc_zne_reps) if rc_zne_reps is not None else 1)
    rng = rng or np.random.default_rng()
    vals = []
    for _ in range(samples):
        if attack_mode not in DETERMINISTIC_MODES:
            vals.append(clean)
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
                    reps.append(_energy(noisy, qubits, hamiltonian, frame))
                zne_vals.append(float(np.mean(reps)))
            est = _zero_noise_extrapolate(rc_scales, zne_vals, fit=rc_zne_fit)
            vals.append(float(est))
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
            vals.append(_energy(noisy, qubits, hamiltonian))
    arr = np.asarray(vals, dtype=float)
    return {"clean_energy": clean, "mean": float(arr.mean()), "std": float(arr.std()), "samples": vals}


def run_cross_noise(
    base_results_dir: str,
    num_runs: int,
    molecule: str,
    noise_families: list[str],
    rates: list[float],
    samples: int,
    mitigation_mode: str,
    rc_zne_scales: Sequence[float],
    rc_zne_fit: str,
    rc_zne_reps: int,
    hamiltonian_json: str | None = None,
):
    ham = None
    if hamiltonian_json:
        ham = np.array(json.loads(Path(hamiltonian_json).read_text()), dtype=complex)
    else:
        try:
            ham = get_standard_hamiltonian(molecule)["matrix"]
        except Exception:
            # fallback simple ZZ chain
            from experiments.analysis.compare_vqe_energy import _zz_chain_hamiltonian
            n_qubits = None
            run0 = Path(base_results_dir) / "run_0" / "circuit_vanilla.json"
            if run0.exists():
                n_qubits = len(load_circuit(run0).all_qubits())
            ham = _zz_chain_hamiltonian(n_qubits or 3)
    results = {}
    for family in noise_families:
        family_res = {"vanilla": [], "robust": [], "quantumnas": []}
        for i in range(num_runs):
            run_dir = Path(base_results_dir) / f"run_{i}"
            paths = {
                "vanilla": run_dir / "circuit_vanilla.json",
                "robust": run_dir / "circuit_robust.json",
                "quantumnas": run_dir / "circuit_quantumnas.json",
            }
            circuits = {k: load_circuit(str(p)) for k, p in paths.items() if p.exists()}
            for rate in rates:
                kwargs = {
                    "attack_mode": family,
                    "samples": samples,
                    "mitigation_mode": mitigation_mode,
                    "rc_zne_scales": rc_zne_scales,
                    "rc_zne_fit": rc_zne_fit,
                    "rc_zne_reps": rc_zne_reps,
                    "epsilon_overrot": rate,
                    "p_x": rate,
                    "p_y": rate,
                    "p_z": rate,
                    "gamma_amp": rate,
                    "gamma_phase": rate,
                    "p_readout": rate,
                }
                for label, circ in circuits.items():
                    res = eval_energy(circ, ham, **kwargs)
                    res["rate"] = rate
                    res["seed"] = i
                    family_res[label].append(res)
        results[family] = family_res
    out = {
        "metadata": {
            "molecule": molecule,
            "noise_families": noise_families,
            "rates": rates,
            "samples_per_point": samples,
            "mitigation_mode": mitigation_mode,
            "rc_zne_scales": list(rc_zne_scales),
            "rc_zne_fit": rc_zne_fit,
            "rc_zne_reps": rc_zne_reps,
        },
        "results": results,
    }
    out_path = Path(base_results_dir) / "vqe_cross_noise.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[vqe_cross_noise] Saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Cross-noise energy robustness for VQE circuits.")
    p.add_argument("--base-results-dir", required=True)
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--molecule", type=str, default="H2")
    p.add_argument("--noise-families", nargs="+", default=["over_rotation", "amplitude_damping", "phase_damping", "readout"])
    p.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--mitigation-mode", choices=[MITIGATION_NONE, MITIGATION_RC_ZNE], default=MITIGATION_RC_ZNE)
    p.add_argument("--rc-zne-scales", type=float, nargs="+", default=RC_ZNE_DEFAULT_SCALES)
    p.add_argument("--rc-zne-fit", type=str, default="linear", choices=["linear", "quadratic"])
    p.add_argument("--rc-zne-reps", type=int, default=1)
    p.add_argument("--hamiltonian-json", type=str, default=None, help="Optional path to a Hamiltonian matrix (JSON list of lists).")
    args = p.parse_args()
    run_cross_noise(
        base_results_dir=args.base_results_dir,
        num_runs=args.num_runs,
        molecule=args.molecule,
        noise_families=args.noise_families,
        rates=args.rates,
        samples=args.samples,
        mitigation_mode=args.mitigation_mode,
        rc_zne_scales=args.rc_zne_scales,
        rc_zne_fit=args.rc_zne_fit,
        rc_zne_reps=args.rc_zne_reps,
        hamiltonian_json=args.hamiltonian_json,
    )


if __name__ == "__main__":
    main()
