"""Simple fidelity + robustness comparison for GHZ experiments.

This script scans result folders (default: results/ghz_3_new and results/ghz_4_new),
loads Cirq JSON circuits produced by different methods (vanilla, robust, quantumnas, etc.),
computes the clean fidelity to the target GHZ state, evaluates a lightweight
noise-agnostic robustness profile (by inserting deterministic noise families
after gates and estimating fidelity under those perturbations), and aggregates
results across seeds and methods.

Outputs per-dataset CSV summaries and two plots:
 - boxplot of robustness index per method (shows distribution across seeds)
 - paired scatter plot of robustness index (vanilla vs robust) per seed

The script is intentionally conservative about simulation workload: default
sample counts and severity grid are small so it runs quickly for a smoke check.
Adjust the parameters at the top of `main` for heavier evaluation.
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import math
import numpy as np
import matplotlib.pyplot as plt

try:
    import cirq
except Exception as e:  # pragma: no cover - graceful fallback
    raise RuntimeError("cirq is required to run this script") from e

# Add repository root to path so `src` imports work when run from experiments/analysis
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from qas_gym.utils import load_circuit


def ghz_statevector(n: int) -> np.ndarray:
    """Return GHZ_n statevector |0..0> + |1..1> normalized."""
    if n <= 0:
        return np.array([1.0], dtype=complex)
    dim = 2 ** n
    vec = np.zeros(dim, dtype=complex)
    vec[0] = 1.0
    vec[-1] = 1.0
    vec = vec / math.sqrt(2.0)
    return vec


def fidelity_pure_target_from_statevector(statevec: np.ndarray, target: np.ndarray) -> float:
    # fidelity for pure target: |<target|state>|^2
    ov = np.vdot(target, statevec)
    return float(np.real(np.abs(ov) ** 2))


def simulate_statevector(circuit: cirq.Circuit) -> np.ndarray:
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    # cirq returns final_state_vector as `final_state_vector` attribute
    try:
        vec = res.final_state_vector
    except Exception:
        # older cirq: use state_vector()
        vec = cirq.final_state_vector(res)
    return np.asarray(vec, dtype=complex)


def apply_noise_family_to_circuit(circuit: cirq.Circuit, family: str, params: Dict[str, float]) -> cirq.Circuit:
    """Insert simple deterministic noise operations after each gate.

    This is intentionally simple: it appends small Kraus-like channels or
    single-qubit rotations after each operation to emulate systematic errors.
    The resulting circuit is simulated with a density-matrix simulator when
    needed.
    """
    noisy_ops = []
    for op in circuit.all_operations():
        noisy_ops.append(op)
        # Skip adding noise after measurement- or classical ops if present
        if isinstance(op, cirq.MeasurementGate):
            continue
        for q in op.qubits:
            if family == "over_rotation":
                eps = float(params.get("epsilon_overrot", 0.0))
                noisy_ops.append(cirq.rx(eps).on(q))
            elif family == "amplitude_damping":
                gamma = float(params.get("gamma_amp", 0.0))
                noisy_ops.append(cirq.amplitude_damp(gamma).on(q))
            elif family == "phase_damping":
                gamma = float(params.get("gamma_phase", 0.0))
                noisy_ops.append(cirq.phase_damp(gamma).on(q))
            elif family == "readout":
                p = float(params.get("p_readout", 0.0))
                noisy_ops.append(cirq.bit_flip(p).on(q))
            elif family == "depolarizing":
                p = float(params.get("p_dep", 0.0))
                noisy_ops.append(cirq.depolarize(p).on(q))
            else:
                # unknown family -> no-op
                continue
    return cirq.Circuit(noisy_ops)


def evaluate_noise_agnostic_profile(
    circuit: cirq.Circuit,
    target_state: np.ndarray,
    severities: Sequence[float],
    samples_per_point: int = 8,
    rng: np.random.Generator | None = None,
) -> Dict:
    """Evaluate mean fidelity under deterministic noise draws for several severities.

    Returns dict with clean_fidelity, per-severity mean/std, and robustness_index (AUC of retention).
    """
    rng = rng or np.random.default_rng(0)
    families = ["amplitude_damping", "phase_damping", "over_rotation", "readout", "depolarizing"]

    # baseline clean fidelity (pure state)
    try:
        statevec = simulate_statevector(circuit)
        clean_fid = fidelity_pure_target_from_statevector(statevec, target_state)
    except Exception:
        clean_fid = 0.0

    profile = []
    for sev in severities:
        vals = []
        for s in range(max(1, int(samples_per_point))):
            fam = str(rng.choice(families))
            # simple scaling heuristics for parameters
            params = {
                "epsilon_overrot": 0.05 * float(sev),
                "gamma_amp": 0.02 * float(sev),
                "gamma_phase": 0.02 * float(sev),
                "p_readout": 0.01 * float(sev),
                "p_dep": 0.01 * float(sev),
            }
            noisy_circ = apply_noise_family_to_circuit(circuit, fam, params)
            # use density-matrix simulator to support non-unitary channels
            try:
                dm_sim = cirq.DensityMatrixSimulator()
                res = dm_sim.simulate(noisy_circ)
                rho = res.final_density_matrix
                # fidelity with pure target = <psi|rho|psi>
                vals.append(float(np.real(target_state.conj() @ (rho @ target_state))))
            except Exception:
                # fallback to statevector when noise channels unsupported
                try:
                    vec = simulate_statevector(noisy_circ)
                    vals.append(fidelity_pure_target_from_statevector(vec, target_state))
                except Exception:
                    vals.append(0.0)
        arr = np.asarray(vals, dtype=float)
        mean_f = float(arr.mean()) if arr.size else 0.0
        std_f = float(arr.std()) if arr.size > 1 else 0.0
        retention = float(mean_f / max(clean_fid, 1e-12)) if clean_fid > 0 else 0.0
        profile.append({"severity": float(sev), "f_mean": mean_f, "f_std": std_f, "retention": retention})

    sev_arr = np.array([p["severity"] for p in profile], dtype=float)
    ret_arr = np.array([p["retention"] for p in profile], dtype=float)
    if sev_arr.size >= 2 and sev_arr[-1] - sev_arr[0] > 0:
        auc = float(np.trapz(ret_arr, sev_arr))
        robustness_index = float(auc / (sev_arr[-1] - sev_arr[0]))
    else:
        robustness_index = float(ret_arr[0]) if ret_arr.size else 0.0

    return {"clean_fidelity": float(clean_fid), "profile": profile, "robustness_index": float(np.clip(robustness_index, 0.0, 1.0))}


def find_circuit_files(base_dir: Path) -> Dict[str, List[Path]]:
    """Find available circuit jsons under a dataset base dir and group by method name."""
    found: Dict[str, List[Path]] = {}
    # Look at compare/run_* folders first
    compare_dir = base_dir / "compare"
    if compare_dir.exists():
        for run in sorted(compare_dir.glob("run_*")):
            for fname in run.glob("circuit_*.json"):
                key = fname.stem.replace("circuit_", "")  # e.g. 'vanilla', 'robust', 'quantumnas'
                found.setdefault(key, []).append(fname)

    # Also look at top-level method folders (adversarial, quantumnas, etc.)
    for method in ("adversarial", "quantumnas", "vanilla", "robust", "compare"):
        p = base_dir / method
        if p.exists():
            for fname in p.rglob("circuit_*.json"):
                key = fname.stem.replace("circuit_", "")
                found.setdefault(key, []).append(fname)

    return found


def aggregate_and_plot(results: Dict[str, List[Dict]], out_dir: Path, dataset_name: str):
    """Make two summary plots and save CSV summary.

    `results` is mapping method -> list of per-circuit result dicts (each must contain 'seed' optional and 'robustness_index', 'clean_fidelity').
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"summary_{dataset_name}.csv"
    # write CSV
    import csv

    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed", "clean_fidelity", "robustness_index"])
        for method, items in results.items():
            for it in items:
                writer.writerow([method, it.get("seed", ""), it.get("clean_fidelity", ""), it.get("robustness_index", "")])

    # Boxplot of robustness_index per method
    methods = sorted(results.keys())
    data = [ [it["robustness_index"] for it in results[m]] for m in methods ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, labels=methods, showmeans=True)
    ax.set_ylabel("Robustness index (normalized retention AUC)")
    ax.set_title(f"Robustness index by method — {dataset_name}")
    plt.tight_layout()
    boxpath = out_dir / f"robustness_box_{dataset_name}.png"
    fig.savefig(str(boxpath), dpi=200)
    plt.close(fig)

    # Paired scatter: if both 'vanilla' and 'robust' exist, plot per-seed pairs
    if "vanilla" in results and "robust" in results:
        # build mapping seed -> value
        def idx_map(arr):
            m = {}
            for it in arr:
                seed = it.get("seed", None)
                if seed is None:
                    # fallback to enumerate
                    continue
                m[int(seed)] = it["robustness_index"]
            return m

        vmap = idx_map(results["vanilla"])
        rmap = idx_map(results["robust"])
        common = sorted(set(vmap) & set(rmap))
        if common:
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            xs = [vmap[s] for s in common]
            ys = [rmap[s] for s in common]
            ax2.scatter(xs, ys)
            for i, s in enumerate(common):
                ax2.annotate(str(s), (xs[i], ys[i]))
            lims = [0, 1]
            ax2.plot(lims, lims, linestyle="--", color="gray")
            ax2.set_xlim(0.0, 1.0)
            ax2.set_ylim(0.0, 1.0)
            ax2.set_xlabel("Vanilla robustness index")
            ax2.set_ylabel("Robust robustness index")
            ax2.set_title(f"Paired vanilla vs robust — {dataset_name}")
            plt.tight_layout()
            pairpath = out_dir / f"paired_vanilla_vs_robust_{dataset_name}.png"
            fig2.savefig(str(pairpath), dpi=200)
            plt.close(fig2)

    return csv_path


def main():
    # User-tweakable parameters for a smoke-run
    base_dirs = [Path("results/ghz_3_new"), Path("results/ghz_4_new")]
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    samples_per_point = 8
    rng = np.random.default_rng(12345)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for base in base_dirs:
        if not base.exists():
            print(f"[compare_circuits] Skipping missing dataset folder {base}")
            continue
        found = find_circuit_files(base)
        dataset_name = base.name
        out_dir = base / f"compare_analysis_{timestamp}"
        per_method_results: Dict[str, List[Dict]] = {}
        for method, paths in found.items():
            for p in paths:
                try:
                    circ = load_circuit(str(p))
                except Exception as e:
                    print(f"[compare_circuits] Failed to load {p}: {e}")
                    continue
                qubits = sorted(list(circ.all_qubits()))
                n = len(qubits)
                target = ghz_statevector(n)
                res_clean = 0.0
                try:
                    vec = simulate_statevector(circ)
                    # note: simulate_statevector orders basis by qubit ordering; assume LineQubit ordering is canonical
                    res_clean = fidelity_pure_target_from_statevector(vec, target)
                except Exception:
                    res_clean = 0.0

                profile = evaluate_noise_agnostic_profile(circ, target, severities, samples_per_point, rng)
                seed = None
                # try to infer seed from path name like 'seed_3' or 'run_2'
                for part in p.parts:
                    if part.startswith("seed_"):
                        try:
                            seed = int(part.split("_")[1])
                        except Exception:
                            seed = None
                    if part.startswith("run_"):
                        try:
                            seed = int(part.split("_")[1])
                        except Exception:
                            seed = seed

                entry = {"path": str(p), "clean_fidelity": res_clean, "robustness_index": profile["robustness_index"]}
                if seed is not None:
                    entry["seed"] = int(seed)
                per_method_results.setdefault(method, []).append(entry)

        csv_path = aggregate_and_plot(per_method_results, out_dir, dataset_name)
        print(f"[compare_circuits] Wrote summary CSV to {csv_path}")
        print(f"[compare_circuits] Plots and details in {out_dir}")


if __name__ == "__main__":
    main()
"""
Circuit comparison and robustness analysis for quantum architecture search.

This module evaluates and compares the robustness of vanilla (baseline) and robust 
(adversarially-trained) circuits under multi-gate saboteur attacks.

Target type and task mode are configured centrally via experiments/config.py:
- TARGET_TYPE: 'toffoli' (default) or 'ghz'
- TASK_MODE: 'state_preparation' (default) or 'unitary_preparation'

Statistical Protocol:
    - Multiple attack samples per circuit for robust statistics
    - Results include mean ± std for attacked fidelities
    - Plots show error bars and sample size annotations
    - Summary files include all hyperparameters and aggregation methods
"""

import os
import sys

# Add repository root to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import cirq
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Sequence

from experiments import config
from qas_gym.utils import (
    apply_noise,
    fidelity_pure_target,
    randomized_compile,
    is_twirl_op,
    build_frame_twirled_noisy_circuit,
    apply_inverse_pauli_frame_to_target,
)


MITIGATION_NONE = "none"
MITIGATION_TWIRL = "twirl"
MITIGATION_RC_ZNE = "rc_zne"

MITIGATION_VARIANTS: dict[str, tuple[str, ...]] = {
    MITIGATION_NONE: ("untwirled",),
    MITIGATION_TWIRL: ("untwirled", "twirled"),
    MITIGATION_RC_ZNE: ("untwirled", "mitigated"),
}

RC_ZNE_DEFAULT_SCALES: tuple[float, ...] = (1.0, 1.5, 2.0)

NOISE_AGNOSTIC_FAMILIES: tuple[str, ...] = (
    "over_rotation",
    "asymmetric_noise",
    "amplitude_damping",
    "phase_damping",
    "readout",
)

# Maximal deterministic noise parameters derived from GHZ-3 cross-noise sweeps.
NOISE_AGNOSTIC_BASE_PARAMS: dict[str, float] = {
    "epsilon_overrot": 0.2,  # RX over-rotation bound (rad)
    "p_x": 0.05,
    "p_y": 0.02,
    "p_z": 0.02,
    "gamma_amp": 0.05,
    "gamma_phase": 0.05,
    "p_readout": 0.05,
}

# Emphasize decoherence-heavy channels that differentiate shorter circuits.
NOISE_AGNOSTIC_DEFAULT_WEIGHTS: dict[str, float] = {
    "amplitude_damping": 0.3,
    "phase_damping": 0.2,
    "over_rotation": 0.2,
    "readout": 0.2,
    "asymmetric_noise": 0.1,
}


def _richardson_extrapolate(scales: Sequence[float], values: Sequence[float]) -> float:
    """Return zero-noise estimate via first-order Richardson extrapolation."""
    if len(scales) != len(values) or not scales:
        raise ValueError("Scale and value lists must match and be non-empty")
    if len(scales) == 1:
        return float(values[0])
    x = np.asarray(scales, dtype=float)
    y = np.asarray(values, dtype=float)
    # Linear fit and evaluate at scale 0
    coeffs = np.polyfit(x, y, deg=1)
    return float(np.polyval(coeffs, 0.0))


def _polyfit_extrapolate(scales: Sequence[float], values: Sequence[float], order: int) -> float:
    """Generic polynomial fit extrapolation evaluated at zero noise."""
    if len(scales) != len(values) or not scales:
        raise ValueError("Scale and value lists must match and be non-empty")
    if len(scales) == 1:
        return float(values[0])
    order = max(1, min(order, len(scales) - 1))
    x = np.asarray(scales, dtype=float)
    y = np.asarray(values, dtype=float)
    coeffs = np.polyfit(x, y, deg=order)
    return float(np.polyval(coeffs, 0.0))


def _zero_noise_extrapolate(scales: Sequence[float], values: Sequence[float], fit: str = "linear") -> float:
    """
    Flexible zero-noise extrapolation with model selection.

    Args:
        scales: Noise scale factors.
        values: Observed fidelities per scale.
        fit: Extrapolation model ('linear'/'richardson' or 'quadratic').
    """
    mode = (fit or "linear").lower()
    if mode in ("linear", "richardson", "order1"):
        return _richardson_extrapolate(scales, values)
    if mode in ("quadratic", "poly2", "order2", "richardson2"):
        if len(scales) < 3:
            return _richardson_extrapolate(scales, values)
        return _polyfit_extrapolate(scales, values, order=2)
    raise ValueError(f"Unknown rc_zne_fit '{fit}'")


def _scale_noise_kwargs(
    attack_mode: str,
    scale: float,
    *,
    epsilon_overrot: float,
    p_x: float,
    p_y: float,
    p_z: float,
    gamma_amp: float,
    gamma_phase: float,
    p_readout: float,
) -> dict[str, float]:
    """Scale deterministic-noise parameters for zero-noise extrapolation."""

    def _clamp_prob(value: float) -> float:
        return float(np.clip(value, 0.0, 0.999999))

    def _scale_damping(prob: float, factor: float) -> float:
        prob = _clamp_prob(prob)
        if prob <= 0.0:
            return 0.0
        # Preserve physicality via survival probability scaling
        survival = 1.0 - prob
        scaled = 1.0 - survival**factor
        return _clamp_prob(scaled)

    scale = float(max(scale, 0.0))
    scaled = {
        "epsilon_overrot": float(np.clip(epsilon_overrot * scale, -np.pi, np.pi)) if attack_mode == "over_rotation" else epsilon_overrot,
        "p_x": _clamp_prob(p_x * scale) if attack_mode == "asymmetric_noise" else p_x,
        "p_y": _clamp_prob(p_y * scale) if attack_mode == "asymmetric_noise" else p_y,
        "p_z": _clamp_prob(p_z * scale) if attack_mode == "asymmetric_noise" else p_z,
        "gamma_amp": _scale_damping(gamma_amp, scale) if attack_mode == "amplitude_damping" else gamma_amp,
        "gamma_phase": _scale_damping(gamma_phase, scale) if attack_mode == "phase_damping" else gamma_phase,
        "p_readout": _clamp_prob(p_readout * scale) if attack_mode == "readout" else p_readout,
    }
    return scaled


def _apply_noise_family_to_circuit(
    circuit: cirq.Circuit,
    family: str,
    params: dict[str, float],
) -> cirq.Circuit:
    """Return a circuit with a deterministic noise family inserted after each gate."""
    noisy_ops = []
    for op in circuit.all_operations():
        noisy_ops.append(op)
        if is_twirl_op(op):
            continue
        for q in op.qubits:
            if family == "over_rotation":
                noisy_ops.append(cirq.rx(params.get("epsilon_overrot", 0.0)).on(q))
            elif family == "asymmetric_noise":
                noisy_ops.append(
                    cirq.asymmetric_depolarize(
                        p_x=params.get("p_x", 0.0),
                        p_y=params.get("p_y", 0.0),
                        p_z=params.get("p_z", 0.0),
                    ).on(q)
                )
            elif family == "amplitude_damping":
                noisy_ops.append(cirq.amplitude_damp(params.get("gamma_amp", 0.0)).on(q))
            elif family == "phase_damping":
                noisy_ops.append(cirq.phase_damp(params.get("gamma_phase", 0.0)).on(q))
            elif family == "readout":
                noisy_ops.append(cirq.bit_flip(params.get("p_readout", 0.0)).on(q))
            else:
                raise ValueError(f"Unsupported noise family '{family}'")
    return cirq.Circuit(noisy_ops)


def _build_noise_agnostic_schedule(
    severities: Sequence[float],
    samples_per_point: int,
    rng: np.random.Generator | None = None,
    family_weights: dict[str, float] | None = None,
) -> list[list[dict[str, str | dict[str, float]]]]:
    """
    Create a reusable per-severity schedule of deterministic noise draws.

    Each severity λ ∈ [0, 1] samples the same number of deterministic families,
    ensuring fair comparisons between vanilla and robust circuits on a seed-by-seed basis.
    """
    rng = rng or np.random.default_rng()
    schedule: list[list[dict[str, str | dict[str, float]]]] = []
    samples_per_point = max(1, int(samples_per_point))
    families = list(NOISE_AGNOSTIC_FAMILIES)
    probs = None
    if family_weights:
        weights = np.array([max(0.0, family_weights.get(fam, 0.0)) for fam in families], dtype=float)
        total = float(weights.sum())
        if total > 0:
            probs = weights / total
        else:
            family_weights = None
            probs = None

    for severity in severities:
        local_events: list[dict[str, str | dict[str, float]]] = []
        for _ in range(samples_per_point):
            if probs is not None:
                family = str(rng.choice(families, p=probs))
            else:
                family = str(rng.choice(families))
            params = _scale_noise_kwargs(
                family,
                float(severity),
                **NOISE_AGNOSTIC_BASE_PARAMS,
            )
            local_events.append({"family": family, "params": params})
        schedule.append(local_events)
    return schedule


def evaluate_noise_agnostic_profile(
    circuit: cirq.Circuit,
    target_state: np.ndarray,
    severities: Sequence[float],
    noise_schedule: Sequence[Sequence[dict[str, str | dict[str, float]]]],
) -> dict:
    """
    Evaluate a deterministic, noise-family-agnostic robustness profile.

    Args:
        circuit: Circuit under test.
        target_state: Desired target quantum state.
        severities: Sequence of λ values in [0, 1], matching GHZ-3 sweeps where λ=1
            corresponds to ε≈0.2 or p≈0.05 errors.
        noise_schedule: Pre-sampled deterministic noise draws shared across circuits.

    Returns:
        Dict storing per-severity mean/std fidelities, retention ratios, and a
        normalized robustness index (area under the retention curve).
    """
    if len(severities) != len(noise_schedule):
        raise ValueError("Severity grid must match noise schedule length")
    qubits = sorted(list(circuit.all_qubits()))
    clean_fid = fidelity_pure_target(circuit, target_state, qubits) if qubits else 0.0
    clean_ref = max(clean_fid, 1e-12)
    profile = []
    for severity, events in zip(severities, noise_schedule):
        sample_vals = []
        for event in events:
            family = str(event["family"])
            params = dict(event["params"])  # shallow copy
            noisy_circuit = _apply_noise_family_to_circuit(circuit, family, params)
            sample_vals.append(fidelity_pure_target(noisy_circuit, target_state, qubits))
        arr = np.array(sample_vals, dtype=float)
        fidelity_mean = float(arr.mean()) if arr.size else 0.0
        fidelity_std = float(arr.std()) if arr.size > 1 else 0.0
        retention_samples = np.clip(arr / clean_ref, 0.0, 1.0) if clean_ref > 0 else np.zeros_like(arr)
        profile.append({
            "severity": float(severity),
            "fidelity_mean": fidelity_mean,
            "fidelity_std": fidelity_std,
            "retention_mean": float(retention_samples.mean()) if retention_samples.size else 0.0,
            "retention_std": float(retention_samples.std()) if retention_samples.size > 1 else 0.0,
            "n_samples": int(arr.size),
        })
    severity_arr = np.asarray([p["severity"] for p in profile], dtype=float)
    retention_arr = np.asarray([p["retention_mean"] for p in profile], dtype=float)
    if severity_arr.size < 2:
        robustness_index = float(retention_arr[0]) if retention_arr.size else 0.0
    else:
        span = float(severity_arr[-1] - severity_arr[0])
        auc = float(np.trapz(retention_arr, severity_arr))
        robustness_index = float(auc / span) if span > 0 else float(auc)
    return {
        "clean_fidelity": float(clean_fid),
        "profile": profile,
        "robustness_index": float(np.clip(robustness_index, 0.0, 1.0)),
    }


def _plot_noise_agnostic_profiles(
    noise_profiles: dict[str, list[dict]],
    out_dir: str,
    log_fn,
    max_severity: float,
):
    """Generate per-seed and overlay plots for noise-agnostic robustness profiles."""
    vanilla_profiles = noise_profiles.get("vanilla", [])
    robust_profiles = noise_profiles.get("robust", [])
    if not vanilla_profiles or not robust_profiles:
        return
    seeds = sorted({int(p.get("seed", -1)) for p in vanilla_profiles})
    if not seeds:
        return
    severity = [pt["severity"] for pt in vanilla_profiles[0]["profile"]]
    cols = min(3, len(seeds))
    rows = int(np.ceil(len(seeds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(rows, cols)
    for idx, seed in enumerate(seeds):
        r_idx, c_idx = divmod(idx, cols)
        ax = axes[r_idx, c_idx]
        prof_v = next((p for p in vanilla_profiles if int(p.get("seed", -1)) == seed), None)
        prof_r = next((p for p in robust_profiles if int(p.get("seed", -1)) == seed), None)
        if prof_v:
            vals_v = [pt["retention_mean"] for pt in prof_v["profile"]]
            ax.plot(severity, vals_v, linestyle="--", color="#54A24B",
                    label=f"Vanilla (RI={prof_v['robustness_index']:.3f})")
        if prof_r:
            vals_r = [pt["retention_mean"] for pt in prof_r["profile"]]
            ax.plot(severity, vals_r, linestyle="-", color="#F58518",
                    label=f"Robust (RI={prof_r['robustness_index']:.3f})")
        ax.set_title(f"Seed {seed}")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        if r_idx == rows - 1:
            ax.set_xlabel(r"Noise severity $\lambda$")
        if c_idx == 0:
            ax.set_ylabel("Retention")
        if prof_v or prof_r:
            ax.legend(loc="lower left", fontsize=8)
    for extra_idx in range(len(seeds), rows * cols):
        ax = axes[extra_idx // cols, extra_idx % cols]
        ax.axis("off")
    fig.suptitle(f"Noise-agnostic robustness per seed (λ max={max_severity:.2f})")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    per_seed_path = os.path.join(out_dir, "noise_agnostic_per_seed.png")
    plt.savefig(per_seed_path, dpi=200)
    plt.close(fig)
    log_fn(f"[compare_circuits] Saved noise-agnostic per-seed plot to {per_seed_path}")

    # Overlay plot with all seeds to visualize consistency
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for prof_v in vanilla_profiles:
        vals = [pt["retention_mean"] for pt in prof_v["profile"]]
        ax2.plot(severity, vals, color="#54A24B", alpha=0.3)
    for prof_r in robust_profiles:
        vals = [pt["retention_mean"] for pt in prof_r["profile"]]
        ax2.plot(severity, vals, color="#F58518", alpha=0.3)
    mean_v = np.mean([[pt["retention_mean"] for pt in prof["profile"]] for prof in vanilla_profiles], axis=0)
    mean_r = np.mean([[pt["retention_mean"] for pt in prof["profile"]] for prof in robust_profiles], axis=0)
    ax2.plot(severity, mean_v, color="#238B45", linewidth=2.5, label="Vanilla (mean)")
    ax2.plot(severity, mean_r, color="#E6550D", linewidth=2.5, label="Robust (mean)")
    ax2.set_xlabel(r"Noise severity $\lambda$")
    ax2.set_ylabel("Retention")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Noise-agnostic retention overlay (all seeds, λ max={max_severity:.2f})")
    ax2.legend()
    overlay_path = os.path.join(out_dir, "noise_agnostic_overlay.png")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200)
    plt.close(fig2)
    log_fn(f"[compare_circuits] Saved noise-agnostic overlay plot to {overlay_path}")
# Import statistical utilities
from utils.stats import (
    aggregate_metrics,
    create_experiment_summary,
    save_experiment_summary,
    get_git_commit_hash,
    format_metric_with_error,
)



# --- Multi-gate saboteur attack evaluation ---
def evaluate_multi_gate_attacks(
    circuit,
    saboteur_agent,
    target_state,
    n_qubits,
    samples=32,
    fallback_error_idx=0,
    saboteur_budget: int = 3,
    rng: np.random.Generator | None = None,
    attack_mode: str = "max",  # 'max' (worst-case), 'policy' (agent), 'random_high' (high-level random), 'over_rotation', 'asymmetric_noise'
    epsilon_overrot: float = 0.1,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0,
    gamma_amp: float = 0.05,
    gamma_phase: float = 0.05,
    p_readout: float = 0.03,
    mitigation_mode: str = MITIGATION_NONE,
    rc_zne_scales: Sequence[float] = RC_ZNE_DEFAULT_SCALES,
    rc_zne_fit: str = "linear",
    rc_zne_reps: int = 1,
):
    """
    Evaluate circuit robustness under multi-gate attacks sampled from saboteur_agent.
    Returns dict with clean fidelity, mean/min/std attacked fidelity, and all samples.
    If saboteur_agent is None, uses zero-vector (no noise) as fallback.
    Budgeted top-k attack mirrors training (default budget=3).

    rc_zne_fit: Extrapolation model for RC-ZNE ('linear' or 'quadratic').
    rc_zne_reps: Number of RC draws to average per noise scale before extrapolation.
    """
    from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
    qubits = sorted(list(circuit.all_qubits()))
    deterministic_modes = {"over_rotation", "asymmetric_noise", "amplitude_damping", "phase_damping", "readout"}
    # 'random_noise' selects one of the deterministic_modes each sample and randomizes its parameters
    random_noise_defaults = {
        "epsilon_overrot_max": 0.2,
        "p_max": 0.12,
        "gamma_max": 0.12,
        "p_readout_max": 0.06,
    }
    rc_scales = tuple(float(s) for s in rc_zne_scales if s is not None and float(s) >= 0.0)
    if mitigation_mode == MITIGATION_RC_ZNE and len(rc_scales) < 2:
        rc_scales = RC_ZNE_DEFAULT_SCALES
    rc_zne_reps = max(1, int(rc_zne_reps) if rc_zne_reps is not None else 1)
    # --- Dimension check ---
    circuit_n_qubits = len(qubits)
    target_dim = target_state.shape[0]
    expected_dim = 2 ** circuit_n_qubits
    if expected_dim != target_dim:
        raise ValueError(
            f"[ERROR] Circuit qubit count ({circuit_n_qubits}) does not match target_state dimension ({target_dim}). "
            f"Expected dimension: {expected_dim}.\n"
            f"Check that the circuit and target_state are for the same number of qubits.\n"
            f"Circuit: {circuit}\nTarget state shape: {target_state.shape}"
        )
    clean_fid = fidelity_pure_target(circuit, target_state, qubits)
    ops = list(circuit.all_operations())
    attacked_vals = []
    rc_zne_scale_values: list[list[float]] = []
    rng = rng or np.random.default_rng()
    all_rates = SaboteurMultiGateEnv.all_error_rates
    max_idx = len(all_rates) - 1
    valid_gate_count = min(len(ops), config.MAX_CIRCUIT_TIMESTEPS)

    for _ in range(samples):
        work_circuit = circuit
        if attack_mode in deterministic_modes or attack_mode == "random_noise":
            if mitigation_mode == MITIGATION_RC_ZNE:
                sample_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
                zne_values: list[float] = []
                for scale in rc_scales:
                    scale_reps: list[float] = []
                    for rep in range(rc_zne_reps):
                        rep_seed = sample_seed + rep
                        local_rng = np.random.default_rng(rep_seed)
                        # If attack_mode is random_noise, sample a deterministic family and parameters per rep
                        if attack_mode == "random_noise":
                            chosen = local_rng.choice(list(deterministic_modes))
                            # sample parameters within reasonable bounds
                            samp_eps = float(local_rng.uniform(-random_noise_defaults["epsilon_overrot_max"], random_noise_defaults["epsilon_overrot_max"]))
                            samp_px = float(local_rng.uniform(0.0, random_noise_defaults["p_max"]))
                            samp_py = float(local_rng.uniform(0.0, random_noise_defaults["p_max"]))
                            samp_pz = float(local_rng.uniform(0.0, random_noise_defaults["p_max"]))
                            samp_gamma_amp = float(local_rng.uniform(0.0, random_noise_defaults["gamma_max"]))
                            samp_gamma_phase = float(local_rng.uniform(0.0, random_noise_defaults["gamma_max"]))
                            samp_p_readout = float(local_rng.uniform(0.0, random_noise_defaults["p_readout_max"]))
                            scaled = _scale_noise_kwargs(
                                chosen,
                                scale,
                                epsilon_overrot=samp_eps,
                                p_x=samp_px,
                                p_y=samp_py,
                                p_z=samp_pz,
                                gamma_amp=samp_gamma_amp,
                                gamma_phase=samp_gamma_phase,
                                p_readout=samp_p_readout,
                            )
                            # remember chosen family for potential debugging (not stored here)
                        else:
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
                        noisy_circuit, frame = build_frame_twirled_noisy_circuit(
                            work_circuit,
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
                        scale_reps.append(
                            fidelity_pure_target(noisy_circuit, target_state, qubits, frame=frame)
                        )
                    zne_values.append(float(np.mean(scale_reps)))
                rc_zne_scale_values.append(zne_values)
                estimate = _zero_noise_extrapolate(rc_scales, zne_values, fit=rc_zne_fit)
                attacked_vals.append(float(np.clip(estimate, 0.0, 1.0)))
                continue

            twirl_for_mode = mitigation_mode == MITIGATION_TWIRL and (attack_mode in deterministic_modes or attack_mode == "random_noise")
            if twirl_for_mode:
                twirl_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
                local_rng = np.random.default_rng(twirl_seed)
                if attack_mode == "random_noise":
                    chosen = local_rng.choice(list(deterministic_modes))
                    samp_eps = float(local_rng.uniform(-random_noise_defaults["epsilon_overrot_max"], random_noise_defaults["epsilon_overrot_max"]))
                    samp_px = float(local_rng.uniform(0.0, random_noise_defaults["p_max"]))
                    samp_py = float(local_rng.uniform(0.0, random_noise_defaults["p_max"]))
                    samp_pz = float(local_rng.uniform(0.0, random_noise_defaults["p_max"]))
                    samp_gamma_amp = float(local_rng.uniform(0.0, random_noise_defaults["gamma_max"]))
                    samp_gamma_phase = float(local_rng.uniform(0.0, random_noise_defaults["gamma_max"]))
                    samp_p_readout = float(local_rng.uniform(0.0, random_noise_defaults["p_readout_max"]))
                    noisy_circuit, frame = build_frame_twirled_noisy_circuit(
                        work_circuit,
                        local_rng,
                        chosen,
                        epsilon_overrot=samp_eps,
                        p_x=samp_px,
                        p_y=samp_py,
                        p_z=samp_pz,
                        gamma_amp=samp_gamma_amp,
                        gamma_phase=samp_gamma_phase,
                        p_readout=samp_p_readout,
                    )
                else:
                    noisy_circuit, frame = build_frame_twirled_noisy_circuit(
                        work_circuit,
                        local_rng,
                        attack_mode,
                        epsilon_overrot=epsilon_overrot,
                        p_x=p_x,
                        p_y=p_y,
                        p_z=p_z,
                        gamma_amp=gamma_amp,
                        gamma_phase=gamma_phase,
                        p_readout=p_readout,
                    )
                attacked_vals.append(fidelity_pure_target(noisy_circuit, target_state, qubits, frame=frame))
            else:
                # Non-RC deterministic/noise insertion path
                noisy_ops = []
                for op in work_circuit.all_operations():
                    noisy_ops.append(op)
                    if is_twirl_op(op):
                        continue
                    for q in op.qubits:
                        # If random_noise, sample a family and params per sample
                        if attack_mode == "random_noise":
                            chosen = rng.choice(list(deterministic_modes))
                            samp_eps = float(rng.uniform(-random_noise_defaults["epsilon_overrot_max"], random_noise_defaults["epsilon_overrot_max"]))
                            samp_px = float(rng.uniform(0.0, random_noise_defaults["p_max"]))
                            samp_py = float(rng.uniform(0.0, random_noise_defaults["p_max"]))
                            samp_pz = float(rng.uniform(0.0, random_noise_defaults["p_max"]))
                            samp_gamma_amp = float(rng.uniform(0.0, random_noise_defaults["gamma_max"]))
                            samp_gamma_phase = float(rng.uniform(0.0, random_noise_defaults["gamma_max"]))
                            samp_p_readout = float(rng.uniform(0.0, random_noise_defaults["p_readout_max"]))
                            fam = chosen
                        else:
                            fam = attack_mode
                        if fam == "over_rotation":
                            noisy_ops.append(cirq.rx(samp_eps if attack_mode == "random_noise" else epsilon_overrot).on(q))
                        elif fam == "asymmetric_noise":
                            noisy_ops.append(cirq.asymmetric_depolarize(p_x=(samp_px if attack_mode == "random_noise" else p_x), p_y=(samp_py if attack_mode == "random_noise" else p_y), p_z=(samp_pz if attack_mode == "random_noise" else p_z)).on(q))
                        elif fam == "amplitude_damping":
                            noisy_ops.append(cirq.amplitude_damp((samp_gamma_amp if attack_mode == "random_noise" else gamma_amp)).on(q))
                        elif fam == "phase_damping":
                            noisy_ops.append(cirq.phase_damp((samp_gamma_phase if attack_mode == "random_noise" else gamma_phase)).on(q))
                        elif fam == "readout":
                            noisy_ops.append(cirq.bit_flip((samp_p_readout if attack_mode == "random_noise" else p_readout)).on(q))
                noisy_circuit = cirq.Circuit(noisy_ops)
                attacked_vals.append(fidelity_pure_target(noisy_circuit, target_state, qubits))
            continue

        sab_action = None
        ops_full = list(work_circuit.all_operations())
        payload_ops = [op for op in ops_full if not is_twirl_op(op)]
        valid_gate_count = min(len(payload_ops), config.MAX_CIRCUIT_TIMESTEPS)
        budget = min(saboteur_budget, valid_gate_count)

        if attack_mode == "max":
            # Worst-case: assign max error to all gates but honor budgeted subset (tie-break randomly)
            sab_action = np.full(valid_gate_count, max_idx, dtype=int)
            budget = min(saboteur_budget, valid_gate_count)
        elif attack_mode == "policy":
            sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(
                work_circuit, n_qubits=n_qubits, max_circuit_timesteps=config.MAX_CIRCUIT_TIMESTEPS
            )
            if saboteur_agent is not None:
                try:
                    sab_action, _ = saboteur_agent.predict(sab_obs, deterministic=False)
                except Exception:
                    sab_action = None
        else:
            # random_high: random from the top error levels
            high_min = max(0, max_idx - 2)
            sab_action = rng.integers(high_min, max_idx + 1, size=valid_gate_count, dtype=int)
            budget = valid_gate_count


        # Fallback if policy failed
        if sab_action is None:
            high_min = max(0, max_idx - 2)
            sab_action = rng.integers(high_min, max_idx + 1, size=valid_gate_count, dtype=int)
            budget = valid_gate_count

        # Budgeted top-k attack (consistent with training)
        raw_action = np.array(sab_action[:valid_gate_count], dtype=int)
        effective_action = np.zeros_like(raw_action)
        if budget > 0 and len(raw_action) > 0:
            if np.all(raw_action == raw_action[0]):
                # If all scores are equal, choose budgeted gates uniformly at random
                budget_indices = rng.choice(len(raw_action), size=budget, replace=False)
                effective_action[budget_indices] = raw_action[budget_indices]
            else:
                top_k_indices = np.argsort(raw_action)[-budget:]
                effective_action[top_k_indices] = raw_action[top_k_indices]

        noisy_ops = []
        payload_pos = 0
        for op in ops_full:
            noisy_ops.append(op)
            if is_twirl_op(op):
                continue
            idx = int(effective_action[payload_pos]) if payload_pos < len(effective_action) else fallback_error_idx
            idx = max(0, min(idx, max_idx))
            error_rate = all_rates[idx]
            for q in op.qubits:
                noisy_ops.append(cirq.DepolarizingChannel(error_rate).on(q))
            payload_pos += 1
        noisy_circuit = cirq.Circuit(noisy_ops)
        attacked_vals.append(fidelity_pure_target(noisy_circuit, target_state, qubits))
    attacked_arr = np.array(attacked_vals)
    result = {
        "clean_fidelity": float(clean_fid),
        "mean_attacked": float(attacked_arr.mean()),
        "min_attacked": float(attacked_arr.min()),
        "std_attacked": float(attacked_arr.std()),
        "samples": attacked_vals
    }
    if mitigation_mode == MITIGATION_RC_ZNE:
        result["rc_zne_scales_used"] = list(rc_scales)
        result["rc_zne_fit"] = rc_zne_fit
        result["rc_zne_reps"] = rc_zne_reps
        if rc_zne_scale_values:
            scale_arr = np.asarray(rc_zne_scale_values, dtype=float)
            result["rc_zne_scale_mean"] = scale_arr.mean(axis=0).tolist()
            result["rc_zne_scale_std"] = scale_arr.std(axis=0).tolist()
    return result

def calculate_fidelity(circuit: cirq.Circuit, target_state: np.ndarray) -> float:
    """Unified fidelity via fidelity_pure_target helper."""
    qubits = sorted(list(circuit.all_qubits())) if circuit.all_qubits() else []
    return fidelity_pure_target(circuit, target_state, qubits) if qubits else 0.0


def _count_gates_and_cnots(circuit: cirq.Circuit) -> tuple[int, int]:
    """Return total gate count and CNOT count for a circuit."""
    ops = list(circuit.all_operations())
    total = len(ops)
    cnots = sum(1 for op in ops if isinstance(op.gate, cirq.CNotPowGate))
    return total, cnots


# --- Move compare_noise_resilience to top-level for import ---
def compare_noise_resilience(
    base_results_dir,
    num_runs,
    n_qubits,
    samples=32,
    saboteur_budget: int = 3,
    seed: int | None = 42,
    logger=None,
    attack_mode: str = "random_high",
    attack_modes: list[str] | None = None,
    epsilon_overrot: float = 0.1,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0,
    gamma_amp: float = 0.05,
    gamma_phase: float = 0.05,
    p_readout: float = 0.03,
    quantumnas_circuit_path: str | None = None,
    ignore_saboteur: bool = False,
    mitigation_mode: str = MITIGATION_NONE,
    rc_zne_scales: Sequence[float] = RC_ZNE_DEFAULT_SCALES,
    rc_zne_fit: str = "linear",
    rc_zne_reps: int = 1,
    attack_sampling_mode: str = "per_gate_uniform",
    noise_agnostic_enabled: bool = True,
    noise_agnostic_points: int = 16,
    noise_agnostic_samples: int = 8,
    noise_agnostic_max_severity: float = 1.3,
    noise_agnostic_severity_bias: float = 0.7,
    noise_agnostic_uniform_families: bool = False,
):
    """
    Aggregate and compare circuit robustness under multi-gate attacks.
    
    Statistical Protocol:
        - Multiple attack samples per circuit
        - Results include mean ± std for attacked fidelities  
        - Error bars and sample size annotations on plots
    - Automatic A/B variant generation when mitigation_mode!='none'
    
    Args:
        base_results_dir: Base directory containing run subdirectories.
        num_runs: Number of experimental runs to aggregate.
        n_qubits: Number of qubits for this analysis.
        samples: Number of saboteur attack samples per circuit.
        attack_modes: Optional list of attack modes to sweep (e.g., ['random_high', 'asymmetric_noise', 'over_rotation', 'amplitude_damping']).
        quantumnas_circuit_path: Optional explicit path to a Cirq JSON circuit to include.
        logger: Optional logger for output.
        attack_mode: Kept for backward compatibility; if attack_modes is provided, this is ignored.
        gamma_amp: Amplitude damping probability for 'amplitude_damping' attack mode.
        gamma_phase: Phase damping (dephasing) probability for 'phase_damping' attack mode.
        p_readout: Readout error probability for 'readout' attack mode (applied as bit-flip channel).
        mitigation_mode: Mitigation strategy for deterministic noise ('none', 'twirl', 'rc_zne').
            'twirl' reproduces Pauli-frame twirling; 'rc_zne' performs randomized compiling plus zero-noise extrapolation.
        rc_zne_scales: Noise scaling factors used when mitigation_mode='rc_zne'.
        rc_zne_fit: Extrapolation model for RC-ZNE ('linear' or 'quadratic').
        rc_zne_reps: Number of randomized-compiling draws averaged per scale.
        noise_agnostic_enabled: If True, compute deterministic noise-family-agnostic retention curves per seed.
        noise_agnostic_points: Number of severity points λ ∈ [0, 1] for the noise-agnostic sweep.
        noise_agnostic_samples: Number of deterministic noise draws per severity value.
        noise_agnostic_max_severity: Max λ applied to noise scaling (default 1.3 to extend beyond GHZ-3 sweeps).
        noise_agnostic_severity_bias: Bias exponent (<1 focuses sampling near λ max) for severity grid spacing.
        noise_agnostic_uniform_families: If True, sample deterministic noise families uniformly.
    """
    attack_modes = attack_modes or [attack_mode]
    primary_mode = attack_modes[0]

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if mitigation_mode not in MITIGATION_VARIANTS:
        log(f"[compare_circuits] Unknown mitigation_mode '{mitigation_mode}', defaulting to 'none'.")
        mitigation_mode = MITIGATION_NONE
    variants_for_mode = MITIGATION_VARIANTS[mitigation_mode]
    secondary_variant = next((v for v in variants_for_mode if v != "untwirled"), None)
    if rc_zne_scales is None:
        rc_zne_scales = RC_ZNE_DEFAULT_SCALES
    else:
        rc_zne_scales = tuple(rc_zne_scales)
    rc_zne_reps = max(1, int(rc_zne_reps) if rc_zne_reps is not None else 1)
    
    log("--- Aggregating and Comparing Circuit Robustness (Multi-Gate Attacks) ---")
    summary_json = os.path.join(base_results_dir, "robust_eval.json")
    samples_csv = os.path.join(base_results_dir, "attacked_fidelity_samples.csv")
    # Use central config to get target state for circuit robustness evaluation
    target_state = config.get_target_state(n_qubits)

    per_mode_metrics = {
        mode: {
            "vanilla": [],
            "robust": [],
            "quantumnas": [],
            "samples": [],
        }
        for mode in attack_modes
    }
    noise_agnostic_store = {
        "vanilla": [],
        "robust": [],
        "quantumnas": [],
    }
    noise_summary: dict[str, dict[str, float]] = {}
    noise_settings = None

    # Attack policy: prefer a trained saboteur; otherwise fall back to the strongest error level.
    try:
        from stable_baselines3 import PPO
        from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
        saboteur_model_path = os.path.join(base_results_dir, "../saboteur/saboteur_trained_on_architect_model.zip")
        saboteur_agent = None
        if not ignore_saboteur and os.path.exists(saboteur_model_path):
            saboteur_agent = PPO.load(saboteur_model_path)
        fallback_error_idx = len(SaboteurMultiGateEnv.all_error_rates) - 1  # worst-case level
        if saboteur_agent is None:
            log(f"[compare_circuits] No saboteur model found or ignored (ignore_saboteur={ignore_saboteur}); using budgeted max-level fallback.")
    except Exception:
        saboteur_agent = None
        fallback_error_idx = 0
        log("[compare_circuits] Failed to load saboteur model; falling back to no-noise attacks.")

    # Shared QuantumNAS circuit fallback: either explicit flag or auto-discovery
    shared_qnas_path = None
    if quantumnas_circuit_path:
        candidate = os.path.expanduser(quantumnas_circuit_path)
        if os.path.exists(candidate):
            shared_qnas_path = os.path.abspath(candidate)
        else:
            log(f"  Warning: Provided QuantumNAS circuit not found at {candidate}")
    else:
        default_candidate = os.path.abspath(os.path.join(base_results_dir, "..", "quantumnas", "circuit_quantumnas.json"))
        if os.path.exists(default_candidate):
            shared_qnas_path = default_candidate

    severity_grid = None
    max_severity_applied = None
    family_weights = None
    if noise_agnostic_enabled:
        noise_agnostic_points = max(2, int(noise_agnostic_points))
        noise_agnostic_samples = max(1, int(noise_agnostic_samples))
        max_sev = float(max(0.1, noise_agnostic_max_severity))
        max_severity_applied = max_sev
        bias = float(noise_agnostic_severity_bias) if noise_agnostic_severity_bias > 0 else 1.0
        grid = np.linspace(0.0, 1.0, noise_agnostic_points)
        grid = np.power(grid, bias)
        severity_grid = grid * max_sev
        family_weights = None if noise_agnostic_uniform_families else NOISE_AGNOSTIC_DEFAULT_WEIGHTS
        noise_settings = {
            "points": noise_agnostic_points,
            "samples_per_point": noise_agnostic_samples,
            "max_severity": max_sev,
            "severity_bias": bias,
            "family_weights": family_weights,
        }
        log(
            f"[compare_circuits] Noise-agnostic sweep: λ max={max_sev:.2f}, bias={bias:.2f}, "
            f"{noise_agnostic_samples} samples/point, family_weights={'uniform' if family_weights is None else 'decoherence-heavy'}"
        )

    for i in range(num_runs):
        run_dir = os.path.join(base_results_dir, f"run_{i}")
        log(f"\nProcessing Run {i+1}/{num_runs} from {run_dir}")
        vanilla_circuit_file = os.path.join(run_dir, "circuit_vanilla.json")
        robust_circuit_file = os.path.join(run_dir, "circuit_robust.json")
        quantumnas_circuit_file = os.path.join(run_dir, "circuit_quantumnas.json")

        try:
            from qas_gym.utils import load_circuit
            circuit_vanilla = load_circuit(vanilla_circuit_file)
            circuit_robust = load_circuit(robust_circuit_file)
            circuit_qnas = None
            qnas_path_used = None
            if os.path.exists(quantumnas_circuit_file):
                qnas_path_used = quantumnas_circuit_file
            elif shared_qnas_path and os.path.exists(shared_qnas_path):
                qnas_path_used = shared_qnas_path

            if qnas_path_used:
                try:
                    circuit_qnas = load_circuit(qnas_path_used)
                except Exception as exc:
                    log(f"  Warning: Failed to load QuantumNAS circuit from {qnas_path_used}: {exc}")
            noise_schedule = None
            if noise_agnostic_enabled and severity_grid is not None:
                schedule_seed = None if seed is None else seed + 10000 + i
                schedule_rng = np.random.default_rng(schedule_seed)
                noise_schedule = _build_noise_agnostic_schedule(
                    severity_grid, noise_agnostic_samples, schedule_rng, family_weights=family_weights
                )
        except FileNotFoundError as e:
            log(f"  Warning: Could not find circuit files in {run_dir}. Skipping run. Error: {e}")
            continue

        for mode in attack_modes:
            rng = np.random.default_rng(None if seed is None else seed + i)
            for variant in variants_for_mode:
                if variant == "untwirled":
                    eval_mitigation = MITIGATION_NONE
                elif variant == "twirled":
                    eval_mitigation = MITIGATION_TWIRL
                elif variant == "mitigated":
                    eval_mitigation = MITIGATION_RC_ZNE
                else:
                    eval_mitigation = MITIGATION_NONE
                # Allow sampling-mode-aware per-circuit budget adjustments
                # "per_gate_uniform" keeps the provided saboteur_budget unchanged.
                # "equalized_fraction" scales per-circuit budget so that the fraction of gates
                # attacked is approximately constant across circuits in this run.
                gates_v, cnots_v = _count_gates_and_cnots(circuit_vanilla)
                gates_r, cnots_r = _count_gates_and_cnots(circuit_robust)
                gates_q, cnots_q = (None, None)
                if circuit_qnas is not None:
                    gates_q, cnots_q = _count_gates_and_cnots(circuit_qnas)

                if attack_sampling_mode == "equalized_fraction":
                    counts = [c for c in (gates_v, gates_r, gates_q) if c is not None and c > 0]
                    if counts:
                        avg_count = float(np.mean(counts))
                    else:
                        avg_count = float(max(1, saboteur_budget))
                    target_fraction = float(saboteur_budget) / max(1.0, avg_count)
                    sab_budget_v = max(1, int(round(target_fraction * max(1, gates_v))))
                    sab_budget_r = max(1, int(round(target_fraction * max(1, gates_r))))
                    sab_budget_q = max(1, int(round(target_fraction * max(1, gates_q)))) if gates_q is not None else saboteur_budget
                else:
                    sab_budget_v = sab_budget_r = sab_budget_q = saboteur_budget

                metrics_v = evaluate_multi_gate_attacks(
                    circuit_vanilla, saboteur_agent, target_state, n_qubits,
                    samples=samples, fallback_error_idx=fallback_error_idx,
                    saboteur_budget=sab_budget_v, rng=rng, attack_mode=mode,
                    epsilon_overrot=epsilon_overrot, p_x=p_x, p_y=p_y, p_z=p_z,
                    gamma_amp=gamma_amp, gamma_phase=gamma_phase, p_readout=p_readout,
                    mitigation_mode=eval_mitigation,
                    rc_zne_scales=rc_zne_scales,
                    rc_zne_fit=rc_zne_fit,
                    rc_zne_reps=rc_zne_reps,
                )
                metrics_v["variant"] = variant
                metrics_v["seed"] = i
                metrics_v["circuit_path"] = vanilla_circuit_file
                gates_v, cnots_v = _count_gates_and_cnots(circuit_vanilla)
                metrics_v["gate_count"] = gates_v
                metrics_v["cnot_count"] = cnots_v
                per_mode_metrics[mode]["vanilla"].append(metrics_v)
                for val in metrics_v["samples"]:
                    per_mode_metrics[mode]["samples"].append([i, "vanilla", f"{mode}_{variant}", val, metrics_v.get("gate_count", None)])
                if noise_agnostic_enabled and variant == "untwirled" and noise_schedule is not None and severity_grid is not None:
                    profile_v = evaluate_noise_agnostic_profile(
                        circuit_vanilla,
                        target_state,
                        severity_grid,
                        noise_schedule,
                    )
                    profile_v.update({
                        "seed": i,
                        "circuit_type": "vanilla",
                        "circuit_path": vanilla_circuit_file,
                        "gate_count": gates_v,
                        "cnot_count": cnots_v,
                    })
                    noise_agnostic_store["vanilla"].append(profile_v)

                metrics_r = evaluate_multi_gate_attacks(
                    circuit_robust, saboteur_agent, target_state, n_qubits,
                    samples=samples, fallback_error_idx=fallback_error_idx,
                    saboteur_budget=saboteur_budget, rng=rng, attack_mode=mode,
                    epsilon_overrot=epsilon_overrot, p_x=p_x, p_y=p_y, p_z=p_z,
                    gamma_amp=gamma_amp, gamma_phase=gamma_phase, p_readout=p_readout,
                    mitigation_mode=eval_mitigation,
                    rc_zne_scales=rc_zne_scales,
                    rc_zne_fit=rc_zne_fit,
                    rc_zne_reps=rc_zne_reps,
                )
                metrics_r["variant"] = variant
                metrics_r["seed"] = i
                metrics_r["circuit_path"] = robust_circuit_file
                metrics_r["gate_count"] = gates_r
                metrics_r["cnot_count"] = cnots_r
                per_mode_metrics[mode]["robust"].append(metrics_r)
                for val in metrics_r["samples"]:
                    per_mode_metrics[mode]["samples"].append([i, "robust", f"{mode}_{variant}", val, metrics_r.get("gate_count", None)])
                if noise_agnostic_enabled and variant == "untwirled" and noise_schedule is not None and severity_grid is not None:
                    profile_r = evaluate_noise_agnostic_profile(
                        circuit_robust,
                        target_state,
                        severity_grid,
                        noise_schedule,
                    )
                    profile_r.update({
                        "seed": i,
                        "circuit_type": "robust",
                        "circuit_path": robust_circuit_file,
                        "gate_count": gates_r,
                        "cnot_count": cnots_r,
                    })
                    noise_agnostic_store["robust"].append(profile_r)

                if circuit_qnas is not None:
                    metrics_q = evaluate_multi_gate_attacks(
                        circuit_qnas, saboteur_agent, target_state, n_qubits,
                        samples=samples, fallback_error_idx=fallback_error_idx,
                        saboteur_budget=saboteur_budget, rng=rng, attack_mode=mode,
                        epsilon_overrot=epsilon_overrot, p_x=p_x, p_y=p_y, p_z=p_z,
                        gamma_amp=gamma_amp, gamma_phase=gamma_phase, p_readout=p_readout,
                        mitigation_mode=eval_mitigation,
                        rc_zne_scales=rc_zne_scales,
                        rc_zne_fit=rc_zne_fit,
                        rc_zne_reps=rc_zne_reps,
                    )
                    metrics_q["variant"] = variant
                    metrics_q["seed"] = i
                    metrics_q["circuit_path"] = qnas_path_used or quantumnas_circuit_file
                    metrics_q["gate_count"] = gates_q
                    metrics_q["cnot_count"] = cnots_q
                    per_mode_metrics[mode]["quantumnas"].append(metrics_q)
                    for val in metrics_q["samples"]:
                        per_mode_metrics[mode]["samples"].append([i, "quantumnas", f"{mode}_{variant}", val, metrics_q.get("gate_count", None)])
                    if noise_agnostic_enabled and variant == "untwirled" and noise_schedule is not None and severity_grid is not None:
                        profile_q = evaluate_noise_agnostic_profile(
                            circuit_qnas,
                            target_state,
                            severity_grid,
                            noise_schedule,
                        )
                        profile_q.update({
                            "seed": i,
                            "circuit_type": "quantumnas",
                            "circuit_path": qnas_path_used or quantumnas_circuit_file,
                            "gate_count": gates_q,
                            "cnot_count": cnots_q,
                        })
                        noise_agnostic_store["quantumnas"].append(profile_q)

    aggregated_by_mode = {}
    for mode, buckets in per_mode_metrics.items():
        vm = buckets["vanilla"]
        rm = buckets["robust"]
        qm = buckets["quantumnas"]
        if not vm or not rm:
            continue
        # Separate variants
        def _split_variant(metrics_list, variant):
            return [m["mean_attacked"] for m in metrics_list if m.get("variant") == variant]

        vanilla_means = _split_variant(vm, "untwirled")
        vanilla_sec = _split_variant(vm, secondary_variant) if secondary_variant else []
        robust_means = _split_variant(rm, "untwirled")
        robust_sec = _split_variant(rm, secondary_variant) if secondary_variant else []
        qnas_means = _split_variant(qm, "untwirled") if qm else []
        qnas_sec = _split_variant(qm, secondary_variant) if qm and secondary_variant else []

        vanilla_overall = aggregate_metrics(vanilla_means)
        robust_overall = aggregate_metrics(robust_means)
        qnas_overall = aggregate_metrics(qnas_means) if qnas_means else None
        vanilla_sec_overall = aggregate_metrics(vanilla_sec) if vanilla_sec else None
        robust_sec_overall = aggregate_metrics(robust_sec) if robust_sec else None
        qnas_sec_overall = aggregate_metrics(qnas_sec) if qnas_sec else None

        aggregated_by_mode[mode] = {
            "vanilla": vanilla_overall,
            "robust": robust_overall,
            "quantumnas": qnas_overall,
        }
        if secondary_variant:
            aggregated_by_mode[mode][f"vanilla_{secondary_variant}"] = vanilla_sec_overall
            aggregated_by_mode[mode][f"robust_{secondary_variant}"] = robust_sec_overall
            aggregated_by_mode[mode][f"quantumnas_{secondary_variant}"] = qnas_sec_overall
            if secondary_variant == "twirled":
                aggregated_by_mode[mode]["vanilla_twirl"] = vanilla_sec_overall
                aggregated_by_mode[mode]["robust_twirl"] = robust_sec_overall
                aggregated_by_mode[mode]["quantumnas_twirl"] = qnas_sec_overall

        log(f"\nOverall Statistics [{mode}] (n={len(vanilla_means)} runs, {samples} samples each):")
        log(
            f"  Vanilla: {format_metric_with_error(vanilla_overall['mean'], vanilla_overall['std'], int(vanilla_overall['n']))}"
        )
        if vanilla_sec_overall:
            log(
                f"    └ Mitigated ({secondary_variant}): {format_metric_with_error(vanilla_sec_overall['mean'], vanilla_sec_overall['std'], int(vanilla_sec_overall['n']))}"
            )
        log(
            f"  Robust:  {format_metric_with_error(robust_overall['mean'], robust_overall['std'], int(robust_overall['n']))}"
        )
        if robust_sec_overall:
            log(
                f"    └ Mitigated ({secondary_variant}): {format_metric_with_error(robust_sec_overall['mean'], robust_sec_overall['std'], int(robust_sec_overall['n']))}"
            )
        if qnas_overall:
            log(
                f"  HEA baseline: {format_metric_with_error(qnas_overall['mean'], qnas_overall['std'], int(qnas_overall['n']))}"
            )
            if qnas_sec_overall:
                log(
                    f"    └ Mitigated ({secondary_variant}): {format_metric_with_error(qnas_sec_overall['mean'], qnas_sec_overall['std'], int(qnas_sec_overall['n']))}"
                )

    if noise_agnostic_enabled:
        log("\nNoise-agnostic robustness indices (λ ∈ [0, 1]):")
        for label in ("vanilla", "robust", "quantumnas"):
            entries = noise_agnostic_store.get(label, [])
            if not entries:
                continue
            values = [float(entry.get("robustness_index", 0.0)) for entry in entries]
            agg = aggregate_metrics(values)
            log(f"  {label.capitalize()}: {format_metric_with_error(agg['mean'], agg['std'], int(agg['n']))}")

    # Backward-compatible primary mode aliases
    primary_metrics = per_mode_metrics.get(primary_mode, {})
    all_metrics_vanilla = primary_metrics.get("vanilla", [])
    all_metrics_robust = primary_metrics.get("robust", [])
    all_metrics_qnas = primary_metrics.get("quantumnas", [])
    vanilla_overall = aggregated_by_mode.get(primary_mode, {}).get("vanilla")
    robust_overall = aggregated_by_mode.get(primary_mode, {}).get("robust")
    qnas_overall = aggregated_by_mode.get(primary_mode, {}).get("quantumnas")

    # Save results to JSON with statistical info
    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_qubits": n_qubits,
            "num_runs": num_runs,
            "samples_per_circuit": samples,
            "primary_attack_mode": primary_mode,
            "attack_modes": attack_modes,
            "mitigation_mode": mitigation_mode,
            "mitigation_secondary_variant": secondary_variant,
            "rc_zne_scales": list(rc_zne_scales) if mitigation_mode == MITIGATION_RC_ZNE else None,
            "rc_zne_fit": rc_zne_fit if mitigation_mode == MITIGATION_RC_ZNE else None,
            "rc_zne_reps": rc_zne_reps if mitigation_mode == MITIGATION_RC_ZNE else None,
            "statistical_protocol": {
                "aggregation_method": "mean ± std",
                "samples_per_circuit": samples,
            },
            "noise_agnostic_settings": noise_settings if noise_agnostic_enabled else None,
        },
        "per_mode": {
            mode: {
                "vanilla": buckets["vanilla"],
                "robust": buckets["robust"],
                "quantumnas": buckets["quantumnas"],
                "aggregated": aggregated_by_mode.get(mode, {}),
            }
            for mode, buckets in per_mode_metrics.items()
        },
        # Backward-compatible aliases for the primary mode
        "vanilla": all_metrics_vanilla,
        "robust": all_metrics_robust,
        "quantumnas": all_metrics_qnas,
    }
    if noise_agnostic_enabled:
        results_data["noise_agnostic_profiles"] = noise_agnostic_store

    if noise_agnostic_enabled:
        noise_summary = {}
        for label, entries in noise_agnostic_store.items():
            if not entries:
                continue
            noise_summary[label] = aggregate_metrics([float(entry.get("robustness_index", 0.0)) for entry in entries])
        results_data["noise_agnostic_summary"] = noise_summary

    if vanilla_overall and robust_overall:
        results_data["aggregated"] = aggregated_by_mode.get(primary_mode, {})
    
    with open(summary_json, "w") as f:
        json.dump(results_data, f, indent=2)
    log(f"\nRobustness summary saved to {summary_json}")

    # Write all sample values to CSV (with attack_mode column)
    with open(samples_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_idx", "circuit_type", "attack_mode", "attacked_fidelity", "gate_count"])
        for mode, buckets in per_mode_metrics.items():
            writer.writerows(buckets["samples"])
    log(f"Attacked fidelity samples saved to {samples_csv}")

    # --- Plot comparison of vanilla vs robust with error bars ---
    try:
        if all_metrics_vanilla and all_metrics_robust:
            def _series(metrics_list, overall_base, overall_variant, variant_label):
                untwirled = {m["seed"]: m for m in metrics_list if m.get("variant") == "untwirled" and "seed" in m}
                mitigated = (
                    {m["seed"]: m for m in metrics_list if m.get("variant") == variant_label and "seed" in m}
                    if variant_label else {}
                )
                seeds_sorted = sorted(untwirled.keys())
                labels_local = [f"seed_{s}" for s in seeds_sorted]
                base_means = [untwirled[s]["mean_attacked"] for s in seeds_sorted]
                base_stds = [untwirled[s]["std_attacked"] for s in seeds_sorted]
                mitigated_raw = [mitigated.get(s, {}).get("mean_attacked", 0.0) for s in seeds_sorted]
                mitigated_stds = [mitigated.get(s, {}).get("std_attacked", 0.0) for s in seeds_sorted]
                variant_gain = [max(0.0, m - b) for m, b in zip(mitigated_raw, base_means)] if variant_label else [0.0] * len(base_means)
                labels_local.append("Mean")
                base_means.append(overall_base["mean"] if overall_base else 0.0)
                base_stds.append(overall_base["std"] if overall_base else 0.0)
                if variant_label:
                    agg_variant = overall_variant["mean"] if overall_variant else 0.0
                    variant_gain.append(max(0.0, agg_variant - base_means[-1]))
                    mitigated_stds.append(overall_variant["std"] if overall_variant else 0.0)
                else:
                    variant_gain.append(0.0)
                    mitigated_stds.append(0.0)
                mitigated_stds = [ts if g > 0 else 0.0 for ts, g in zip(mitigated_stds, variant_gain)]
                has_variant = variant_label is not None and (bool(mitigated) or overall_variant is not None)
                return labels_local, base_means, base_stds, variant_gain, mitigated_stds, has_variant

            agg_primary = aggregated_by_mode.get(primary_mode, {})

            variant_key = secondary_variant
            labels_v, means_v, stds_v, variant_v_gain, variant_v_std, variant_v_present = _series(
                all_metrics_vanilla,
                vanilla_overall,
                agg_primary.get(f"vanilla_{variant_key}") if variant_key else None,
                variant_key,
            )
            labels_r, means_r, stds_r, variant_r_gain, variant_r_std, variant_r_present = _series(
                all_metrics_robust,
                robust_overall,
                agg_primary.get(f"robust_{variant_key}") if variant_key else None,
                variant_key,
            )
            labels_q, means_q, stds_q, variant_q_gain, variant_q_std, variant_q_present = ([], [], [], [], [], False)
            if all_metrics_qnas:
                labels_q, means_q, stds_q, variant_q_gain, variant_q_std, variant_q_present = _series(
                    all_metrics_qnas,
                    qnas_overall,
                    agg_primary.get(f"quantumnas_{variant_key}") if variant_key else None,
                    variant_key,
                )

            labels = labels_v  # assume same seeds across methods
            x = np.arange(len(labels))
            width = 0.22  # gap between method groups

            fig, ax = plt.subplots(figsize=(12, 6))
            used_labels = set()
            err_kw = {"capsize": 4, "capthick": 1.2, "elinewidth": 1.0}

            if variant_key == "twirled":
                variant_display_name = "Twirl gain"
            elif variant_key == "mitigated":
                variant_display_name = (
                    "RC-ZNE gain" if mitigation_mode == MITIGATION_RC_ZNE else "Mitigated gain"
                )
            elif variant_key:
                variant_display_name = f"{variant_key.capitalize()} gain"
            else:
                variant_display_name = "Mitigation gain"

            base_colors = {
                "vanilla": "#54A24B",
                "robust": "#F58518",
                "quantumnas": "#4C78A8",
            }
            variant_colors_by_key = {
                "twirled": {
                    "vanilla": "#a1d99b",
                    "robust": "#fdae6b",
                    "quantumnas": "#9ecae1",
                },
                "mitigated": {
                    "vanilla": "#74c476",
                    "robust": "#fdd0a2",
                    "quantumnas": "#6baed6",
                },
            }
            variant_colors = variant_colors_by_key[variant_key] if variant_key in variant_colors_by_key else {}

            def _plot_method(offset, base_means, base_stds, variant_gain, variant_stds, base_color, variant_color, name, has_variant):
                lb = name if name not in used_labels else None
                ax.bar(
                    x + offset,
                    base_means,
                    width=width,
                    yerr=base_stds,
                    error_kw=err_kw,
                    color=base_color,
                    alpha=0.9,
                    label=lb,
                )
                if lb:
                    used_labels.add(lb)
                if has_variant and any(g > 0 for g in variant_gain):
                    lt = f"{name} ({variant_display_name})"
                    lt = lt if lt not in used_labels else None
                    ax.bar(
                        x + offset,
                        variant_gain,
                        width=width,
                        bottom=base_means,
                        yerr=variant_stds,
                        error_kw=err_kw,
                        color=variant_color,
                        alpha=0.75,
                        label=lt,
                    )
                    if lt:
                        used_labels.add(lt)

            offsets = [-width, 0, width]
            _plot_method(
                offsets[0],
                means_v,
                stds_v,
                variant_v_gain,
                variant_v_std,
                base_colors["vanilla"],
                variant_colors.get("vanilla", base_colors["vanilla"]),
                "RL baseline",
                variant_v_present,
            )
            _plot_method(
                offsets[1],
                means_r,
                stds_r,
                variant_r_gain,
                variant_r_std,
                base_colors["robust"],
                variant_colors.get("robust", base_colors["robust"]),
                "Robust",
                variant_r_present,
            )
            if means_q:
                _plot_method(
                    offsets[2],
                    means_q,
                    stds_q,
                    variant_q_gain,
                    variant_q_std,
                    base_colors["quantumnas"],
                    variant_colors.get("quantumnas", base_colors["quantumnas"]),
                    "HEA baseline",
                    variant_q_present,
                )

            ax.set_ylabel("Mean Attacked Fidelity")
            ax.set_title(
                "Robustness Comparison (Primary Attack Mode)\n"
                f"Mode: {primary_mode} | n={samples} samples/circuit (±1 std)"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            def _max_height(base_vals, gain_vals, has_twirl):
                if not base_vals:
                    return 0.0
                if has_twirl:
                    return max(b + g for b, g in zip(base_vals, gain_vals))
                return max(base_vals)

            max_height = max(
                _max_height(means_v, variant_v_gain, variant_v_present),
                _max_height(means_r, variant_r_gain, variant_r_present),
                _max_height(means_q, variant_q_gain, variant_q_present) if means_q else 0.0,
            )
            ax.set_ylim(0, max(1.05, max_height + 0.1))

            # Annotate gate counts (mean across circuits if available)
            def _mean_count(metrics, key):
                return np.mean([m.get(key, 0) for m in metrics]) if metrics else None
            gate_v = _mean_count(all_metrics_vanilla, "gate_count")
            gate_r = _mean_count(all_metrics_robust, "gate_count")
            gate_q = _mean_count(all_metrics_qnas, "gate_count") if all_metrics_qnas else None
            info = []
            if gate_v is not None:
                info.append(f"RL baseline gates≈{gate_v:.1f}")
            if gate_r is not None:
                info.append(f"Robust gates≈{gate_r:.1f}")
            if gate_q is not None:
                info.append(f"HEA gates≈{gate_q:.1f}")
            if info:
                ax.text(0.02, 0.02, "\n".join(info), transform=ax.transAxes,
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85))

            plt.tight_layout()
            out_path = os.path.join(base_results_dir, "robustness_comparison.png")
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
            log(f"[compare_circuits] Saved comparison plot to {out_path}")

            # --- Additional size-aware analysis and plotting ---
            try:
                # Compute loss per gate for each seed (use mean_attacked per circuit)
                def _loss_per_gate(metrics_list):
                    vals = []
                    for m in metrics_list:
                        gate_count = m.get("gate_count", None)
                        mean_att = m.get("mean_attacked", None)
                        if gate_count is None or gate_count <= 0 or mean_att is None:
                            continue
                        loss_per_gate = (1.0 - float(mean_att)) / float(gate_count)
                        vals.append(loss_per_gate)
                    return vals

                loss_v = _loss_per_gate(all_metrics_vanilla)
                loss_r = _loss_per_gate(all_metrics_robust)
                loss_q = _loss_per_gate(all_metrics_qnas) if all_metrics_qnas else []

                # Boxplot: loss per gate
                labels_plot = ["RL baseline", "Robust"] + (["HEA baseline"] if loss_q else [])
                data_plot = [loss_v, loss_r] + ([loss_q] if loss_q else [])
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.boxplot(data_plot, labels=labels_plot, showmeans=True)
                ax2.set_ylabel("Loss per gate = (1 - attacked_fidelity) / gate_count")
                ax2.set_title("Size-normalized damage per gate (per-circuit means)")
                out2 = os.path.join(base_results_dir, "loss_per_gate_boxplot.png")
                plt.tight_layout()
                plt.savefig(out2, dpi=200)
                plt.close(fig2)
                log(f"[compare_circuits] Saved loss-per-gate boxplot to {out2}")

                # Scatter: attacked fidelity vs gate count (per-seed)
                def _gate_vs_fidelity(metrics_list):
                    g = []
                    fvals = []
                    for m in metrics_list:
                        gc = m.get("gate_count", None)
                        ma = m.get("mean_attacked", None)
                        if gc is None or ma is None:
                            continue
                        g.append(gc)
                        fvals.append(ma)
                    return np.array(g), np.array(fvals)

                gv_v, fv_v = _gate_vs_fidelity(all_metrics_vanilla)
                gv_r, fv_r = _gate_vs_fidelity(all_metrics_robust)
                gv_q, fv_q = _gate_vs_fidelity(all_metrics_qnas) if all_metrics_qnas else (np.array([]), np.array([]))

                fig3, ax3 = plt.subplots(figsize=(7, 5))
                if gv_v.size:
                    ax3.scatter(gv_v, fv_v, label="RL baseline", alpha=0.7, color=base_colors.get("vanilla"))
                    if gv_v.size > 1:
                        coeff = np.polyfit(gv_v, fv_v, 1)
                        xs = np.linspace(gv_v.min(), gv_v.max(), 100)
                        ax3.plot(xs, np.polyval(coeff, xs), color=base_colors.get("vanilla"), linestyle='--')
                if gv_r.size:
                    ax3.scatter(gv_r, fv_r, label="Robust", alpha=0.7, color=base_colors.get("robust"))
                    if gv_r.size > 1:
                        coeff = np.polyfit(gv_r, fv_r, 1)
                        xs = np.linspace(min(gv_r.min(), gv_v.min() if gv_v.size else gv_r.min()), max(gv_r.max(), gv_v.max() if gv_v.size else gv_r.max()), 100)
                        ax3.plot(xs, np.polyval(coeff, xs), color=base_colors.get("robust"), linestyle='--')
                if gv_q.size:
                    ax3.scatter(gv_q, fv_q, label="HEA baseline", alpha=0.7, color=base_colors.get("quantumnas"))
                    if gv_q.size > 1:
                        coeff = np.polyfit(gv_q, fv_q, 1)
                        xs = np.linspace(gv_q.min(), gv_q.max(), 100)
                        ax3.plot(xs, np.polyval(coeff, xs), color=base_colors.get("quantumnas"), linestyle='--')

                ax3.set_xlabel("Gate count (per circuit)")
                ax3.set_ylabel("Mean attacked fidelity")
                ax3.set_title("Attacked fidelity vs gate count (per-seed)")
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                out3 = os.path.join(base_results_dir, "attacked_vs_gatecount_scatter.png")
                plt.tight_layout()
                plt.savefig(out3, dpi=200)
                plt.close(fig3)
                log(f"[compare_circuits] Saved attacked vs gatecount scatter to {out3}")
                # --- Per-seed means scatter with overall mean ± std annotation ---
                try:
                    def _seed_means(metrics_list):
                        return [(m.get("seed", None), m.get("mean_attacked", None)) for m in metrics_list if m.get("variant") == "untwirled"]

                    seed_vals_v = sorted(_seed_means(all_metrics_vanilla), key=lambda x: x[0])
                    seed_vals_r = sorted(_seed_means(all_metrics_robust), key=lambda x: x[0])
                    seed_vals_q = sorted(_seed_means(all_metrics_qnas), key=lambda x: x[0]) if all_metrics_qnas else []

                    seeds = [s for s, _ in seed_vals_v]
                    vals_v = [v for _, v in seed_vals_v]
                    vals_r = [v for _, v in seed_vals_r]
                    vals_q = [v for _, v in seed_vals_q] if seed_vals_q else []

                    fig4, ax4 = plt.subplots(figsize=(10, 5))
                    xs = np.arange(len(seeds))
                    if vals_v:
                        ax4.scatter(xs - 0.2, vals_v, label='RL baseline', alpha=0.8, color=base_colors['vanilla'])
                        mv, sv = float(np.mean(vals_v)), float(np.std(vals_v))
                        ax4.errorbar(len(seeds)+0.5, mv, yerr=sv, fmt='o', color=base_colors['vanilla'], label=f'RL mean±std={mv:.3f}±{sv:.3f}')
                    if vals_r:
                        ax4.scatter(xs + 0.0, vals_r, label='Robust', alpha=0.8, color=base_colors['robust'])
                        mr, sr = float(np.mean(vals_r)), float(np.std(vals_r))
                        ax4.errorbar(len(seeds)+0.5, mr, yerr=sr, fmt='o', color=base_colors['robust'], label=f'Robust mean±std={mr:.3f}±{sr:.3f}')
                    if vals_q:
                        ax4.scatter(xs + 0.2, vals_q, label='HEA baseline', alpha=0.8, color=base_colors['quantumnas'])
                        mq, sq = float(np.mean(vals_q)), float(np.std(vals_q))
                        ax4.errorbar(len(seeds)+0.5, mq, yerr=sq, fmt='o', color=base_colors['quantumnas'], label=f'HEA mean±std={mq:.3f}±{sq:.3f}')

                    ax4.set_xticks(xs)
                    ax4.set_xticklabels([f'seed_{int(s)}' for s in seeds], rotation=45)
                    ax4.set_ylabel('Mean attacked fidelity (per-seed)')
                    ax4.set_title('Per-seed attacked fidelity (untwirled) with overall mean±std')
                    ax4.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
                    plt.tight_layout()
                    out4 = os.path.join(base_results_dir, 'per_seed_means.png')
                    plt.savefig(out4, dpi=200)
                    plt.close(fig4)
                    log(f"[compare_circuits] Saved per-seed means plot to {out4}")
                except Exception as e:
                    log(f"[compare_circuits] Failed to produce per-seed means plot: {e}")
            except Exception as e:
                log(f"[compare_circuits] Failed to produce size-aware plots: {e}")
    except Exception as e:
        log(f"[compare_circuits] Failed to plot comparison: {e}")

    if noise_agnostic_enabled and any(noise_agnostic_store.values()):
        try:
            max_for_plot = max_severity_applied if max_severity_applied is not None else 1.0
            _plot_noise_agnostic_profiles(noise_agnostic_store, base_results_dir, log, max_for_plot)
        except Exception as exc:
            log(f"[compare_circuits] Failed to plot noise-agnostic profiles: {exc}")

    # Create experiment summary file
    if vanilla_overall and robust_overall:
        hyperparameters = {
            "n_qubits": n_qubits,
            "num_runs": num_runs,
            "samples_per_circuit": samples,
        }
        if mitigation_mode == MITIGATION_RC_ZNE:
            hyperparameters.update({
                "rc_zne_scales": list(rc_zne_scales),
                "rc_zne_fit": rc_zne_fit,
                "rc_zne_reps": rc_zne_reps,
            })
        
        aggregated_results = {
            "vanilla_fidelity": vanilla_overall,
            "robust_fidelity": robust_overall,
        }
        if qnas_overall:
            aggregated_results["quantumnas_fidelity"] = qnas_overall
        # Gate/CNOT aggregates
        def _agg_counts(metrics):
            if not metrics:
                return None
            gates = [m.get("gate_count", 0) for m in metrics]
            cnots = [m.get("cnot_count", 0) for m in metrics]
            return {
                "gate_count_mean": float(np.mean(gates)),
                "gate_count_std": float(np.std(gates)),
                "cnot_count_mean": float(np.mean(cnots)),
                "cnot_count_std": float(np.std(cnots)),
            }
        aggregated_results["vanilla_counts"] = _agg_counts(all_metrics_vanilla)
        aggregated_results["robust_counts"] = _agg_counts(all_metrics_robust)
        if all_metrics_qnas:
            aggregated_results["quantumnas_counts"] = _agg_counts(all_metrics_qnas)
        if noise_agnostic_enabled and noise_summary:
            aggregated_results["noise_agnostic_summary"] = noise_summary
        
        summary = create_experiment_summary(
            experiment_name="circuit_robustness_comparison",
            n_seeds=num_runs,
            seeds_used=list(range(num_runs)),
            hyperparameters=hyperparameters,
            aggregated_results=aggregated_results,
            commit_hash=get_git_commit_hash(),
            additional_notes=(
                f"Robustness comparison using {samples} attack samples per circuit. "
                f"Primary attack mode: {primary_mode}. Other evaluated modes: {', '.join(attack_modes)}"
            )
        )
        save_experiment_summary(summary, base_results_dir, 'experiment_summary.json')

    log("--- Robustness Comparison Finished ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare circuit robustness under multi-gate saboteur attacks.")
    parser.add_argument('--base-results-dir', type=str, required=True, help='Base directory containing run subdirectories')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of experimental runs to aggregate (default: 3)')
    parser.add_argument('--n-qubits', type=int, required=True, help='Number of qubits for this analysis')
    parser.add_argument('--samples', type=int, default=32, help='Number of saboteur attack samples per circuit')
    parser.add_argument('--attack-mode', type=str, default='random_high',
                        choices=['max', 'policy', 'random_high', 'random_noise', 'over_rotation', 'asymmetric_noise', 'amplitude_damping', 'phase_damping', 'readout'],
                        help=("Primary attack mode: saboteur-based ('max','policy','random_high') or deterministic noise "
                              "('over_rotation','asymmetric_noise','amplitude_damping','phase_damping','readout')."))
    parser.add_argument('--attack-modes', nargs='+', default=None,
                        help=("Optional list of attack modes to sweep; first entry is used for plotting/back-compat. "
                              "Examples: random_high max asymmetric_noise over_rotation amplitude_damping phase_damping readout"))
    parser.add_argument('--epsilon-overrot', type=float, default=0.1, help='Over-rotation angle (radians) if attack-mode=over_rotation')
    parser.add_argument('--p-x', type=float, default=0.05, help='Asymmetric noise p_x if attack-mode=asymmetric_noise')
    parser.add_argument('--p-y', type=float, default=0.0, help='Asymmetric noise p_y if attack-mode=asymmetric_noise')
    parser.add_argument('--p-z', type=float, default=0.0, help='Asymmetric noise p_z if attack-mode=asymmetric_noise')
    parser.add_argument('--gamma-amp', type=float, default=0.05, help='Amplitude damping probability if attack-mode=amplitude_damping')
    parser.add_argument('--gamma-phase', type=float, default=0.05, help='Phase damping probability if attack-mode=phase_damping')
    parser.add_argument('--p-readout', type=float, default=0.03, help='Readout bit-flip probability if attack-mode=readout')
    parser.add_argument('--quantumnas-circuit', type=str, default=None,
                        help='Optional path to a Cirq JSON QuantumNAS circuit. If omitted, looks under ../quantumnas/.')
    parser.add_argument('--ignore-saboteur', action='store_true', help='Skip loading saboteur policy and use non-policy attacks.')
    parser.add_argument('--mitigation-mode', choices=[MITIGATION_NONE, MITIGATION_TWIRL, MITIGATION_RC_ZNE], default=MITIGATION_NONE,
                        help="Mitigation strategy applied during evaluation ('none', 'twirl', 'rc_zne').")
    parser.add_argument('--rc-zne-scales', type=float, nargs='+', default=None,
                        help='Noise scale factors for rc_zne mitigation (default: 1.0 1.5 2.0).')
    parser.add_argument('--rc-zne-fit', type=str, default="linear", choices=["linear", "quadratic"],
                        help="Extrapolation model for rc_zne ('linear' or 'quadratic').")
    parser.add_argument('--rc-zne-reps', type=int, default=1,
                        help='Number of RC draws averaged per scale for rc_zne extrapolation (default: 1).')
    parser.add_argument('--attack-sampling-mode', type=str, default='per_gate_uniform',
                        choices=['per_gate_uniform', 'equalized_fraction'],
                        help=('Attack sampling mode. "per_gate_uniform" keeps saboteur_budget fixed per circuit. '
                              '"equalized_fraction" scales per-circuit budget so the fraction of gates attacked is roughly equal across circuits.'))
    parser.add_argument('--skip-noise-agnostic', action='store_true',
                        help='Disable noise-family-agnostic robustness curves.')
    parser.add_argument('--noise-agnostic-points', type=int, default=16,
                        help='Number of severity points for the noise-agnostic sweep (default: 16).')
    parser.add_argument('--noise-agnostic-samples', type=int, default=8,
                        help='Number of deterministic noise samples per severity (default: 8).')
    parser.add_argument('--noise-agnostic-max-severity', type=float, default=1.3,
                        help='Maximum λ applied in the noise-agnostic sweep (default: 1.3).')
    parser.add_argument('--noise-agnostic-severity-bias', type=float, default=0.7,
                        help='Bias exponent (<1 emphasizes high noise) for λ grid spacing (default: 0.7).')
    parser.add_argument('--noise-agnostic-uniform-families', action='store_true',
                        help='Use uniform sampling over deterministic noise families (default: weighted toward decoherence).')
    args = parser.parse_args()

    compare_noise_resilience(
        base_results_dir=args.base_results_dir,
        num_runs=args.num_runs,
        n_qubits=args.n_qubits,
        samples=args.samples,
        attack_mode=args.attack_mode,
        attack_modes=args.attack_modes,
        epsilon_overrot=args.epsilon_overrot,
        p_x=args.p_x,
        p_y=args.p_y,
        p_z=args.p_z,
        gamma_amp=args.gamma_amp,
        gamma_phase=args.gamma_phase,
        p_readout=args.p_readout,
        quantumnas_circuit_path=args.quantumnas_circuit,
        ignore_saboteur=args.ignore_saboteur,
        mitigation_mode=args.mitigation_mode,
        rc_zne_scales=args.rc_zne_scales,
        rc_zne_fit=args.rc_zne_fit,
        rc_zne_reps=args.rc_zne_reps,
        attack_sampling_mode=args.attack_sampling_mode,
        noise_agnostic_enabled=not args.skip_noise_agnostic,
        noise_agnostic_points=args.noise_agnostic_points,
        noise_agnostic_samples=args.noise_agnostic_samples,
        noise_agnostic_max_severity=args.noise_agnostic_max_severity,
        noise_agnostic_severity_bias=args.noise_agnostic_severity_bias,
        noise_agnostic_uniform_families=args.noise_agnostic_uniform_families,
    )
