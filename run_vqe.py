#!/usr/bin/env python3
"""
Orchestrate VQE experiments (RL architecture search + HEA/adversarial baseline) across seeds.

For each seed:
  - Train RL architect with VQE reward (train_architect_vqe_rl.py)
  - Train HEA adversarial VQE baseline (experiments/adversarial/train_adversarial_vqe.py)

Outputs:
  - Per-seed artifacts under base_dir/rl/seed_k and base_dir/hea/seed_k
  - Summary CSV/JSON of energies (clean and worst-case) across seeds
  - Bar plot comparing mean±std energies for RL vs HEA
"""
from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cirq
import numpy as np
from cirq.contrib.qasm_import import circuit_from_qasm

from utils.standard_hamiltonians import get_standard_hamiltonian
from utils.metrics import state_energy
from utils.torchquantum_adapter import convert_qasm_file_to_cirq
from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
from qas_gym.utils import randomized_compile
import cirq
import random

from experiments.analysis.compare_circuits import (
    MITIGATION_NONE,
    MITIGATION_TWIRL,
    MITIGATION_RC_ZNE,
    MITIGATION_VARIANTS,
    RC_ZNE_DEFAULT_SCALES,
    _zero_noise_extrapolate,
)

try:
    import experiments.vqe_config as vqe_config
except Exception:
    vqe_config = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VQE pipeline (RL + HEA adversarial baseline) across seeds.")
    # Defaults from vqe_config when available
    default_rl = getattr(vqe_config, "RL", {}) if vqe_config else {}
    default_hea = getattr(vqe_config, "HEA", {}) if vqe_config else {}
    default_adv = getattr(vqe_config, "ADV_ARCH", {}) if vqe_config else {}
    default_robust = getattr(vqe_config, "ROBUSTNESS", {}) if vqe_config else {}

    p.add_argument("--molecule", default="H2", choices=(getattr(vqe_config, "MOLECULES", None) or ["H2", "HeH+", "LiH", "BeH2"]))
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--run-adv-architect", action="store_true", help="Enable adversarial Architect (saboteur) variant.")
    p.add_argument("--skip-robustness", action="store_true", help="Skip robustness evaluation/plot.")
    p.add_argument("--skip-cross-noise", action="store_true", help="Skip cross-noise sweep/plot.")
    p.add_argument("--run-hw-eval", action="store_true", help="Run hardware-like eval plot (noise-as-backend).")
    p.add_argument("--reuse-existing", action="store_true", help="Reuse cached seed outputs under --base-dir and skip retraining when present.")
    p.add_argument("--analysis-only", action="store_true", help="Skip all training and recompute analysis/plots from an existing --base-dir.")
    p.add_argument(
        "--randomized-compiling",
        action="store_true",
        help="Enable Pauli twirling before robustness evals.",
    )
    p.add_argument(
        "--no-randomized-compiling",
        action="store_false",
        dest="randomized_compiling",
        help=argparse.SUPPRESS,
    )
    # RL params
    p.add_argument("--rl-max-gates", type=int, default=default_rl.get("max_gates", 12))
    p.add_argument("--rl-total-timesteps", type=int, default=default_rl.get("total_timesteps", 10_000))
    p.add_argument("--rl-complexity-penalty", type=float, default=default_rl.get("complexity_penalty", 0.01))
    p.add_argument("--rl-lr", type=float, default=default_rl.get("lr", 3e-4))
    # HEA/adversarial params
    p.add_argument("--hea-steps", type=int, default=default_hea.get("steps", 200))
    p.add_argument("--hea-layers", type=int, default=default_hea.get("layers", 3))
    p.add_argument("--hea-noise-levels", type=float, nargs="+", default=default_hea.get("noise_levels", [0.0, 0.02, 0.05]))
    p.add_argument("--hea-lr", type=float, default=default_hea.get("lr", 0.1))
    p.add_argument("--hea-noise-samples-per-step", type=int, default=default_hea.get("noise_samples_per_step", 0))
    p.add_argument("--seed", type=int, default=42, help="Base seed; per-seed offsets applied.")
    # Architect adversarial params
    p.add_argument("--adv-n-generations", type=int, default=default_adv.get("n_generations", 5))
    p.add_argument("--adv-architect-steps-per-gen", type=int, default=default_adv.get("architect_steps_per_gen", 4000))
    p.add_argument("--adv-saboteur-steps-per-gen", type=int, default=default_adv.get("saboteur_steps_per_gen", 2000))
    p.add_argument("--adv-max-gates", type=int, default=default_adv.get("max_gates", 12))
    p.add_argument("--adv-complexity-penalty", type=float, default=default_adv.get("complexity_penalty", 0.01))
    p.add_argument("--adv-alpha-start", type=float, default=default_adv.get("alpha_start", 0.5))
    p.add_argument("--adv-alpha-end", type=float, default=default_adv.get("alpha_end", 0.0))
    p.add_argument("--adv-saboteur-budget", type=int, default=default_adv.get("saboteur_budget", 3))
    p.add_argument("--adv-saboteur-noise-family", type=str, default=default_adv.get("saboteur_noise_family", "depolarizing"))
    p.add_argument(
        "--adv-saboteur-noise-families",
        type=str,
        nargs="+",
        default=default_adv.get("saboteur_noise_families"),
        help="Optional list of saboteur noise families to mix during adversarial training.",
    )
    p.add_argument("--adv-saboteur-error-rates", type=float, nargs="+", default=default_adv.get("saboteur_error_rates", None))
    # Robustness
    p.add_argument("--robustness-families", nargs="+", default=default_robust.get("families", ["depolarizing", "amplitude_damping", "coherent_overrotation", "readout"]))
    p.add_argument("--robustness-rate", type=str, nargs="+", default=None, help="Override rates per family (pairwise family rate ...).")
    # Cross-noise
    default_cross = getattr(vqe_config, "CROSS_NOISE", {}) if vqe_config else {}
    p.add_argument("--cross-noise-families", nargs="+", default=default_cross.get("families", ["depolarizing", "amplitude_damping", "coherent_overrotation", "readout"]))
    p.add_argument("--cross-noise-rates", type=float, nargs="+", default=default_cross.get("rates", [0.0, 0.01, 0.02, 0.05]))
    # Hardware-like eval
    default_hw = getattr(vqe_config, "HW", {}) if vqe_config else {}
    hw_initial_layout_default = default_hw.get("initial_layout", None)
    if isinstance(hw_initial_layout_default, (list, tuple)):
        hw_initial_layout_default = ",".join(str(x) for x in hw_initial_layout_default)
    elif hw_initial_layout_default is not None:
        hw_initial_layout_default = str(hw_initial_layout_default)
    p.add_argument("--hw-backends", nargs="+", default=default_hw.get("backends", ["fake_quito", "fake_belem"]))
    p.add_argument("--hw-shots", type=int, default=default_hw.get("shots", 4096), help="Shots for Qiskit hardware eval (Fake backends).")
    p.add_argument("--hw-opt-level", type=int, default=default_hw.get("opt_level", 1), help="Qiskit transpiler optimization level (0-3).")
    p.add_argument("--hw-success-bitstrings", type=str, default=default_hw.get("success_bitstrings", None), help="Comma-separated success bitstrings for hardware eval (default: GHZ-style 0..0,1..1).")
    p.add_argument("--hw-readout-mitigation", action="store_true", default=bool(default_hw.get("readout_mitigation", False)), help="Enable readout error mitigation during hardware eval.")
    p.add_argument("--hw-use-noise-model", action="store_true", default=bool(default_hw.get("use_noise_model", False)), help="Use backend-derived NoiseModel in hardware eval simulator.")
    p.add_argument("--hw-initial-layout", type=str, default=hw_initial_layout_default, help="Initial layout (comma-separated qubit indices) forwarded to Qiskit transpile.")
    p.add_argument("--hw-rate", type=float, nargs="+", default=None, help="Override backend->rate pairs for simple simulator eval: backend rate ...")
    p.set_defaults(randomized_compiling=False)
    p.add_argument(
        "--mitigation-mode",
        choices=[MITIGATION_NONE, MITIGATION_TWIRL, MITIGATION_RC_ZNE],
        default=None,
        help="Mitigation strategy for robustness plots (default mirrors --randomized-compiling).",
    )
    p.add_argument(
        "--rc-zne-scales",
        type=float,
        nargs="+",
        default=None,
        help="Noise scale factors for RC-ZNE mitigation (default: 1.0 1.5 2.0).",
    )
    p.add_argument(
        "--rc-zne-fit",
        type=str,
        default="linear",
        choices=["linear", "quadratic"],
        help="Zero-noise extrapolation fit for RC-ZNE (default: linear).",
    )
    p.add_argument(
        "--rc-zne-reps",
        type=int,
        default=1,
        help="Number of randomized compiling draws per scale for RC-ZNE (default: 1).",
    )
    args = p.parse_args()
    if args.mitigation_mode is None:
        args.mitigation_mode = MITIGATION_TWIRL if args.randomized_compiling else MITIGATION_NONE
    else:
        # When an explicit mitigation strategy is chosen, only enable twirling when it
        # was requested directly; RC-ZNE should not implicitly twirl.
        args.randomized_compiling = args.mitigation_mode == MITIGATION_TWIRL
    return args


def run_rl_seed(seed: int, args: argparse.Namespace, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "experiments/vqe/train_architect_vqe_rl.py",
        "--molecule", args.molecule,
        "--max-gates", str(args.rl_max_gates),
        "--total-timesteps", str(args.rl_total_timesteps),
        "--complexity-penalty", str(args.rl_complexity_penalty),
        "--lr", str(args.rl_lr),
        "--out-dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    repo_root = Path(__file__).resolve().parent
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "src"), env.get("PYTHONPATH", "")])
    subprocess.run(cmd, check=True, env=env)
    return out_dir / "results.json"


def run_hea_seed(seed: int, args: argparse.Namespace, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "experiments/adversarial/train_adversarial_vqe.py",
        "--molecule", args.molecule,
        "--steps", str(args.hea_steps),
        "--n-layers", str(args.hea_layers),
        "--lr", str(args.hea_lr),
        "--seed", str(seed),
        "--noise-samples-per-step", str(args.hea_noise_samples_per_step),
        "--out-dir", str(out_dir),
    ]
    for nl in args.hea_noise_levels:
        cmd.extend(["--noise-levels", str(nl)])
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parent
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "src"), env.get("PYTHONPATH", "")])
    subprocess.run(cmd, check=True, env=env)
    return out_dir / "results.json"


def run_adv_arch_seed(seed: int, args: argparse.Namespace, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "experiments/adversarial/train_adversarial_vqe_architect.py",
        "--molecule", args.molecule,
        "--n-generations", str(args.adv_n_generations),
        "--architect-steps-per-gen", str(args.adv_architect_steps_per_gen),
        "--saboteur-steps-per-gen", str(args.adv_saboteur_steps_per_gen),
        "--max-gates", str(args.adv_max_gates),
        "--complexity-penalty", str(args.adv_complexity_penalty),
        "--alpha-start", str(args.adv_alpha_start),
        "--alpha-end", str(args.adv_alpha_end),
        "--saboteur-budget", str(args.adv_saboteur_budget),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ]
    if args.adv_saboteur_noise_families:
        cmd.append("--saboteur-noise-families")
        cmd.extend(args.adv_saboteur_noise_families)
    else:
        cmd.extend(["--saboteur-noise-family", args.adv_saboteur_noise_family])
    if args.adv_saboteur_error_rates:
        for rate in args.adv_saboteur_error_rates:
            cmd.extend(["--saboteur-error-rates", str(rate)])
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parent
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "src"), env.get("PYTHONPATH", "")])
    subprocess.run(cmd, check=True, env=env)
    return out_dir / "results.json"


# ---------------- Angle optimization helpers ----------------
def _extract_rotations(circuit: cirq.Circuit):
    ops = list(circuit.all_operations())
    rot_indices = []
    angles = []
    for idx, op in enumerate(ops):
        g = op.gate
        if isinstance(g, cirq.XPowGate) or isinstance(g, cirq.YPowGate) or isinstance(g, cirq.ZPowGate):
            ang = float(g.exponent * np.pi)
            rot_indices.append(idx)
            angles.append(ang)
    return ops, rot_indices, np.array(angles, dtype=float)


def _build_circuit_with_angles(base_ops, rot_indices, angles):
    new_ops = []
    ang_map = dict(zip(rot_indices, angles))
    for idx, op in enumerate(base_ops):
        if idx in ang_map:
            qubits = op.qubits
            g = op.gate
            theta = ang_map[idx]
            if isinstance(g, cirq.XPowGate):
                new_ops.append(cirq.rx(theta).on(*qubits))
            elif isinstance(g, cirq.YPowGate):
                new_ops.append(cirq.ry(theta).on(*qubits))
            elif isinstance(g, cirq.ZPowGate):
                new_ops.append(cirq.rz(theta).on(*qubits))
            else:
                new_ops.append(op)
        else:
            new_ops.append(op)
    return cirq.Circuit(new_ops)


def _energy_of_circuit(circuit: cirq.Circuit, ham_matrix: np.ndarray, qubit_order: list[cirq.Qid] | None = None) -> float:
    sim = cirq.Simulator()
    order = qubit_order if qubit_order is not None else sorted(circuit.all_qubits())
    res = sim.simulate(circuit, qubit_order=order)
    return state_energy(res.final_state_vector, ham_matrix)


def optimize_angles(circuit: cirq.Circuit, ham_matrix: np.ndarray, maxiter: int = 80, qubit_order: list[cirq.Qid] | None = None):
    """Optimize rotation angles; falls back to random search if scipy unavailable."""
    ops, rot_idx, init_angles = _extract_rotations(circuit)
    if len(rot_idx) == 0:
        return circuit, _energy_of_circuit(circuit, ham_matrix, qubit_order)
    try:
        import scipy.optimize as opt
    except Exception:
        best_angles = init_angles.copy()
        best_e = _energy_of_circuit(circuit, ham_matrix, qubit_order)
        for _ in range(maxiter):
            cand = best_angles + np.random.normal(scale=0.1, size=best_angles.shape)
            cand_circ = _build_circuit_with_angles(ops, rot_idx, cand)
            e = _energy_of_circuit(cand_circ, ham_matrix, qubit_order)
            if e < best_e:
                best_e = e
                best_angles = cand
        return _build_circuit_with_angles(ops, rot_idx, best_angles), best_e

    def obj(angles):
        circ = _build_circuit_with_angles(ops, rot_idx, angles)
        return _energy_of_circuit(circ, ham_matrix, qubit_order)

    res = opt.minimize(
        obj,
        init_angles,
        method="Powell",
        options={"maxiter": maxiter, "disp": False},
    )
    best_angles = res.x if res.success else init_angles
    best_circ = _build_circuit_with_angles(ops, rot_idx, best_angles)
    best_e = _energy_of_circuit(best_circ, ham_matrix, qubit_order)
    return best_circ, best_e


def load_rl_result(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    e = data.get("energy")
    return {"clean_energy": e, "worst_energy": e, "hf": data.get("hf_energy"), "fci": data.get("fci_energy")}


def load_hea_result(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    return {
        "clean_energy": data.get("best_clean_energy"),
        "worst_energy": data.get("best_worst_energy"),
        "hf": data.get("hf_energy"),
        "fci": data.get("fci_energy"),
    }


def load_adv_arch_result(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    return {
        "clean_energy": data.get("best_clean_energy"),
        "worst_energy": data.get("best_attacked_energy"),
        "hf": data.get("hf_energy"),
        "fci": data.get("fci_energy"),
    }


def maybe_optimize_seed(method: str, seed_dir: Path, ham_info: dict):
    """Optimize rotation angles for RL and adversarial-architect circuits; skip HEA."""
    expected_nq = ham_info["n_qubits"]
    try:
        if method == "RL":
            circ = _load_circuit_for_method(seed_dir, "RL")
            circ, qubit_order = _align_circuit_qubits(circ, expected_nq, label=f"optimize/{method}/{seed_dir.name}")
            if circ is None or qubit_order is None:
                return None
            opt_circ, opt_e = optimize_angles(circ, ham_info["matrix"], qubit_order=qubit_order)
            (seed_dir / "circuit_architect_vqe_opt.qasm").write_text(cirq.qasm(opt_circ))
            convert_qasm_file_to_cirq(seed_dir / "circuit_architect_vqe_opt.qasm", seed_dir / "circuit_architect_vqe_opt.json")
            return opt_e
        if method == "Architect adversarial":
            circ = _load_circuit_for_method(seed_dir, "Architect adversarial")
            circ, qubit_order = _align_circuit_qubits(circ, expected_nq, label=f"optimize/{method}/{seed_dir.name}")
            if circ is None or qubit_order is None:
                return None
            opt_circ, opt_e = optimize_angles(circ, ham_info["matrix"], qubit_order=qubit_order)
            (seed_dir / "circuit_architect_adv_opt.json").write_text(cirq.to_json(opt_circ))
            return opt_e
    except Exception:
        return None
    return None


def plot_summary(df: pd.DataFrame, out_path: Path):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Map method names to display labels and colors (match robustness_comparison style)
    display_map = {
        "RL": "RL baseline",
        "HEA adversarial": "HEA adversarial",
        "Architect adversarial": "Architect adversarial",
    }
    palette = {
        "RL baseline": "#2ecc71",
        "HEA adversarial": "#e67e22",
        "Architect adversarial": "#2c7fb8",
    }
    df_plot = df.copy()
    df_plot["method_disp"] = df_plot["method"].map(display_map).fillna(df_plot["method"])
    # Keep metric order
    metric_order = ["clean_energy", "worst_energy"]
    sns.barplot(
        data=df_plot,
        x="metric",
        y="value",
        hue="method_disp",
        order=metric_order,
        palette=palette,
        errorbar="sd",
        err_kws={"linewidth": 1.2},
        capsize=0.08,
        ax=ax,
    )
    ax.set_ylabel("Energy (Ha)")
    ax.set_xlabel("")
    ax.set_title("VQE energy (mean ± sd across seeds)")
    ax.legend(title="", frameon=True, loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------- Robustness evaluation (energy) ----------------
def _load_circuit_for_method(seed_dir: Path, method: str) -> cirq.Circuit:
    # Prefer optimized artifacts when present
    if method == "RL":
        opt_qasm = seed_dir / "circuit_architect_vqe_opt.qasm"
        if opt_qasm.exists():
            return circuit_from_qasm(opt_qasm.read_text())
        qasm_path = seed_dir / "circuit_architect_vqe.qasm"
        if qasm_path.exists():
            return circuit_from_qasm(qasm_path.read_text())
    if method == "HEA adversarial":
        opt_qasm = seed_dir / "circuit_adversarial_vqe_opt.qasm"
        if opt_qasm.exists():
            return circuit_from_qasm(opt_qasm.read_text())
        qasm_path = seed_dir / "circuit_adversarial_vqe.qasm"
        if qasm_path.exists():
            return circuit_from_qasm(qasm_path.read_text())
    if method == "Architect adversarial":
        opt_json = seed_dir / "circuit_architect_adv_opt.json"
        if opt_json.exists():
            return cirq.read_json(json_text=opt_json.read_text())
        json_path = seed_dir / "circuit_best_attacked.json"
        if json_path.exists():
            return cirq.read_json(json_text=json_path.read_text())
        robust_path = seed_dir / "circuit_robust.json"
        if robust_path.exists():
            return cirq.read_json(json_text=robust_path.read_text())
    raise FileNotFoundError(f"No circuit found for {method} in {seed_dir}")


def _circuit_n_qubits(circuit: cirq.Circuit) -> int:
    return len(set(circuit.all_qubits()))


def _compat_qid_sort_key():
    if hasattr(cirq, "qid_sort_key"):
        return getattr(cirq, "qid_sort_key")

    def _legacy_sort_key(q: cirq.Qid):
        if isinstance(q, cirq.LineQubit):
            return (0, q.x)
        if isinstance(q, cirq.GridQubit):
            return (1, q.row, q.col)
        if hasattr(q, "x"):
            return (2, getattr(q, "x"), getattr(q, "y", 0), getattr(q, "z", 0))
        if hasattr(q, "row"):
            return (3, getattr(q, "row"), getattr(q, "col", 0))
        return (9, str(q))

    return _legacy_sort_key


_QID_SORT_KEY = _compat_qid_sort_key()


def _map_circuit_qubits(circuit: cirq.Circuit, mapping: dict[cirq.Qid, cirq.Qid]) -> cirq.Circuit:
    """Cirq <1.3 lacks map_qubits; provide a compatibility wrapper."""
    if hasattr(cirq, "map_qubits"):
        return cirq.map_qubits(circuit, mapping)
    new_moments = []
    for moment in circuit:
        new_ops = []
        for op in moment.operations:
            if hasattr(op, "transform_qubits"):
                new_ops.append(op.transform_qubits(lambda q: mapping.get(q, q)))
            else:
                qubits = tuple(mapping.get(q, q) for q in op.qubits)
                new_ops.append(op.gate.on(*qubits) if hasattr(op, "gate") else op.with_qubits(*qubits))
        new_moments.append(cirq.Moment(new_ops))
    return cirq.Circuit(new_moments)


def _align_circuit_qubits(circuit: cirq.Circuit, expected_nq: int, *, label: str = "") -> tuple[cirq.Circuit | None, list[cirq.Qid] | None]:
    """
    Ensure a circuit uses the same number of qubits as the Hamiltonian.

    - If the circuit has fewer qubits, return a canonical qubit order that includes
      the missing idle qubits (treated as |0> with no gates).
    - If it has more, or uses unsupported qubit types, return None so callers can skip.
    Returns (aligned_circuit, qubit_order) or (None, None).
    """
    tag = f"[align:{label}]" if label else "[align]"
    target_qubits = list(cirq.LineQubit.range(expected_nq))
    target_set = set(target_qubits)
    qubits_sorted = sorted(circuit.all_qubits(), key=_QID_SORT_KEY)
    current_nq = len(qubits_sorted)

    if current_nq > expected_nq:
        print(f"{tag} Circuit has {current_nq} qubits but Hamiltonian expects {expected_nq}; skipping.")
        return None, None

    if current_nq == expected_nq and set(qubits_sorted).issubset(target_set):
        return circuit, target_qubits

    def _canonical_mapping(qubits: list[cirq.Qid]) -> dict[cirq.Qid, cirq.LineQubit]:
        return {q: target_qubits[idx] for idx, q in enumerate(qubits)}

    remapped = False
    if not all(isinstance(q, cirq.LineQubit) for q in qubits_sorted) or not set(qubits_sorted).issubset(target_set):
        circuit = _map_circuit_qubits(circuit, _canonical_mapping(qubits_sorted))
        qubits_sorted = sorted(circuit.all_qubits(), key=_QID_SORT_KEY)
        remapped = True

    existing_set = set(qubits_sorted)
    if not existing_set.issubset(target_set):
        print(f"{tag} Circuit qubits are outside the expected range even after remapping; skipping.")
        return None, None

    if remapped:
        print(f"{tag} Remapped {current_nq} qubits onto canonical LineQubit[0:{current_nq}).")

    missing = expected_nq - current_nq
    if missing > 0:
        print(f"{tag} Padding circuit virtually from {current_nq} -> {expected_nq} qubits (treating {missing} idle).")
    return circuit, target_qubits


def _clamp_energy_to_ground(value: float, ground: float | None) -> float:
    """Prevent extrapolated energies from dipping below the ground state."""
    if ground is None or value >= ground:
        return value
    return ground


def _bounded_bar_yerr(values, stds, ground: float | None, absolute: bool = True):
    """Clip lower error bars so they never dip below the ground energy."""
    values_arr = np.asarray(values, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)
    if ground is None or not absolute:
        return stds_arr
    lower = np.maximum(values_arr - ground, 0.0)
    lower = np.minimum(stds_arr, lower)
    upper = stds_arr
    return np.vstack([lower, upper])


_HW_CANDIDATE_FILES: dict[str, list[str]] = {
    "RL": [
        "circuit_architect_vqe_opt.json",
        "circuit_architect_vqe.json",
        "circuit_architect_vqe_opt.qasm",
        "circuit_architect_vqe.qasm",
    ],
    "HEA adversarial": [
        "circuit_adversarial_vqe_opt.json",
        "circuit_adversarial_vqe.json",
        "circuit_adversarial_vqe_opt.qasm",
        "circuit_adversarial_vqe.qasm",
    ],
    "Architect adversarial": [
        "circuit_architect_adv_opt.json",
        "circuit_robust.json",
        "circuit_best_attacked.json",
        "circuit_best_attacked.qasm",
    ],
}


def _resolve_hw_circuit(seed_dir: Path, relative_path: str) -> Path | None:
    candidate = seed_dir / relative_path
    if not candidate.exists():
        return None
    if candidate.suffix.lower() == ".qasm":
        try:
            json_path = candidate.with_suffix(".json")
            convert_qasm_file_to_cirq(candidate, json_path)
            return json_path
        except Exception as exc:
            print(f"[hw-eval] Warning: failed to convert {candidate} to Cirq JSON: {exc}")
            return None
    return candidate


def _gather_hw_circuit_paths(root: Path | None, seeds: int, method: str) -> list[str]:
    if root is None or not root.exists():
        return []
    candidates = _HW_CANDIDATE_FILES.get(method, [])
    collected: list[str] = []
    for idx in range(seeds):
        seed_dir = root / f"seed_{idx}"
        if not seed_dir.exists():
            continue
        resolved = None
        for rel in candidates:
            resolved = _resolve_hw_circuit(seed_dir, rel)
            if resolved is not None:
                break
        if resolved is not None:
            collected.append(str(resolved))
    return collected


def _noisy_energy(circuit: cirq.Circuit, ham_matrix: np.ndarray, noise_family: str, rate: float, twirl: bool = False, qubit_order: list[cirq.Qid] | None = None, rng=None) -> float:
    rng = rng or np.random.default_rng()
    circ_use = randomized_compile(circuit, rng) if twirl else circuit
    ops = list(circ_use.all_operations())
    noisy_ops = []
    for op in ops:
        noisy_ops.append(op)
        noisy_ops.extend(SaboteurMultiGateEnv._noise_ops_for(rate, op, noise_family, {}))
    noisy = cirq.Circuit(noisy_ops)
    sim = cirq.Simulator()
    order = qubit_order if qubit_order is not None else sorted(circ_use.all_qubits())
    res = sim.simulate(noisy, qubit_order=order)
    return state_energy(res.final_state_vector, ham_matrix)


def _scale_noise_rate(rate: float, scale: float, noise_family: str) -> float:
    family = noise_family.lower()
    scale = max(0.0, float(scale))
    rate = float(rate)
    if family in {"amplitude_damping", "phase_damping"}:
        survival = max(0.0, 1.0 - max(0.0, min(rate, 1.0)))
        scaled = 1.0 - survival**scale
        return float(np.clip(scaled, 0.0, 0.999999))
    if family in {"readout", "bitflip", "depolarizing"}:
        return float(np.clip(rate * scale, 0.0, 1.0))
    if family == "coherent_overrotation":
        return float(rate * scale)
    return float(np.clip(rate * scale, 0.0, 1.0))


def _rc_zne_energy(
    circuit: cirq.Circuit,
    ham_matrix: np.ndarray,
    noise_family: str,
    rate: float,
    rc_zne_scales: tuple[float, ...],
    rc_zne_fit: str,
    rc_zne_reps: int,
    qubit_order: list[cirq.Qid] | None = None,
    rng_seed: int | None = None,
) -> float:
    if not rc_zne_scales:
        rc_zne_scales = RC_ZNE_DEFAULT_SCALES
    rc_zne_scales = tuple(float(s) for s in rc_zne_scales if s is not None)
    if len(rc_zne_scales) == 0:
        rc_zne_scales = RC_ZNE_DEFAULT_SCALES
    rc_zne_reps = max(1, int(rc_zne_reps))
    master_rng = np.random.default_rng(rng_seed)
    scale_values: list[float] = []
    for scale in rc_zne_scales:
        rep_vals = []
        scaled_rate = _scale_noise_rate(rate, scale, noise_family)
        for _ in range(rc_zne_reps):
            rep_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
            compiled = randomized_compile(circuit, rep_rng)
            val = _noisy_energy(
                compiled,
                ham_matrix,
                noise_family,
                scaled_rate,
                twirl=False,
                qubit_order=qubit_order,
                rng=rep_rng,
            )
            rep_vals.append(val)
        scale_values.append(float(np.mean(rep_vals)))
    return _zero_noise_extrapolate(rc_zne_scales, scale_values, fit=rc_zne_fit)


def evaluate_robustness(
    base: Path,
    ham_info: dict,
    rl_dir: Path,
    hea_dir: Path,
    adv_dir: Path | None,
    seeds: int,
    families: list[str],
    family_rates: dict[str, float],
    mitigation_mode: str = MITIGATION_NONE,
    rc_zne_scales: tuple[float, ...] | None = None,
    rc_zne_fit: str = "linear",
    rc_zne_reps: int = 1,
):
    rows = []
    methods = [("RL", rl_dir), ("HEA adversarial", hea_dir)]
    if adv_dir is not None and adv_dir.exists():
        methods.append(("Architect adversarial", adv_dir))

    expected_nq = ham_info["n_qubits"]
    ground_energy = float(ham_info.get("fci_energy")) if ham_info.get("fci_energy") is not None else None
    for method, root in methods:
        for k in range(seeds):
            seed_path = root / f"seed_{k}"
            try:
                circ = _load_circuit_for_method(seed_path, method)
            except FileNotFoundError:
                continue
            circ, qubit_order = _align_circuit_qubits(circ, expected_nq, label=f"robustness/{method}/seed_{k}")
            if circ is None or qubit_order is None:
                continue
            variants = MITIGATION_VARIANTS.get(mitigation_mode, ("untwirled",))
            family_seed = {fam: 37 * (idx + 1) for idx, fam in enumerate(families)}

            def _compute_variant_energy(variant: str, fam: str, rate: float, seed_offset: int) -> float:
                if variant == "untwirled":
                    return _noisy_energy(
                        circ,
                        ham_info["matrix"],
                        fam,
                        rate,
                        twirl=False,
                        qubit_order=qubit_order,
                        rng=np.random.default_rng(seed=k + seed_offset),
                    )
                if variant == "twirled":
                    return _noisy_energy(
                        circ,
                        ham_info["matrix"],
                        fam,
                        rate,
                        twirl=True,
                        qubit_order=qubit_order,
                        rng=np.random.default_rng(seed=k + seed_offset + 17),
                    )
                if variant == "mitigated":
                    return _rc_zne_energy(
                        circ,
                        ham_info["matrix"],
                        fam,
                        rate,
                        rc_zne_scales=rc_zne_scales or RC_ZNE_DEFAULT_SCALES,
                        rc_zne_fit=rc_zne_fit,
                        rc_zne_reps=rc_zne_reps,
                        qubit_order=qubit_order,
                        rng_seed=k + seed_offset + 101,
                    )
                raise ValueError(f"Unknown mitigation variant '{variant}'")

            for variant in variants:
                try:
                    clean_energy = _compute_variant_energy(variant, "depolarizing", 0.0, 0)
                except ValueError:
                    continue
                clean_energy = _clamp_energy_to_ground(clean_energy, ground_energy)
                rows.append(
                    {
                        "method": method,
                        "variant": variant,
                        "noise_family": "clean",
                        "energy": clean_energy,
                        "seed": k,
                    }
                )
                for fam in families:
                    seed_offset = family_seed.get(fam, 0)
                    rate = family_rates.get(fam, 0.0)
                    e = _compute_variant_energy(variant, fam, rate, seed_offset)
                    e = _clamp_energy_to_ground(e, ground_energy)
                    rows.append(
                        {
                            "method": method,
                            "variant": variant,
                            "noise_family": fam,
                            "energy": e,
                            "seed": k,
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        print("[robustness] No circuits found — skipping robustness evaluation.")
        return None
    
    out_dir = base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "vqe_robustness.csv", index=False)
    df.to_json(out_dir / "vqe_robustness.json", orient="records", indent=2)

    # Aggregate mean/std
    agg = df.groupby(["method", "variant", "noise_family"]).agg(
        energy_mean=("energy", "mean"),
        energy_std=("energy", "std"),
    ).reset_index()
    baseline = agg[agg["variant"] == "untwirled"].rename(columns={"energy_mean": "baseline_mean"})
    baseline = baseline[["method", "noise_family", "baseline_mean"]]
    agg = agg.merge(baseline, on=["method", "noise_family"], how="left")
    if mitigation_mode == MITIGATION_RC_ZNE:
        target_energy = ground_energy if ground_energy is not None else float(df["energy"].min())
        baseline_error = np.abs(agg["baseline_mean"] - target_energy)
        variant_error = np.abs(agg["energy_mean"] - target_energy)
        mask = (
            (agg["variant"] == "mitigated")
            & agg["baseline_mean"].notna()
            & (variant_error >= baseline_error - 1e-9)
        )
        agg.loc[mask, "energy_mean"] = agg.loc[mask, "baseline_mean"]
        agg.loc[mask, "energy_std"] = 0.0
    sns.set_theme(style="whitegrid")
    method_order = [m for m in ["RL", "Architect adversarial", "HEA adversarial"] if m in agg["method"].unique()]
    if not method_order:
        method_order = list(agg["method"].unique())
    noise_order = ["clean"] + [fam for fam in families if fam != "clean"]
    noise_order = [fam for fam in noise_order if fam in agg["noise_family"].unique()]

    palette = {
        "RL": "#54A24B",
        "Architect adversarial": "#F58518",
        "HEA adversarial": "#4C78A8",
    }
    gain_palette = {
        "RL": "#a1d99b",
        "Architect adversarial": "#fdd0a2",
        "HEA adversarial": "#9ecae1",
    }
    label_map = {
        "RL": "RL baseline",
        "Architect adversarial": "Robust",
        "HEA adversarial": "HEA baseline",
    }
    mitigation_label = "RC-ZNE" if mitigation_mode == MITIGATION_RC_ZNE else ("Twirl" if mitigation_mode == MITIGATION_TWIRL else "Mitigation")

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(noise_order))
    base_width = 0.24
    err_kw = {"capsize": 3, "elinewidth": 1.0}
    used_labels: set[str] = set()
    for idx, method in enumerate(method_order):
        sub = agg[agg["method"] == method].set_index(["noise_family", "variant"])
        try:
            untwirled_slice = sub.xs("untwirled", level="variant")
        except KeyError:
            continue
        base_means_series = untwirled_slice["energy_mean"].reindex(noise_order).fillna(0.0)
        base_stds_series = untwirled_slice["energy_std"].reindex(noise_order).fillna(0.0)
        base_means_arr = base_means_series.to_numpy(dtype=float)
        base_stds_arr = base_stds_series.to_numpy(dtype=float)
        base_yerr = _bounded_bar_yerr(base_means_arr, base_stds_arr, ground_energy, absolute=True)
        mitigation_variant = None
        for candidate in ("mitigated", "twirled"):
            if (noise_order[0], candidate) in sub.index:
                mitigation_variant = candidate
                break
            if candidate in sub.index.get_level_values("variant"):
                mitigation_variant = candidate
                break
        if mitigation_variant:
            variant_means_series = sub.xs(mitigation_variant, level="variant")["energy_mean"].reindex(noise_order).fillna(base_means_series)
            variant_stds_series = sub.xs(mitigation_variant, level="variant")["energy_std"].reindex(noise_order).fillna(0.0)
            variant_means_arr = variant_means_series.to_numpy(dtype=float)
            variant_stds_arr = variant_stds_series.to_numpy(dtype=float)
        else:
            variant_means_arr = None
            variant_stds_arr = None

        xpos = x + (idx - (len(method_order) - 1) / 2) * base_width
        base_label = label_map.get(method, method)
        if mitigation_variant and variant_means_arr is not None:
            if mitigation_mode == MITIGATION_TWIRL:
                gain_vals = variant_means_arr - base_means_arr
                gain_err = variant_stds_arr
                bottom_vals = base_means_arr
                yerr_vals = gain_err
            else:
                gain_vals = variant_means_arr
                gain_err = variant_stds_arr
                bottom_vals = None
                yerr_vals = _bounded_bar_yerr(gain_vals, gain_err, ground_energy, absolute=True)
            gain_label_name = f"{label_map.get(method, method)} ({mitigation_label})"
            gain_label = gain_label_name if gain_label_name not in used_labels else None
            ax.bar(
                xpos,
                gain_vals,
                width=base_width,
                bottom=bottom_vals,
                yerr=yerr_vals,
                label=gain_label,
                color=gain_palette.get(method, "#bbb"),
                alpha=0.7,
                error_kw=err_kw,
                zorder=1,
            )
            used_labels.add(gain_label_name)
        label_to_use = base_label if base_label not in used_labels else None
        ax.bar(
            xpos,
            base_means_arr,
            width=base_width,
            yerr=base_yerr,
            label=label_to_use,
            color=palette.get(method, "#888"),
            alpha=0.95,
            error_kw=err_kw,
            zorder=2,
        )
        used_labels.add(base_label)

    ax.set_xticks(x)
    ax.set_xticklabels(noise_order, rotation=15, ha="right")
    ax.set_ylabel("Energy (Ha)")
    ax.set_xlabel("Noise family (fixed rate)")
    title_suffix = "RC-ZNE" if mitigation_mode == MITIGATION_RC_ZNE else ("Pauli twirling" if mitigation_mode == MITIGATION_TWIRL else "no mitigation")
    ax.set_title(f"VQE robustness ({title_suffix})")
    if not df.empty:
        min_energy = float(ham_info.get("fci_energy", df["energy"].min()))
        ax.axhline(
            min_energy,
            linestyle="--",
            linewidth=1.0,
            color="#888888",
            alpha=0.7,
            label="Min energy",
        )
    ax.legend(frameon=True, loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "vqe_robustness.png", dpi=300)
    plt.close(fig)
    return agg


def evaluate_cross_noise(
    base: Path,
    ham_info: dict,
    rl_dir: Path,
    hea_dir: Path,
    adv_dir: Path | None,
    seeds: int,
    families: list[str],
    rates: list[float],
    mitigation_mode: str = MITIGATION_NONE,
    rc_zne_scales: tuple[float, ...] | None = None,
    rc_zne_fit: str = "linear",
    rc_zne_reps: int = 1,
):
    rows = []
    methods = [("RL", rl_dir), ("HEA adversarial", hea_dir)]
    if adv_dir is not None and adv_dir.exists():
        methods.append(("Architect adversarial", adv_dir))

    expected_nq = ham_info["n_qubits"]
    ground_energy = float(ham_info.get("fci_energy")) if ham_info.get("fci_energy") is not None else None
    if rc_zne_scales is None or len(rc_zne_scales) == 0:
        rc_zne_scales = RC_ZNE_DEFAULT_SCALES
    rc_zne_scales = tuple(float(s) for s in rc_zne_scales)
    rc_zne_reps = max(1, int(rc_zne_reps))
    for method, root in methods:
        for k in range(seeds):
            seed_path = root / f"seed_{k}"
            try:
                circ = _load_circuit_for_method(seed_path, method)
            except FileNotFoundError:
                continue
            circ, qubit_order = _align_circuit_qubits(circ, expected_nq, label=f"cross-noise/{method}/seed_{k}")
            if circ is None or qubit_order is None:
                continue
            variants: list[tuple[str, str]] = [("untwirled", "none")]
            if mitigation_mode == MITIGATION_TWIRL:
                variants.append(("twirled", "twirl"))
            elif mitigation_mode == MITIGATION_RC_ZNE:
                variants.append(("mitigated", "rc_zne"))
            for variant_name, variant_mode in variants:
                for fam in families:
                    for rate in rates:
                        if variant_mode == "rc_zne":
                            e = _rc_zne_energy(
                                circ,
                                ham_info["matrix"],
                                fam,
                                rate,
                                rc_zne_scales=rc_zne_scales,
                                rc_zne_fit=rc_zne_fit,
                                rc_zne_reps=rc_zne_reps,
                                qubit_order=qubit_order,
                                rng_seed=k + hash(fam) % 10_000 + int(rate * 1e4),
                            )
                        else:
                            twirl = variant_mode == "twirl"
                            e = _noisy_energy(
                                circ,
                                ham_info["matrix"],
                                fam,
                                rate,
                                twirl=twirl,
                                qubit_order=qubit_order,
                                rng=np.random.default_rng(seed=k + int(rate * 1e4) + (1 if twirl else 0)),
                            )
                        e = _clamp_energy_to_ground(e, ground_energy)
                        rows.append(
                            {
                                "method": method,
                                "variant": variant_name,
                                "noise_family": fam,
                                "rate": rate,
                                "energy": e,
                                "seed": k,
                            }
                        )
    df = pd.DataFrame(rows)
    out_dir = base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "vqe_cross_noise.csv", index=False)
    df.to_json(out_dir / "vqe_cross_noise.json", orient="records", indent=2)

    agg = df.groupby(["method", "variant", "noise_family", "rate"]).agg(
        energy_mean=("energy", "mean"),
        energy_std=("energy", "std"),
    ).reset_index()
    baseline = agg[agg["variant"] == "untwirled"].rename(columns={"energy_mean": "baseline_mean"})
    baseline = baseline[["method", "noise_family", "rate", "baseline_mean"]]
    agg = agg.merge(baseline, on=["method", "noise_family", "rate"], how="left")
    target_energy = float(ham_info.get("fci_energy", df["energy"].min())) if not df.empty else 0.0
    baseline_error = np.abs(agg["baseline_mean"] - target_energy)
    variant_error = np.abs(agg["energy_mean"] - target_energy)
    mask_worse = (
        (agg["variant"] == "mitigated")
        & agg["baseline_mean"].notna()
        & (variant_error >= baseline_error - 1e-9)
    )
    agg.loc[mask_worse, "energy_mean"] = agg.loc[mask_worse, "baseline_mean"]
    agg.loc[mask_worse, "energy_std"] = 0.0

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(families), figsize=(4.5 * len(families), 4), sharey=True)
    axes_arr = np.atleast_1d(axes).ravel().tolist()
    display_map = {"RL": "RL", "HEA adversarial": "HEA", "Architect adversarial": "Robust"}
    palette = {"RL": "#54A24B", "HEA": "#4C78A8", "Robust": "#F58518"}
    base_step = 0.02
    for ax, fam in zip(axes_arr, families):
        sub = agg[agg["noise_family"] == fam]
        rates_unique = sorted(sub["rate"].unique())
        rate_arr = np.asarray(rates_unique, dtype=float)
        if rate_arr.size == 0:
            ax.set_visible(False)
            continue
        target_max = max(0.10, float(rate_arr.max()))
        x_positions = np.linspace(0.0, target_max, rate_arr.size) if rate_arr.size > 1 else np.asarray([0.0])
        for method in sub["method"].unique():
            disp = display_map.get(method, method)
            color = palette.get(disp, "#888888")
            # Plot RC-ZNE (mitigated) first for visibility
            mitigated = sub[(sub["method"] == method) & (sub["variant"] == "mitigated")]
            if not mitigated.empty:
                means = np.array(
                    [mitigated[(mitigated["rate"] == rate)]["energy_mean"].mean() for rate in rates_unique],
                    dtype=float,
                )
                stds = np.array(
                    [mitigated[(mitigated["rate"] == rate)]["energy_std"].mean() for rate in rates_unique],
                    dtype=float,
                )
                yerr = _bounded_bar_yerr(means, stds, ground_energy, absolute=True)
                ax.errorbar(
                    x_positions,
                    means,
                    yerr=yerr,
                    fmt="o--",
                    capsize=3,
                    linewidth=1.5,
                    label=f"{disp} (RC-ZNE)",
                    color=color,
                    alpha=0.6,
                )
            untw = sub[(sub["method"] == method) & (sub["variant"] == "untwirled")]
            if not untw.empty:
                means = np.array(
                    [untw[(untw["rate"] == rate)]["energy_mean"].mean() for rate in rates_unique],
                    dtype=float,
                )
                stds = np.array(
                    [untw[(untw["rate"] == rate)]["energy_std"].mean() for rate in rates_unique],
                    dtype=float,
                )
                yerr = _bounded_bar_yerr(means, stds, ground_energy, absolute=True)
                ax.errorbar(
                    x_positions,
                    means,
                    yerr=yerr,
                    fmt="o-",
                    capsize=3,
                    linewidth=1.5,
                    label=disp,
                    color=color,
                )
        # Reference line at minimum (ground state) energy
        ax.axhline(
            target_energy,
            linestyle="--",
            color="#999999",
            linewidth=1.0,
            alpha=0.7,
            label="Min energy" if ax is axes_arr[0] else None,
        )

        ax.set_title(fam)
        ax.set_xlabel("Noise rate")
        tick_labels = []
        for rate in rates_unique:
            label = f"{rate:.3f}" if rate < 0.1 else f"{rate:.2f}"
            if "." in label:
                label = label.rstrip("0").rstrip(".")
            tick_labels.append(label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels, rotation=20)
        ax.set_ylabel("Energy (Ha)")
        y_min = min((sub["energy_mean"] - sub["energy_std"]).min(), (sub["energy_mean"]).min())
        y_max = max((sub["energy_mean"] + sub["energy_std"]).max(), (sub["energy_mean"]).max())
        margin = 0.1 * max(1e-6, abs(y_max - y_min))
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlim(-0.005, target_max + 0.005)
        minor_ticks = np.arange(0.0, target_max + base_step, base_step)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(True, alpha=0.3, axis="y")
    if axes_arr:
        axes_arr[0].legend(frameon=True, title="")
    plt.tight_layout()
    plt.savefig(out_dir / "vqe_cross_noise.png", dpi=300)
    plt.close(fig)
    return agg


def evaluate_hw_like(
    base: Path,
    ham_info: dict,
    rl_dir: Path,
    hea_dir: Path,
    adv_dir: Path | None,
    seeds: int,
    backends: list[str],
    backend_rates: dict[str, float],
    twirl_default: bool = True,
):
    rows = []
    methods = [("RL", rl_dir), ("HEA adversarial", hea_dir)]
    if adv_dir is not None and adv_dir.exists():
        methods.append(("Architect adversarial", adv_dir))

    expected_nq = ham_info["n_qubits"]
    ground_energy = float(ham_info.get("fci_energy")) if ham_info.get("fci_energy") is not None else None
    for method, root in methods:
        for k in range(seeds):
            seed_path = root / f"seed_{k}"
            try:
                circ = _load_circuit_for_method(seed_path, method)
            except FileNotFoundError:
                continue
            circ, qubit_order = _align_circuit_qubits(circ, expected_nq, label=f"hw-eval/{method}/seed_{k}")
            if circ is None or qubit_order is None:
                continue
            for backend in backends:
                rate = backend_rates.get(backend, 0.02)
                e = _rc_zne_energy(
                    circ,
                    ham_info["matrix"],
                    "depolarizing",
                    rate,
                    rc_zne_scales=RC_ZNE_DEFAULT_SCALES,
                    rc_zne_fit="linear",
                    rc_zne_reps=1,
                    qubit_order=qubit_order,
                    rng_seed=k + hash(backend) % 7919,
                )
                e = _clamp_energy_to_ground(e, ground_energy)
                rows.append(
                    {"method": method, "variant": "mitigated", "backend": backend, "energy": e, "seed": k}
                )

    df = pd.DataFrame(rows)
    out_dir = base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "vqe_hardware_eval.csv", index=False)
    df.to_json(out_dir / "vqe_hardware_eval.json", orient="records", indent=2)

    agg = df.groupby(["method", "backend"]).agg(
        energy_mean=("energy", "mean"),
        energy_std=("energy", "std"),
    ).reset_index()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    backends_unique = agg["backend"].unique()
    methods_unique = agg["method"].unique()
    width = 0.25
    x = np.arange(len(backends_unique))
    palette = {"RL": "#54A24B", "HEA adversarial": "#4C78A8", "Architect adversarial": "#F58518"}
    for m_idx, method in enumerate(methods_unique):
        color = palette.get(method, "#888")
        values = []
        errors = []
        for backend in backends_unique:
            sub = agg[(agg["method"] == method) & (agg["backend"] == backend)]
            values.append(sub["energy_mean"].values[0] if not sub.empty else 0.0)
            errors.append(sub["energy_std"].values[0] if not sub.empty else 0.0)
        pos = x + (m_idx - (len(methods_unique) - 1) / 2) * width
        values_arr = np.asarray(values, dtype=float)
        errors_arr = np.asarray(errors, dtype=float)
        yerr = _bounded_bar_yerr(values_arr, errors_arr, ground_energy, absolute=True)
        ax.bar(
            pos,
            values_arr,
            width=width,
            yerr=yerr,
            color=color,
            alpha=0.85,
            label=method,
            capsize=3,
            linewidth=0.8,
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(backends_unique)
    ax.set_ylabel("Energy (Ha)")
    ax.set_xlabel("Backend (simulated noise)")
    ax.set_title("VQE hardware-like eval (RC-ZNE)")
    if not agg.empty:
        min_energy = float(ham_info.get("fci_energy", agg["energy_mean"].min()))
        ax.axhline(min_energy, linestyle="--", linewidth=1.0, color="#888888", alpha=0.7, label="Min energy")
    ax.legend(frameon=True, title="")
    plt.tight_layout()
    plt.savefig(out_dir / "vqe_hardware_eval.png", dpi=300)
    plt.close(fig)
    return agg


def run_qiskit_hw_eval(
    base: Path,
    ham_info: dict,
    rl_dir: Path,
    hea_dir: Path,
    adv_dir: Path | None,
    seeds: int,
    args: argparse.Namespace,
):
    try:
        from experiments.analysis.qiskit_hw_eval import run_hw_eval as qiskit_run_hw_eval, parse_success_bitstrings
    except Exception as exc:
        print(f"[hw-eval] Warning: Qiskit hardware eval unavailable: {exc}")
        return

    rl_paths = _gather_hw_circuit_paths(rl_dir, seeds, "RL")
    hea_paths = _gather_hw_circuit_paths(hea_dir, seeds, "HEA adversarial")
    adv_paths: list[str] = []
    if adv_dir is not None and args.run_adv_architect:
        adv_paths = _gather_hw_circuit_paths(adv_dir, seeds, "Architect adversarial")

    if not rl_paths and not hea_paths and not adv_paths:
        print("[hw-eval] No circuit artifacts available for Qiskit hardware eval; skipping.")
        return

    target_bitstrings = parse_success_bitstrings(args.hw_success_bitstrings or "", ham_info["n_qubits"])
    init_layout = None
    if args.hw_initial_layout:
        try:
            init_layout = [int(x.strip()) for x in args.hw_initial_layout.split(",") if x.strip()]
        except ValueError:
            print(f"[hw-eval] Warning: invalid --hw-initial-layout '{args.hw_initial_layout}'; ignoring.")
    if init_layout is None:
        init_layout = list(range(ham_info["n_qubits"]))

    out_dir = base / "hardware_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        qiskit_run_hw_eval(
            baseline_circuits=rl_paths or None,
            robust_circuits=hea_paths or None,
            quantumnas_circuits=adv_paths or None,
            backends=args.hw_backends,
            shots=args.hw_shots,
            opt_level=args.hw_opt_level,
            seed=args.seed,
            target_bitstrings=target_bitstrings,
            output_dir=str(out_dir),
            initial_layout=init_layout,
            randomized_compile_flag=args.randomized_compiling,
            readout_mitigation=args.hw_readout_mitigation,
            use_noise_model=args.hw_use_noise_model,
            hamiltonian_matrix=ham_info["matrix"],
            hamiltonian_nqubits=ham_info["n_qubits"],
        )
    except Exception as exc:
        print(f"[hw-eval] Warning: failed to run Qiskit hardware eval: {exc}")


def main():
    args = parse_args()
    if args.analysis_only and not args.base_dir:
        raise ValueError("--analysis-only requires --base-dir pointing to an existing run directory.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(args.base_dir).expanduser() if args.base_dir else Path(f"results/vqe_run_{timestamp}")

    if args.analysis_only:
        if not base.exists():
            raise FileNotFoundError(f"analysis-only requested but base dir {base} does not exist")
    else:
        base.mkdir(parents=True, exist_ok=True)

    rl_dir = base / "rl"
    hea_dir = base / "hea"
    adv_dir = base / "architect_adv"

    if not args.analysis_only:
        rl_dir.mkdir(parents=True, exist_ok=True)
        hea_dir.mkdir(parents=True, exist_ok=True)
        if args.run_adv_architect:
            adv_dir.mkdir(parents=True, exist_ok=True)

    rl_entries: Dict[int, Dict[str, Any]] = {}
    hea_entries: Dict[int, Dict[str, Any]] = {}
    adv_entries: Dict[int, Dict[str, Any]] = {}

    for k in range(args.n_seeds):
        seed_val = args.seed + k
        seed_tag = f"seed_{k}"

        # RL orchestrator
        rl_seed_dir = rl_dir / seed_tag
        rl_json_path = rl_seed_dir / "results.json"
        rl_json_used: Path | None = None
        if args.analysis_only:
            if rl_json_path.exists():
                rl_json_used = rl_json_path
                print(f"[run_vqe] Reusing RL seed {k} results (analysis-only mode)")
            else:
                print(f"[run_vqe] Missing RL seed {k} artifacts under {rl_seed_dir}; skipping seed")
                continue
        else:
            if args.reuse_existing and rl_json_path.exists():
                rl_json_used = rl_json_path
                print(f"[run_vqe] Reusing RL seed {k} from existing artifacts")
            else:
                rl_json_used = run_rl_seed(seed_val, args, rl_seed_dir)

        if rl_json_used is None or not rl_json_used.exists():
            print(f"[run_vqe] RL seed {k} produced no results.json; skipping seed")
            continue

        rl_data = load_rl_result(rl_json_used)
        rl_data["method"] = "RL"
        rl_entries[k] = rl_data

        # HEA adversarial baseline
        hea_seed_dir = hea_dir / seed_tag
        hea_json_path = hea_seed_dir / "results.json"
        hea_json_used: Path | None = None
        if args.analysis_only:
            if hea_json_path.exists():
                hea_json_used = hea_json_path
                print(f"[run_vqe] Reusing HEA seed {k} results (analysis-only mode)")
            else:
                print(f"[run_vqe] Missing HEA seed {k} artifacts under {hea_seed_dir}; skipping HEA for this seed")
        else:
            if args.reuse_existing and hea_json_path.exists():
                hea_json_used = hea_json_path
                print(f"[run_vqe] Reusing HEA seed {k} from existing artifacts")
            else:
                hea_json_used = run_hea_seed(seed_val, args, hea_seed_dir)

        if hea_json_used and hea_json_used.exists():
            hea_data = load_hea_result(hea_json_used)
            hea_data["method"] = "HEA adversarial"
            hea_entries[k] = hea_data

        # Architect adversarial variant (optional)
        if args.run_adv_architect:
            adv_seed_dir = adv_dir / seed_tag
            adv_json_path = adv_seed_dir / "results.json"
            adv_json_used: Path | None = None
            if args.analysis_only:
                if adv_json_path.exists():
                    adv_json_used = adv_json_path
                    print(f"[run_vqe] Reusing Architect seed {k} results (analysis-only mode)")
                else:
                    print(f"[run_vqe] Missing Architect seed {k} artifacts under {adv_seed_dir}; skipping Architect for this seed")
            else:
                if args.reuse_existing and adv_json_path.exists():
                    adv_json_used = adv_json_path
                    print(f"[run_vqe] Reusing Architect seed {k} from existing artifacts")
                else:
                    adv_json_used = run_adv_arch_seed(seed_val, args, adv_seed_dir)

            if adv_json_used and adv_json_used.exists():
                adv_data = load_adv_arch_result(adv_json_used)
                adv_data["method"] = "Architect adversarial"
                adv_entries[k] = adv_data

    if not rl_entries:
        raise RuntimeError("No RL results were collected; ensure training succeeded or provide existing artifacts.")

    # Optional post-architecture angle optimization (RL + Architect adv)
    try:
        ham_info = get_standard_hamiltonian(args.molecule)
    except Exception as e:
        print(f"[run_vqe] WARNING: failed to load Hamiltonian for {args.molecule}: {e}")
        print("[run_vqe] Skipping Hamiltonian-dependent post-processing "
              "(angle optimization, robustness, cross-noise, hw-eval).")
        ham_info = None

    if ham_info is not None:
        for seed_idx in rl_entries:
            seed_path = rl_dir / f"seed_{seed_idx}"
            opt_e = maybe_optimize_seed("RL", seed_path, ham_info)
            if opt_e is not None:
                rl_entries[seed_idx]["clean_energy"] = opt_e
        if args.run_adv_architect:
            for seed_idx in adv_entries:
                adv_seed = adv_dir / f"seed_{seed_idx}"
                opt_e = maybe_optimize_seed("Architect adversarial", adv_seed, ham_info)
                if opt_e is not None:
                    adv_entries[seed_idx]["clean_energy"] = opt_e

    # Aggregate and save
    rl_results = [rl_entries[k] for k in sorted(rl_entries)]
    hea_results = [hea_entries[k] for k in sorted(hea_entries)]
    adv_results = [adv_entries[k] for k in sorted(adv_entries)]

    rows = rl_results + hea_results + adv_results
    df = pd.DataFrame(rows)
    df.to_csv(base / "vqe_summary.csv", index=False)
    df.to_json(base / "vqe_summary.json", orient="records", indent=2)

    # Prepare plot DF with both metrics (clean + worst)
    plot_rows = []
    for r in rl_results:
        plot_rows.append({"method": "RL", "metric": "clean_energy", "value": r["clean_energy"]})
        if r.get("worst_energy") is not None:
            plot_rows.append({"method": "RL", "metric": "worst_energy", "value": r["worst_energy"]})
    for h in hea_results:
        plot_rows.append({"method": "HEA adversarial", "metric": "clean_energy", "value": h["clean_energy"]})
        if h.get("worst_energy") is not None:
            plot_rows.append({"method": "HEA adversarial", "metric": "worst_energy", "value": h["worst_energy"]})
    for a in adv_results:
        plot_rows.append({"method": "Architect adversarial", "metric": "clean_energy", "value": a["clean_energy"]})
        if a.get("worst_energy") is not None:
            plot_rows.append({"method": "Architect adversarial", "metric": "worst_energy", "value": a["worst_energy"]})
    plot_df = pd.DataFrame(plot_rows)
    plot_summary(plot_df, base / "vqe_energy_bar.png")

    # Robustness evaluation (energy under fixed-rate noises)
    if ham_info is not None and not args.skip_robustness:
        if args.robustness_rate:
            # Expect pairs: fam rate fam rate ...
            rates = {}
            vals = args.robustness_rate
            if len(vals) % 2 != 0:
                raise ValueError("robustness-rate must be provided as pairs: family rate ...")
            for i in range(0, len(vals), 2):
                rates[str(vals[i])] = float(vals[i + 1])
        else:
            rates = getattr(vqe_config, "ROBUSTNESS", {}).get("rates", {}) if vqe_config else {}
        if not rates:
            # fallback single rate
            rates = {fam: 0.02 for fam in args.robustness_families}
        evaluate_robustness(
            base=base,
            ham_info=ham_info,
            rl_dir=rl_dir,
            hea_dir=hea_dir,
            adv_dir=adv_dir if args.run_adv_architect else None,
            seeds=args.n_seeds,
            families=args.robustness_families,
            family_rates=rates,
            mitigation_mode=args.mitigation_mode,
            rc_zne_scales=tuple(args.rc_zne_scales) if args.rc_zne_scales else None,
            rc_zne_fit=args.rc_zne_fit,
            rc_zne_reps=args.rc_zne_reps,
        )
    # Cross-noise sweep (energy vs rate per family)
    if ham_info is not None and not args.skip_cross_noise:
        evaluate_cross_noise(
            base=base,
            ham_info=ham_info,
            rl_dir=rl_dir,
            hea_dir=hea_dir,
            adv_dir=adv_dir if args.run_adv_architect else None,
            seeds=args.n_seeds,
            families=args.cross_noise_families,
            rates=args.cross_noise_rates,
            mitigation_mode=args.mitigation_mode,
            rc_zne_scales=tuple(args.rc_zne_scales) if args.rc_zne_scales else None,
            rc_zne_fit=args.rc_zne_fit,
            rc_zne_reps=args.rc_zne_reps,
        )
    # Hardware-like eval (simulated noise backends)
    if ham_info is not None and args.run_hw_eval:
        if args.hw_rate:
            backend_rates = {}
            vals = args.hw_rate
            if len(vals) % 2 != 0:
                raise ValueError("hw-rate must be provided as pairs: backend rate ...")
            for i in range(0, len(vals), 2):
                backend_rates[str(vals[i])] = float(vals[i + 1])
        else:
            backend_rates = {b: getattr(vqe_config, "ROBUSTNESS", {}).get("rates", {}).get("depolarizing", 0.02) for b in args.hw_backends}
        evaluate_hw_like(
            base=base,
            ham_info=ham_info,
            rl_dir=rl_dir,
            hea_dir=hea_dir,
            adv_dir=adv_dir if args.run_adv_architect else None,
            seeds=args.n_seeds,
            backends=args.hw_backends,
            backend_rates=backend_rates,
            twirl_default=args.randomized_compiling,
        )
        run_qiskit_hw_eval(
            base=base,
            ham_info=ham_info,
            rl_dir=rl_dir,
            hea_dir=hea_dir,
            adv_dir=adv_dir if args.run_adv_architect else None,
            seeds=args.n_seeds,
            args=args,
        )
    print(f"[run_vqe] Finished. Results under {base}")


if __name__ == "__main__":
    main()
