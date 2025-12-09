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
from typing import List, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cirq
import numpy as np
from cirq.contrib.qasm_import import circuit_from_qasm

from utils.standard_hamiltonians import get_standard_hamiltonian
from utils.metrics import state_energy
from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
from qas_gym.utils import randomized_compile
import cirq
import random

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
    p.add_argument("--hw-backends", nargs="+", default=default_hw.get("backends", ["fake_quito", "fake_belem"]))
    p.add_argument("--hw-rate", type=float, nargs="+", default=None, help="Override backend->rate pairs: backend rate ...")
    p.set_defaults(randomized_compiling=False)
    return p.parse_args()


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
        "--saboteur-noise-family", args.adv_saboteur_noise_family,
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ]
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


def _energy_of_circuit(circuit: cirq.Circuit, ham_matrix: np.ndarray) -> float:
    sim = cirq.Simulator()
    res = sim.simulate(circuit, qubit_order=sorted(circuit.all_qubits()))
    return state_energy(res.final_state_vector, ham_matrix)


def optimize_angles(circuit: cirq.Circuit, ham_matrix: np.ndarray, maxiter: int = 80):
    """Optimize rotation angles; falls back to random search if scipy unavailable."""
    ops, rot_idx, init_angles = _extract_rotations(circuit)
    if len(rot_idx) == 0:
        return circuit, _energy_of_circuit(circuit, ham_matrix)
    try:
        import scipy.optimize as opt
    except Exception:
        best_angles = init_angles.copy()
        best_e = _energy_of_circuit(circuit, ham_matrix)
        for _ in range(maxiter):
            cand = best_angles + np.random.normal(scale=0.1, size=best_angles.shape)
            cand_circ = _build_circuit_with_angles(ops, rot_idx, cand)
            e = _energy_of_circuit(cand_circ, ham_matrix)
            if e < best_e:
                best_e = e
                best_angles = cand
        return _build_circuit_with_angles(ops, rot_idx, best_angles), best_e

    def obj(angles):
        circ = _build_circuit_with_angles(ops, rot_idx, angles)
        return _energy_of_circuit(circ, ham_matrix)

    res = opt.minimize(
        obj,
        init_angles,
        method="Powell",
        options={"maxiter": maxiter, "disp": False},
    )
    best_angles = res.x if res.success else init_angles
    best_circ = _build_circuit_with_angles(ops, rot_idx, best_angles)
    best_e = _energy_of_circuit(best_circ, ham_matrix)
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
    try:
        if method == "RL":
            circ = _load_circuit_for_method(seed_dir, "RL")
            opt_circ, opt_e = optimize_angles(circ, ham_info["matrix"])
            (seed_dir / "circuit_architect_vqe_opt.qasm").write_text(cirq.qasm(opt_circ))
            convert_qasm_file_to_cirq(seed_dir / "circuit_architect_vqe_opt.qasm", seed_dir / "circuit_architect_vqe_opt.json")
            return opt_e
        if method == "Architect adversarial":
            circ = _load_circuit_for_method(seed_dir, "Architect adversarial")
            opt_circ, opt_e = optimize_angles(circ, ham_info["matrix"])
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
        errwidth=1.2,
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


def _noisy_energy(circuit: cirq.Circuit, ham_matrix: np.ndarray, noise_family: str, rate: float, twirl: bool = False, rng=None) -> float:
    rng = rng or np.random.default_rng()
    circ_use = randomized_compile(circuit, rng) if twirl else circuit
    ops = list(circ_use.all_operations())
    noisy_ops = []
    for op in ops:
        noisy_ops.append(op)
        noisy_ops.extend(SaboteurMultiGateEnv._noise_ops_for(rate, op, noise_family, {}))
    noisy = cirq.Circuit(noisy_ops)
    sim = cirq.Simulator()
    res = sim.simulate(noisy, qubit_order=sorted(circ_use.all_qubits()))
    return state_energy(res.final_state_vector, ham_matrix)


def evaluate_robustness(
    base: Path,
    ham_info: dict,
    rl_dir: Path,
    hea_dir: Path,
    adv_dir: Path | None,
    seeds: int,
    families: list[str],
    family_rates: dict[str, float],
    twirl_default: bool = True,
):
    rows = []
    methods = [("RL", rl_dir), ("HEA adversarial", hea_dir)]
    if adv_dir is not None and adv_dir.exists():
        methods.append(("Architect adversarial", adv_dir))

    expected_nq = ham_info["n_qubits"]
    for method, root in methods:
        for k in range(seeds):
            seed_path = root / f"seed_{k}"
            try:
                circ = _load_circuit_for_method(seed_path, method)
            except FileNotFoundError:
                continue
            if _circuit_n_qubits(circ) != expected_nq:
                print(f"[robustness] Skipping {method} seed {k}: qubit mismatch ({_circuit_n_qubits(circ)} != {expected_nq})")
                continue
            for variant, twirl in (("untwirled", False), ("twirled", True if twirl_default else False)):
                clean_energy = _noisy_energy(circ, ham_info["matrix"], "depolarizing", 0.0, twirl=twirl, rng=np.random.default_rng(seed=k + (1 if twirl else 0)))
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
                    rate = family_rates.get(fam, 0.0)
                    e = _noisy_energy(circ, ham_info["matrix"], fam, rate, twirl=twirl, rng=np.random.default_rng(seed=k + hash(fam) % 9973 + (1 if twirl else 0)))
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
    out_dir = base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "vqe_robustness.csv", index=False)
    df.to_json(out_dir / "vqe_robustness.json", orient="records", indent=2)

    # Aggregate mean/std
    agg = df.groupby(["method", "variant", "noise_family"]).agg(
        energy_mean=("energy", "mean"),
        energy_std=("energy", "std"),
    ).reset_index()
    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    methods = agg["method"].unique()
    families = agg["noise_family"].unique()
    colors = {
        "untwirled": {"clean": "#4C78A8", "noisy": "#4C78A8"},
        "twirled": {"clean": "#9ecae1", "noisy": "#9ecae1"},
    }
    x_positions = np.arange(len(families))
    width = 0.2
    for m_idx, method in enumerate(methods):
        for f_idx, fam in enumerate(families):
            sub = agg[(agg["method"] == method) & (agg["noise_family"] == fam)]
            unt = sub[sub["variant"] == "untwirled"]
            twr = sub[sub["variant"] == "twirled"]
            base = unt["energy_mean"].values[0] if not unt.empty else 0
            top = twr["energy_mean"].values[0] if not twr.empty else 0
            ax.bar(
                x_positions[f_idx] + m_idx * width,
                base,
                width=width,
                label=f"{method} (untwirled)" if f_idx == 0 else None,
                color="#4C78A8" if m_idx == 0 else None,
                alpha=0.8,
            )
            ax.bar(
                x_positions[f_idx] + m_idx * width,
                top,
                bottom=base,
                width=width,
                label=f"{method} (twirled)" if f_idx == 0 else None,
                color="#9ecae1",
                alpha=0.8,
            )
    ax.set_xticks(x_positions + (len(methods) - 1) * width / 2)
    ax.set_xticklabels(families)
    ax.set_ylabel("Energy (Ha)")
    ax.set_xlabel("Noise family (fixed rate)")
    ax.set_title("VQE robustness (twirled vs untwirled)")
    ax.legend(frameon=True, title="")
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
    twirl_default: bool = True,
):
    rows = []
    methods = [("RL", rl_dir), ("HEA adversarial", hea_dir)]
    if adv_dir is not None and adv_dir.exists():
        methods.append(("Architect adversarial", adv_dir))

    expected_nq = ham_info["n_qubits"]
    for method, root in methods:
        for k in range(seeds):
            seed_path = root / f"seed_{k}"
            try:
                circ = _load_circuit_for_method(seed_path, method)
            except FileNotFoundError:
                continue
            if _circuit_n_qubits(circ) != expected_nq:
                print(f"[cross-noise] Skipping {method} seed {k}: qubit mismatch ({_circuit_n_qubits(circ)} != {expected_nq})")
                continue
            for variant, twirl in (("untwirled", False), ("twirled", True if twirl_default else False)):
                for fam in families:
                    for rate in rates:
                        e = _noisy_energy(circ, ham_info["matrix"], fam, rate, twirl=twirl, rng=np.random.default_rng(seed=k + int(rate * 1e4) + (1 if twirl else 0)))
                        rows.append(
                            {
                                "method": method,
                                "variant": variant,
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

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(families), figsize=(4 * len(families), 4), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, fam in zip(axes, families):
        sub = agg[agg["noise_family"] == fam]
        rates_unique = sorted(sub["rate"].unique())
        for method in sub["method"].unique():
            for rate in rates_unique:
                base = sub[(sub["method"] == method) & (sub["rate"] == rate) & (sub["variant"] == "untwirled")]
                twr = sub[(sub["method"] == method) & (sub["rate"] == rate) & (sub["variant"] == "twirled")]
                base_val = base["energy_mean"].values[0] if not base.empty else 0
                twr_val = twr["energy_mean"].values[0] if not twr.empty else 0
                ax.bar(
                    rate + 0.01 * hash(method) % 3 * 0,  # small jitter not needed
                    base_val,
                    width=0.005,
                    color="#4C78A8",
                    alpha=0.8,
                    label=f"{method} untwirled" if rate == rates_unique[0] else None,
                )
                ax.bar(
                    rate + 0.01 * hash(method) % 3 * 0,
                    twr_val,
                    bottom=base_val,
                    width=0.005,
                    color="#9ecae1",
                    alpha=0.8,
                    label=f"{method} twirled" if rate == rates_unique[0] else None,
                )
        ax.set_title(fam)
        ax.set_xlabel("Noise rate")
        ax.set_ylabel("Energy (Ha)")
    axes[0].legend(frameon=True)
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
    for method, root in methods:
        for k in range(seeds):
            seed_path = root / f"seed_{k}"
            try:
                circ = _load_circuit_for_method(seed_path, method)
            except FileNotFoundError:
                continue
            if _circuit_n_qubits(circ) != expected_nq:
                print(f"[hw-eval] Skipping {method} seed {k}: qubit mismatch ({_circuit_n_qubits(circ)} != {expected_nq})")
                continue
            for variant, twirl in (("untwirled", False), ("twirled", True if twirl_default else False)):
                for backend in backends:
                    rate = backend_rates.get(backend, 0.02)
                    e = _noisy_energy(circ, ham_info["matrix"], "depolarizing", rate, twirl=twirl, rng=np.random.default_rng(seed=k + hash(backend) % 7919 + (1 if twirl else 0)))
                    rows.append(
                        {"method": method, "variant": variant, "backend": backend, "energy": e, "seed": k}
                    )

    df = pd.DataFrame(rows)
    out_dir = base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "vqe_hardware_eval.csv", index=False)
    df.to_json(out_dir / "vqe_hardware_eval.json", orient="records", indent=2)

    agg = df.groupby(["method", "variant", "backend"]).agg(
        energy_mean=("energy", "mean"),
        energy_std=("energy", "std"),
    ).reset_index()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    backends_unique = agg["backend"].unique()
    methods_unique = agg["method"].unique()
    width = 0.25
    x = np.arange(len(backends_unique))
    for m_idx, method in enumerate(methods_unique):
        for b_idx, backend in enumerate(backends_unique):
            sub = agg[(agg["method"] == method) & (agg["backend"] == backend)]
            base = sub[sub["variant"] == "untwirled"]
            twr = sub[sub["variant"] == "twirled"]
            base_val = base["energy_mean"].values[0] if not base.empty else 0
            twr_val = twr["energy_mean"].values[0] if not twr.empty else 0
            pos = x[b_idx] + m_idx * width
            ax.bar(pos, base_val, width=width, color="#4C78A8", alpha=0.8, label=f"{method} untwirled" if b_idx == 0 else None)
            ax.bar(pos, twr_val, bottom=base_val, width=width, color="#9ecae1", alpha=0.8, label=f"{method} twirled" if b_idx == 0 else None)
    ax.set_xticks(x + (len(methods_unique) - 1) * width / 2)
    ax.set_xticklabels(backends_unique)
    ax.set_ylabel("Energy (Ha)")
    ax.set_xlabel("Backend (simulated noise)")
    ax.set_title("VQE hardware-like eval (twirled vs untwirled)")
    ax.legend(frameon=True, title="")
    plt.tight_layout()
    plt.savefig(out_dir / "vqe_hardware_eval.png", dpi=300)
    plt.close(fig)
    return agg


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(args.base_dir).expanduser() if args.base_dir else Path(f"results/vqe_run_{timestamp}")
    base.mkdir(parents=True, exist_ok=True)

    rl_dir = base / "rl"
    hea_dir = base / "hea"
    adv_dir = base / "architect_adv"

    rl_results = []
    hea_results = []
    adv_results = []
    for k in range(args.n_seeds):
        seed_val = args.seed + k
        rl_json = run_rl_seed(seed_val, args, rl_dir / f"seed_{k}")
        hea_json = run_hea_seed(seed_val, args, hea_dir / f"seed_{k}")
        rl_data = load_rl_result(rl_json)
        rl_data.update({"method": "RL"})
        hea_data = load_hea_result(hea_json)
        hea_data.update({"method": "HEA adversarial"})
        rl_results.append(rl_data)
        hea_results.append(hea_data)
        if args.run_adv_architect:
            adv_json = run_adv_arch_seed(seed_val, args, adv_dir / f"seed_{k}")
            adv_data = load_adv_arch_result(adv_json)
            adv_data.update({"method": "Architect adversarial"})
            adv_results.append(adv_data)

    # Optional post-architecture angle optimization (RL + Architect adv)
    ham_info = get_standard_hamiltonian(args.molecule)
    for k in range(args.n_seeds):
        seed_path = rl_dir / f"seed_{k}"
        opt_e = maybe_optimize_seed("RL", seed_path, ham_info)
        if opt_e is not None and k < len(rl_results):
            rl_results[k]["clean_energy"] = opt_e
        if args.run_adv_architect:
            adv_seed = adv_dir / f"seed_{k}"
            opt_e = maybe_optimize_seed("Architect adversarial", adv_seed, ham_info)
            if opt_e is not None and k < len(adv_results):
                adv_results[k]["clean_energy"] = opt_e

    # Aggregate and save
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
    if not args.skip_robustness:
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
        ham_info = get_standard_hamiltonian(args.molecule)
        evaluate_robustness(
            base=base,
            ham_info=ham_info,
            rl_dir=rl_dir,
            hea_dir=hea_dir,
            adv_dir=adv_dir if args.run_adv_architect else None,
            seeds=args.n_seeds,
            families=args.robustness_families,
            family_rates=rates,
            twirl_default=args.randomized_compiling,
        )
    # Cross-noise sweep (energy vs rate per family)
    if not args.skip_cross_noise:
        evaluate_cross_noise(
            base=base,
            ham_info=ham_info,
            rl_dir=rl_dir,
            hea_dir=hea_dir,
            adv_dir=adv_dir if args.run_adv_architect else None,
            seeds=args.n_seeds,
            families=args.cross_noise_families,
            rates=args.cross_noise_rates,
            twirl=args.randomized_compiling,
        )
    # Hardware-like eval (simulated noise backends)
    if args.run_hw_eval:
        ham_info = get_standard_hamiltonian(args.molecule)
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
            twirl=args.randomized_compiling,
        )
    print(f"[run_vqe] Finished. Results under {base}")


if __name__ == "__main__":
    main()
