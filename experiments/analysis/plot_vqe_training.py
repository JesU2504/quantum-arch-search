#!/usr/bin/env python3
"""
Plot VQE training traces and summarize best energies across methods.

Usage:
    python -m experiments.analysis.plot_vqe_training --molecule H2 \
        --out-dir results/vqe_plots_h2 \
        --adv-glob "results/adversarial_vqe_h2_seed*/results.json" \
        --qnas-glob "results/qnas_vqe_h2_seed*/results.json" \
        --rl-glob "results/vqe_architect_h2_seed*/vqe_h2_greedy_results.json"
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

try:
    from utils.standard_hamiltonians import get_standard_hamiltonian  # type: ignore
except Exception:
    get_standard_hamiltonian = None


def _mixed_state_energy(pauli_terms, n_qubits: int) -> float:
    """
    Energy of maximally mixed state for a Pauli-sum Hamiltonian.
    Only identity strings contribute.
    """
    dim = 2**n_qubits
    trace = 0.0
    for coeff, pauli in pauli_terms:
        if all(p == "I" for p in pauli):
            trace += coeff * dim
    return trace / dim


def _energy_from_circuit_or_qasm(circuit_or_qasm, pauli_terms, noise_p: float | None, n_qubits: int):
    """
    Simulate circuit (cirq.Circuit or QASM string) with optional gate-dependent noise
    and compute energy. Noise model: per-qubit amplitude damping + small Z
    over-rotation after each gate.
    """
    if isinstance(circuit_or_qasm, str):
        circuit = circuit_from_qasm(circuit_or_qasm)
    else:
        circuit = circuit_or_qasm
    # Remap to LineQubit order 0..n_qubits-1 for consistent simulation
    all_qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    qmap = {}
    for i, q in enumerate(sorted(circuit.all_qubits(), key=lambda q: str(q))):
        if i < n_qubits:
            qmap[q] = all_qubits[i]
    circuit = circuit.transform_qubits(lambda q: qmap.get(q, q))
    if noise_p and noise_p > 0:
        noisy_ops = []
        for op in circuit.all_operations():
            noisy_ops.append(op)
            for q in op.qubits:
                try:
                    noisy_ops.append(cirq.amplitude_damp(noise_p).on(q))
                except Exception:
                    try:
                        noisy_ops.append(cirq.AmplitudeDampingChannel(noise_p).on(q))
                    except Exception:
                        noisy_ops.append(cirq.DepolarizingChannel(noise_p).on(q))
                noisy_ops.append(cirq.rz(noise_p * np.pi / 4).on(q))
        circuit = cirq.Circuit(noisy_ops)
    sim = cirq.DensityMatrixSimulator()
    result = sim.simulate(circuit, qubit_order=all_qubits)
    rho = result.final_density_matrix
    mats = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.array([[1, 0], [0, -1]])}
    energy = 0.0
    for coeff, label in pauli_terms:
        op = mats[label[0]]
        for p in label[1:]:
            op = np.kron(op, mats[p])
        energy += coeff * np.real(np.trace(op @ rho))
    return energy


def _load_adv(glob_pat: str) -> List[Dict]:
    runs = []
    for path in sorted(glob.glob(glob_pat)):
        d = json.load(open(path))
        seed = d.get("seed")
        if seed is None:
            # Try to extract seed from directory name (seedX)
            parts = Path(path).parts
            for p in parts:
                if p.startswith("seed"):
                    try:
                        seed = int(p.replace("seed", ""))
                        break
                    except ValueError:
                        continue
        runs.append(
            {
                "path": path,
                "seed": seed,
                "best": d.get("best_clean_energy"),
                "trace": d.get("history", []),
                "eval": d.get("eval_sweep", []),
            }
        )
    return runs


def _load_qnas(glob_pat: str) -> List[Dict]:
    runs = []
    for path in sorted(glob.glob(glob_pat)):
        d = json.load(open(path))
        runs.append(
            {
                "path": path,
                "seed": d.get("seed"),
                "best": d.get("best_energy"),
                "trace": d.get("energy_trace", []),
            }
        )
    return runs


def _load_rl(glob_pat: str) -> List[Dict]:
    runs = []
    for path in sorted(glob.glob(glob_pat)):
        d = json.load(open(path))
        energies = [r["optimized_energy"] for r in d.get("results", []) if r.get("optimized_energy") is not None]
        # attempt to load best circuit if available
        best_circuit = None
        base = Path(path).parent
        candidate = base / "episode_logs" / "best_circuit.json"
        if candidate.exists():
            try:
                best_circuit = cirq.read_json(json_text=candidate.read_text())
            except Exception:
                best_circuit = None
        runs.append({"path": path, "best": min(energies) if energies else None, "trace": energies, "circuit": best_circuit})
    return runs


def plot_traces(molecule: str, out_dir: Path, adv, qnas, rl, fci=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    # Pick first seed (or best available) for clarity
    if adv:
        xs = [p["step"] for p in adv[0]["trace"]]
        ys = [p["clean_energy"] for p in adv[0]["trace"]]
        plt.plot(xs, ys, color="C0", marker="o", markersize=3, label=f"Adversarial seed{adv[0].get('seed')}")
    if qnas:
        xs = [p["step"] for p in qnas[0]["trace"]]
        ys = [p["energy"] for p in qnas[0]["trace"]]
        plt.plot(xs, ys, color="C1", linestyle="--", marker="s", markersize=3, label=f"QuantumNAS seed{qnas[0].get('seed')}")
    if rl:
        xs = list(range(1, len(rl[0]["trace"]) + 1))
        ys = rl[0]["trace"]
        plt.plot(xs, ys, color="C2", linestyle=":", marker="^", markersize=3, label=f"RL seed0")

    if fci is not None:
        plt.axhline(fci, color="k", linestyle=":", linewidth=1, label="FCI target")
    plt.xlabel("Step / Episode")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE training traces (single seed per method)")
    plt.legend(fontsize=8, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{molecule.lower()}_training_traces.png", dpi=200)


def plot_noisy_traces(molecule: str, out_dir: Path, adv, qnas, rl, fci=None, noise_levels=None):
    """
    Plot clean vs noisy (approximate depolarizing) training traces for a few noise levels.
    Uses the same single-seed traces as plot_traces; noise curves are dashed and
    model cumulative depolarizing with depth: alpha_k = 1-(1-p)^k. When per-step
    qasm is available, simulate true noisy energy.
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.02, 0.05]
    noise_levels = [float(p) for p in noise_levels]

    mixed_energy = None
    pauli_terms = None
    n_qubits = None
    if get_standard_hamiltonian is not None:
        try:
            ham_info = get_standard_hamiltonian(molecule)
            mixed_energy = _mixed_state_energy(ham_info["pauli_terms"], ham_info["n_qubits"])
            pauli_terms = ham_info["pauli_terms"]
            n_qubits = ham_info["n_qubits"]
        except Exception:
            mixed_energy = None

    # Prepare clean traces once
    clean_traces = []
    if adv:
        xs = np.array([p["step"] for p in adv[0]["trace"]], dtype=float)
        ys = np.array([p["clean_energy"] for p in adv[0]["trace"]], dtype=float)
        clean_traces.append(("Adversarial", "C0", "o", xs, ys, adv[0].get("seed")))
    if qnas:
        xs = np.array([p["step"] for p in qnas[0]["trace"]], dtype=float)
        ys = np.array([p["energy"] for p in qnas[0]["trace"]], dtype=float)
        clean_traces.append(("QuantumNAS", "C1", "s", xs, ys, qnas[0].get("seed")))
    if rl:
        xs = np.arange(1, len(rl[0]["trace"]) + 1, dtype=float)
        ys = np.array(rl[0]["trace"], dtype=float)
        clean_traces.append(("RL", "C2", "^", xs, ys, 0))

    for p in noise_levels:
        plt.figure(figsize=(8, 5))
        for name, color, marker, xs, ys, seed in clean_traces:
            plt.plot(xs, ys, color=color, marker=marker, markersize=3, label=f"{name} seed{seed} (clean)")
            if mixed_energy is not None:
                # Cumulative depolarizing: alpha grows with depth
                if name == "Adversarial" and adv and pauli_terms is not None:
                    qasms = [pt["qasm"] for pt in adv[0]["trace"] if pt.get("qasm")]
                    noisy = np.array([_energy_from_circuit_or_qasm(qasm, pauli_terms, p, n_qubits) for qasm in qasms])
                elif name == "QuantumNAS" and qnas and pauli_terms is not None:
                    qasms = [pt["qasm"] for pt in qnas[0]["trace"] if pt.get("qasm")]
                    noisy = np.array([_energy_from_circuit_or_qasm(qasm, pauli_terms, p, n_qubits) for qasm in qasms])
                elif name == "RL" and rl and pauli_terms is not None and rl[0].get("circuit") is not None:
                    noisy_val = _energy_from_circuit_or_qasm(rl[0]["circuit"], pauli_terms, p, n_qubits)
                    noisy = np.full_like(xs, noisy_val)
                else:
                    alpha = 1.0 - np.power((1.0 - p), xs)
                    noisy = (1 - alpha) * ys + alpha * mixed_energy
                plt.plot(xs, noisy, color=color, linestyle="--", marker=marker, markersize=3, alpha=0.5,
                         label=f"{name} seed{seed} (p={p:.2f})")
        if fci is not None:
            plt.axhline(fci, color="k", linestyle=":", linewidth=1, label="FCI target")
        plt.xlabel("Step / Episode")
        plt.ylabel("Energy (Ha)")
        plt.title(f"{molecule} VQE traces clean vs noisy (p={p:.2f})")
        plt.legend(fontsize=8, ncol=1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        suffix = f"{int(p*1000):03d}m"  # e.g., 050m for 5%
        plt.savefig(out_dir / f"{molecule.lower()}_training_traces_noise_{suffix}.png", dpi=200)
        plt.close()


def plot_eval_sweep(molecule: str, out_dir: Path, adv, qnas, rl, fci=None, noise_grid=None):
    """
    Plot energy vs depolarizing noise for all methods.

    For methods without stored sweeps, we approximate:
        E(p) = (1 - p) * best_clean + p * E(mixed_state)
    using the standard Hamiltonian to derive the mixed-state energy.
    """
    if noise_grid is None:
        target_x = np.linspace(0.0, 0.05, 6)
    else:
        target_x = np.array(noise_grid, dtype=float)

    mixed_energy = None
    pauli_terms = None
    n_qubits = None
    if get_standard_hamiltonian is not None:
        try:
            ham_info = get_standard_hamiltonian(molecule)
            mixed_energy = _mixed_state_energy(ham_info["pauli_terms"], ham_info["n_qubits"])
            pauli_terms = ham_info["pauli_terms"]
            n_qubits = ham_info["n_qubits"]
        except Exception:
            mixed_energy = None

    plt.figure(figsize=(6, 4))
    curves = []
    # Adversarial: use recorded sweep if present, else simulate last circuit or approximate
    for run in adv:
        if run.get("eval"):
            xs, ys = zip(*run["eval"])
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            # interpolate to target grid if needed
            if xs.shape != target_x.shape or np.any(xs != target_x):
                ys = np.interp(target_x, xs, ys)
            xs = target_x
        elif mixed_energy is not None and pauli_terms is not None and run.get("trace"):
            last_qasm = next((pt["qasm"] for pt in reversed(run["trace"]) if pt.get("qasm")), None)
            if last_qasm:
                ys = np.array([_energy_from_circuit_or_qasm(last_qasm, pauli_terms, p, n_qubits) for p in target_x])
            else:
                ys = (1 - target_x) * run.get("best", 0) + target_x * mixed_energy
            xs = target_x
        else:
            continue
        curves.append(("Adv", xs, ys))

    # QuantumNAS approximate sweep
    for run in qnas:
        if mixed_energy is None:
            continue
        if pauli_terms is not None and run.get("trace"):
            last_qasm = next((pt["qasm"] for pt in reversed(run["trace"]) if pt.get("qasm")), None)
            if last_qasm:
                ys = np.array([_energy_from_circuit_or_qasm(last_qasm, pauli_terms, p, n_qubits) for p in target_x])
                xs = target_x
            else:
                xs = target_x
                ys = (1 - xs) * run.get("best", 0) + xs * mixed_energy
        else:
            if run.get("best") is None:
                continue
            xs = target_x
            ys = (1 - xs) * run["best"] + xs * mixed_energy
        curves.append(("QNAS", xs, ys))

    # RL: simulate best circuit if available, else approximate
    for idx, run in enumerate(rl):
        if mixed_energy is None:
            continue
        if pauli_terms is not None and run.get("circuit") is not None:
            ys = np.array([_energy_from_circuit_or_qasm(run["circuit"], pauli_terms, p, n_qubits) for p in target_x])
            xs = target_x
        elif run.get("best") is not None:
            xs = target_x
            ys = (1 - xs) * run["best"] + xs * mixed_energy
        else:
            continue
        curves.append(("RL", xs, ys))

    all_y = []

    def _aggregate_noise(curves, label, color, linestyle, marker):
        nonlocal all_y
        ys_list = [np.array(y, dtype=float) for name, x, y in curves if name == label]
        if not ys_list:
            return
        xs = curves[0][1]
        ys_stack = np.stack(ys_list, axis=0)
        mean = ys_stack.mean(axis=0)
        std = ys_stack.std(axis=0, ddof=0)
        plt.plot(xs, mean, color=color, linestyle=linestyle, marker=marker, label=f"{label} (mean, n={len(ys_list)})")
        plt.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)
        all_y.extend(mean.tolist())
        all_y.extend((mean - std).tolist())
        all_y.extend((mean + std).tolist())

    _aggregate_noise(curves, "Adv", "C0", "-", "o")
    _aggregate_noise(curves, "QNAS", "C1", "--", "s")
    _aggregate_noise(curves, "RL", "C2", ":", "^")

    if fci is not None:
        plt.axhline(fci, color="k", linestyle=":", linewidth=1, label="FCI target")
        all_y.append(fci)
    plt.xlabel("Depolarizing rate")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} energy vs depolarizing noise (mean ± std)")
    if all_y:
        ymin, ymax = min(all_y), max(all_y)
        pad = max(1e-3, 0.05 * (ymax - ymin if ymax > ymin else 1e-3))
        plt.ylim(ymin - pad, ymax + pad)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{molecule.lower()}_noise_sweep.png", dpi=200)


def write_summary(out_dir: Path, adv, qnas, rl):
    rows = []
    for name, runs in [("Adversarial", adv), ("QuantumNAS", qnas), ("RL", rl)]:
        bests = [r["best"] for r in runs if r["best"] is not None]
        if not bests:
            continue
        rows.append(
            {
                "method": name,
                "mean_best": float(np.mean(bests)),
                "std_best": float(np.std(bests, ddof=0)),
                "n_seeds": len(bests),
            }
        )
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "mean_best", "std_best", "n_seeds"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Plot VQE training and robustness.")
    ap.add_argument("--molecule", default="H2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adv-glob", required=True)
    ap.add_argument("--qnas-glob", required=True)
    ap.add_argument("--rl-glob", required=True)
    ap.add_argument("--fci", type=float, default=None)
    ap.add_argument(
        "--trace-noise-levels",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05],
        help="Depolarizing rates to overlay as dashed noisy traces.",
    )
    ap.add_argument(
        "--sweep-noise-levels",
        type=float,
        nargs="+",
        default=None,
        help="Noise levels for the summary sweep plot (defaults to 0..0.05).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    adv = _load_adv(args.adv_glob)
    qnas = _load_qnas(args.qnas_glob)
    rl = _load_rl(args.rl_glob)

    plot_traces(args.molecule, out_dir, adv, qnas, rl, args.fci)
    plot_noisy_traces(args.molecule, out_dir, adv, qnas, rl, args.fci, args.trace_noise_levels)
    plot_eval_sweep(args.molecule, out_dir, adv, qnas, rl, args.fci, args.sweep_noise_levels)
    write_summary(out_dir, adv, qnas, rl)
    print(f"Saved plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
