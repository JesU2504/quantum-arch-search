#!/usr/bin/env python3
"""
Lightweight hardware-eval harness using IBM-style backends (Fake Quito/Belem/Athens/Yorktown).

Runs shot-based simulations with realistic noise (via Fake backends) on baseline,
robust, and QuantumNAS circuits. Computes simple success probabilities on
target bitstrings (GHZ defaults to all-zeros or all-ones), records transpiled
depth/gate counts, and saves JSON + plots.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np

from qas_gym.utils import load_circuit
from utils.qiskit_conversion import cirq_to_qiskit, ensure_measurements

# Qiskit imports (Aer + Fake backends)
try:
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover - fallback for older installs
    from qiskit.providers.aer import AerSimulator  # type: ignore
from qiskit import transpile
import matplotlib.pyplot as plt

# Resolve Fake backends across qiskit versions (include newer devices like Lagos/Oslo)
FakeBackendMap: Dict[str, object] = {}
fake_class_candidates = [
    # Legacy provider namespace
    ("qiskit.providers.fake_provider", [
        "FakeQuito", "FakeBelem", "FakeAthens", "FakeYorktown",
        "FakeLagos", "FakeOslo",
    ]),
    # Runtime namespace (newer qiskit-ibm-runtime)
    ("qiskit_ibm_runtime.fake_provider", [
        "FakeQuito", "FakeQuitoV2", "FakeBelem", "FakeBelemV2",
        "FakeAthens", "FakeAthensV2", "FakeYorktown",
        "FakeLagos", "FakeLagosV2", "FakeOslo", "FakeOsloV2",
    ]),
    # Deprecated ibm_provider namespace
    ("qiskit_ibm_provider.fake_provider", [
        "FakeQuito", "FakeQuitoV2", "FakeBelem", "FakeBelemV2",
        "FakeAthens", "FakeAthensV2", "FakeYorktown",
        "FakeLagos", "FakeLagosV2", "FakeOslo", "FakeOsloV2",
    ]),
]
for mod_name, cls_candidates in fake_class_candidates:
    try:
        mod = __import__(mod_name, fromlist=cls_candidates)
        for cls_name in cls_candidates:
            if hasattr(mod, cls_name):
                FakeBackendMap[cls_name.lower()] = getattr(mod, cls_name)
    except Exception:
        continue


FAKE_BACKENDS = {}
for key, cls in FakeBackendMap.items():
    if cls is None:
        continue
    norm = key
    # normalize variants to snake case
    norm = norm.replace("fake", "fake_")
    norm = norm.replace("v2", "").replace("V2", "")
    FAKE_BACKENDS[norm] = cls


def resolve_backends(names: List[str]):
    backends = {}
    for name in names:
        key = name.lower()
        if key not in FAKE_BACKENDS:
            raise ValueError(f"Unknown backend '{name}'. Supported: {list(FAKE_BACKENDS)}")
        backends[key] = FAKE_BACKENDS[key]()
    return backends


def load_qiskit_circuit_from_json(path: str):
    cirq_circuit = load_circuit(path)
    return cirq_to_qiskit(cirq_circuit)


def gate_counts(qc) -> Dict[str, int]:
    counts = qc.count_ops()
    return {k: int(v) for k, v in counts.items()}


def compute_success_prob(counts: Dict[str, int], success_bitstrings: List[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(sum(counts.get(bs, 0) for bs in success_bitstrings) / total)


def evaluate_on_backend(
    circuits: Dict[str, list[str]],
    backend_name: str,
    backend_obj,
    shots: int,
    target_bitstrings: List[str],
    opt_level: int,
    seed: int,
    initial_layout: List[int] | None = None,
) -> Dict[str, Dict]:
    results = []
    sim = AerSimulator.from_backend(backend_obj)
    for label, paths in circuits.items():
        for path in paths:
            qc = load_qiskit_circuit_from_json(path)
            qc_meas = ensure_measurements(qc)
            tqc = transpile(
                qc_meas,
                backend=backend_obj,
                optimization_level=opt_level,
                seed_transpiler=seed,
                initial_layout=initial_layout,
            )
            job = sim.run(tqc, shots=shots, seed_simulator=seed)
            res = job.result()
            counts = res.get_counts()
            success_prob = compute_success_prob(counts, target_bitstrings)

            results.append({
                "backend": backend_name,
                "circuit": label,
                "circuit_path": path,
                "shots": shots,
                "success_prob": success_prob,
                "depth": tqc.depth(),
                "width": tqc.num_qubits,
                "gate_counts": gate_counts(tqc),
                "counts": counts,
            })
    return results


def plot_backend_results(all_results: Dict[str, Dict[str, Dict]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Skip plotting here; consolidated plotting handled by plot_hw_fidelity.py
    return


def run_hw_eval(
    baseline_circuit: str = None,
    robust_circuit: str = None,
    quantumnas_circuit: str = None,
    baseline_circuits: list[str] | None = None,
    robust_circuits: list[str] | None = None,
    quantumnas_circuits: list[str] | None = None,
    backends: List[str] = None,
    shots: int = 4096,
    opt_level: int = 3,
    seed: int = 1234,
    target_bitstrings: List[str] = None,
    output_dir: str = None,
    initial_layout: List[int] | None = None,
):
    circuits: Dict[str, list[str]] = {"baseline": [], "robust": [], "quantumnas": []}
    for path in baseline_circuits or ([] if baseline_circuit is None else [baseline_circuit]):
        if path and os.path.exists(path):
            circuits["baseline"].append(path)
    for path in robust_circuits or ([] if robust_circuit is None else [robust_circuit]):
        if path and os.path.exists(path):
            circuits["robust"].append(path)
    for path in quantumnas_circuits or ([] if quantumnas_circuit is None else [quantumnas_circuit]):
        if path and os.path.exists(path):
            circuits["quantumnas"].append(path)
    circuits = {k: v for k, v in circuits.items() if v}
    if not circuits:
        raise ValueError("No valid circuit paths provided.")

    backends = backends or ["fake_quito", "fake_belem"]
    target_bitstrings = target_bitstrings or []
    output_dir = output_dir or f"results/hardware_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    backend_objs = resolve_backends(backends)
    all_results = {}
    for name, backend_obj in backend_objs.items():
        backend_results = evaluate_on_backend(
            circuits,
            backend_name=name,
            backend_obj=backend_obj,
            shots=shots,
            target_bitstrings=target_bitstrings,
            opt_level=opt_level,
            seed=seed,
            initial_layout=initial_layout,
        )
        all_results[name] = backend_results

    # Save JSON
    json_path = os.path.join(output_dir, "hardware_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Plots
    plot_backend_results(all_results, output_dir)

    return all_results


def parse_success_bitstrings(arg: str, n_qubits: int) -> List[str]:
    if arg:
        return [s.strip() for s in arg.split(",") if s.strip()]
    # Default GHZ success: all-zeros and all-ones
    return ["0" * n_qubits, "1" * n_qubits]


def parse_initial_layout(arg: str | None) -> List[int] | None:
    if not arg:
        return None
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [int(p) for p in parts]


def cli():
    parser = argparse.ArgumentParser(description="Evaluate circuits on IBM-style fake backends (Aer).")
    parser.add_argument("--baseline-circuit", type=str, help="Path to baseline circuit JSON")
    parser.add_argument("--robust-circuit", type=str, help="Path to robust circuit JSON")
    parser.add_argument("--quantumnas-circuit", type=str, help="Path to QuantumNAS circuit JSON")
    parser.add_argument("--compare-dir", type=str, default=None,
                        help="Optional compare directory; if provided, evaluate all run_*/ circuits for baseline/robust/quantumnas.")
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["fake_quito", "fake_belem", "fake_athens", "fake_lagos", "fake_oslo"],
        help="Backends to use (e.g., fake_quito fake_belem fake_athens fake_lagos fake_oslo)",
    )
    parser.add_argument("--shots", type=int, default=4096, help="Number of shots")
    parser.add_argument("--opt-level", type=int, default=3, help="Transpiler optimization level (0-3)")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for transpiler and simulator")
    parser.add_argument("--success-bitstrings", type=str, default=None,
                        help="Comma-separated list of success bitstrings (default: GHZ-style 0..0 and 1..1)")
    parser.add_argument("--n-qubits", type=int, required=True, help="Number of qubits (for default GHZ success list)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results/plots")
    parser.add_argument("--initial-layout", type=str, default=None,
                        help="Comma-separated physical qubit indices for transpile initial_layout (e.g., 0,1,2)")
    args = parser.parse_args()

    if not FAKE_BACKENDS:
        raise SystemExit("No fake backends available in this qiskit install; install a version with FakeQuito/FakeBelem or adjust --backends.")

    success_bitstrings = parse_success_bitstrings(args.success_bitstrings, args.n_qubits)
    init_layout = parse_initial_layout(args.initial_layout)
    baseline_list = None
    robust_list = None
    qnas_list = None
    if args.compare_dir:
        base = Path(args.compare_dir)
        run_dirs = sorted([p for p in base.glob("run_*") if p.is_dir()])
        baseline_list = []
        robust_list = []
        qnas_list = []
        for rd in run_dirs:
            b = rd / "circuit_vanilla.json"
            r = rd / "circuit_robust.json"
            q = rd / "circuit_quantumnas.json"
            if b.exists():
                baseline_list.append(str(b))
            if r.exists():
                robust_list.append(str(r))
            if q.exists():
                qnas_list.append(str(q))

    run_hw_eval(
        baseline_circuit=args.baseline_circuit,
        robust_circuit=args.robust_circuit,
        quantumnas_circuit=args.quantumnas_circuit,
        baseline_circuits=baseline_list,
        robust_circuits=robust_list,
        quantumnas_circuits=qnas_list,
        backends=args.backends,
        shots=args.shots,
        opt_level=args.opt_level,
        seed=args.seed,
        target_bitstrings=success_bitstrings,
        output_dir=args.output_dir,
        initial_layout=init_layout,
    )


if __name__ == "__main__":
    cli()
