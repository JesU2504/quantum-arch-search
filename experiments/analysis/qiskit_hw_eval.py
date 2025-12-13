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
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import cirq

from qas_gym.utils import load_circuit, randomized_compile
from utils.qiskit_conversion import cirq_to_qiskit, ensure_measurements

# Qiskit imports (Aer + Fake backends)
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
except Exception:  # pragma: no cover - fallback for older installs
    from qiskit.providers.aer import AerSimulator  # type: ignore
    from qiskit.providers.aer.noise import NoiseModel  # type: ignore
from qiskit import transpile, QuantumRegister
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


DENSITY_SAVE_LABEL = "_density_matrix"


def load_qiskit_circuit_from_json(path: str, randomized_compile_flag: bool = False, rng=None):
    """Load and optionally twirl a circuit for qiskit execution.
    
    Supports gate-insertion twirl (randomized_compile) for hardware-style simulation.
    Note: Frame-based Pauli twirl (from robustness_sweep.py) is deterministic and
    evaluated in the simulator; it does not affect hardware transpilation.
    
    Args:
        path: Path to circuit JSON.
        randomized_compile_flag: If True, apply gate-insertion Pauli twirl.
        rng: Random number generator seed.
    
    Returns:
        Qiskit QuantumCircuit ready for transpilation and execution.
    """
    cirq_circuit = load_circuit(path)
    if randomized_compile_flag:
        rng = rng or np.random.default_rng()
        cirq_circuit = randomized_compile(cirq_circuit, rng)
    # Strip twirl tags before QASM export to keep conversion happy
    ops = []
    for op in cirq_circuit.all_operations():
        if isinstance(op, cirq.TaggedOperation):
            ops.append(op.untagged)
        else:
            ops.append(op)
    return cirq_to_qiskit(cirq.Circuit(ops))


def gate_counts(qc) -> Dict[str, int]:
    counts = qc.count_ops()
    return {k: int(v) for k, v in counts.items()}


def compute_success_prob(counts: Dict[str, int], success_bitstrings: List[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(sum(counts.get(bs, 0) for bs in success_bitstrings) / total)


def _energy_from_density_matrix(dm, ham_matrix: np.ndarray) -> float:
    if hasattr(dm, "data"):
        rho = np.asarray(dm.data, dtype=complex)
    else:
        rho = np.asarray(dm, dtype=complex)
    if ham_matrix.shape != rho.shape:
        raise ValueError(f"Hamiltonian matrix shape {ham_matrix.shape} does not match density matrix shape {rho.shape}")
    return float(np.real(np.trace(ham_matrix @ rho)))


def _build_readout_filter(
    sim,
    backend_obj,
    width: int,
    *,
    shots: int,
    opt_level: int,
    seed: int,
    initial_layout: List[int] | None,
):
    """
    Build a simple confusion-matrix-based readout mitigator for the given backend by calibration.

    Returns a function counts -> mitigated_counts and the calibration counts.
    """
    from qiskit import QuantumCircuit, transpile

    basis_states = [format(i, f"0{width}b") for i in range(2**width)]
    cal_circs = []
    for state in basis_states:
        qc = QuantumCircuit(width, width)
        for idx, bit in enumerate(reversed(state)):  # little-endian prep to match measurement keys
            if bit == "1":
                qc.x(idx)
        qc.measure(range(width), range(width))
        cal_circs.append(qc)

    cal_circs = transpile(
        cal_circs,
        backend=backend_obj,
        optimization_level=opt_level,
        seed_transpiler=seed,
        initial_layout=initial_layout,
    )
    job = sim.run(cal_circs, shots=shots, seed_simulator=seed)
    cal_res = job.result()
    M = np.zeros((2**width, 2**width), dtype=float)
    for prep_idx, state in enumerate(basis_states):
        counts = cal_res.get_counts(prep_idx)
        total = sum(counts.values()) or 1
        for meas, c in counts.items():
            try:
                meas_idx = basis_states.index(meas)
            except ValueError:
                continue
            M[prep_idx, meas_idx] = c / total
    try:
        Minv = np.linalg.pinv(M)
    except Exception:
        return None, None

    def mitigate_counts(raw_counts: Dict[str, int]) -> Dict[str, int]:
        total = sum(raw_counts.values())
        if total <= 0:
            return raw_counts
        p_raw = np.array([raw_counts.get(bs, 0) / total for bs in basis_states], dtype=float)
        p_est = Minv @ p_raw
        p_est = np.clip(p_est, 0.0, 1.0)
        if p_est.sum() > 0:
            p_est = p_est / p_est.sum()
        mitigated_counts = {bs: float(p * total) for bs, p in zip(basis_states, p_est)}
        return mitigated_counts

    return mitigate_counts, cal_res.to_dict()


def _readout_matrix_from_noise_model(noise_model: NoiseModel, width: int) -> np.ndarray | None:
    """Return composite readout confusion matrix from a NoiseModel if available."""
    try:
        # NoiseModel keeps local readout errors keyed by qubit tuples
        mats = []
        for q in range(width):
            err = noise_model._local_readout_errors.get((q,))  # type: ignore[attr-defined]
            if err is None:
                return None
            mats.append(np.asarray(err.probabilities(), dtype=float))
        M = mats[0]
        for mat in mats[1:]:
            M = np.kron(M, mat)
        return M
    except Exception:
        return None


def _mitigator_from_noise_matrix(M: np.ndarray | None):
    if M is None:
        return None
    try:
        Minv = np.linalg.pinv(M)
    except Exception:
        return None

    def mitigate_counts(raw_counts: Dict[str, int], basis_states: List[str]) -> Dict[str, int]:
        total = sum(raw_counts.values())
        if total <= 0:
            return raw_counts
        p_raw = np.array([raw_counts.get(bs, 0) / total for bs in basis_states], dtype=float)
        p_est = Minv @ p_raw
        p_est = np.clip(p_est, 0.0, 1.0)
        if p_est.sum() > 0:
            p_est = p_est / p_est.sum()
        mitigated_counts = {bs: float(p * total) for bs, p in zip(basis_states, p_est)}
        return mitigated_counts

    return mitigate_counts


def evaluate_on_backend(
    circuits: Dict[str, list[str]],
    backend_name: str,
    backend_obj,
    shots: int,
    target_bitstrings: List[str],
    opt_level: int,
    seed: int,
    initial_layout: List[int] | None = None,
    randomized_compile_flag: bool = False,
    readout_mitigation: bool = False,
    use_noise_model: bool = False,
    hamiltonian_matrix: Optional[np.ndarray] = None,
    hamiltonian_nqubits: Optional[int] = None,
) -> Dict[str, Dict]:
    results = []
    noise_model = None
    if use_noise_model:
        try:
            noise_model = NoiseModel.from_backend(backend_obj)
        except Exception as exc:
            print(f"[qiskit_hw_eval] Warning: failed to build noise model from {backend_name}: {exc}")
            noise_model = None

    energy_mode = hamiltonian_matrix is not None

    if noise_model is not None:
        coupling_map = None
        try:
            cfg = backend_obj.configuration() if callable(getattr(backend_obj, "configuration", None)) else None
            coupling_map = getattr(cfg, "coupling_map", None) if cfg else None
        except Exception:
            coupling_map = None
        sim_kwargs = {
            "noise_model": noise_model,
            "basis_gates": getattr(noise_model, "basis_gates", None),
            "coupling_map": coupling_map,
        }
        if energy_mode:
            sim_kwargs["method"] = "density_matrix"
        sim = AerSimulator(**sim_kwargs)
    else:
        sim = AerSimulator.from_backend(backend_obj)
        if energy_mode:
            try:
                sim.set_options(method="density_matrix")
            except Exception:
                sim = AerSimulator(method="density_matrix")

    meas_filter = None
    cal_data = None
    mitigate_from_matrix = None
    basis_states = None
    if readout_mitigation and not energy_mode:
        # Estimate max width to size mitigation tools
        max_width = 0
        for paths in circuits.values():
            for path in paths:
                qc = load_qiskit_circuit_from_json(path, randomized_compile_flag=False)
                max_width = max(max_width, qc.num_qubits)
        basis_states = [format(i, f"0{max_width}b") for i in range(2**max_width)] if max_width > 0 else None
        # Prefer noise-model-derived matrix; fall back to calibration circuits
        if noise_model is not None and max_width > 0:
            M = _readout_matrix_from_noise_model(noise_model, max_width)
            mitigate_from_matrix = _mitigator_from_noise_matrix(M)
            if mitigate_from_matrix:
                print(f"[qiskit_hw_eval] Using noise-model readout matrix for backend {backend_name} (n_qubits={max_width})")
        if mitigate_from_matrix is None:
            try:
                meas_filter, cal_data = _build_readout_filter(
                    sim,
                    backend_obj,
                    max_width,
                    shots=shots,
                    opt_level=opt_level,
                    seed=seed,
                    initial_layout=initial_layout,
                )
                if meas_filter:
                    print(f"[qiskit_hw_eval] Built readout mitigation filter for backend {backend_name} (n_qubits={max_width})")
            except Exception as exc:
                print(f"[qiskit_hw_eval] Warning: readout mitigation unavailable on {backend_name}: {exc}")
                meas_filter = None

    for label, paths in circuits.items():
        for path in paths:
            rng = np.random.default_rng(seed=seed)
            qc = load_qiskit_circuit_from_json(path, randomized_compile_flag=randomized_compile_flag, rng=rng)
            if energy_mode and hamiltonian_nqubits is not None and qc.num_qubits < hamiltonian_nqubits:
                from qiskit import QuantumRegister

                deficit = hamiltonian_nqubits - qc.num_qubits
                anc = QuantumRegister(deficit, "anc")
                qc.add_register(anc)

            entry: Dict[str, object] = {
                "backend": backend_name,
                "circuit": label,
                "circuit_path": path,
                "shots": shots,
            }
            if energy_mode and hamiltonian_matrix is not None:
                qc_energy = qc.copy()
                qc_energy.save_density_matrix(label=DENSITY_SAVE_LABEL)
                entry["depth"] = qc_energy.depth()
                entry["width"] = qc_energy.num_qubits
                entry["gate_counts"] = gate_counts(qc_energy)
                job = sim.run(qc_energy, shots=1, seed_simulator=seed)
                try:
                    dm = job.result().data(0)[DENSITY_SAVE_LABEL]
                    energy = _energy_from_density_matrix(dm, hamiltonian_matrix)
                except Exception as exc:
                    print(f"[qiskit_hw_eval] Warning: failed to compute energy on {backend_name}/{label}: {exc}")
                    energy = float("nan")
                entry["energy"] = energy
                entry["metric"] = "energy"
            else:
                qc_meas = ensure_measurements(qc)
                tqc = transpile(
                    qc_meas,
                    backend=backend_obj,
                    optimization_level=opt_level,
                    seed_transpiler=seed,
                    initial_layout=initial_layout,
                )
                entry["depth"] = tqc.depth()
                entry["width"] = tqc.num_qubits
                entry["gate_counts"] = gate_counts(tqc)
                job = sim.run(tqc, shots=shots, seed_simulator=seed)
                res = job.result()
                counts_raw = res.get_counts()
                counts = counts_raw
                success_prob_raw = compute_success_prob(counts_raw, target_bitstrings)
                success_prob = success_prob_raw
                mitigated_counts = None
                if meas_filter is not None:
                    try:
                        mitigated_counts = meas_filter(counts_raw)
                        success_prob = compute_success_prob(mitigated_counts, target_bitstrings)
                    except Exception as exc:
                        print(f"[qiskit_hw_eval] Warning: failed to apply readout mitigation on {backend_name}/{label}: {exc}")
                        mitigated_counts = None
                elif mitigate_from_matrix is not None and basis_states is not None:
                    try:
                        mitigated_counts = mitigate_from_matrix(counts_raw, basis_states)
                        success_prob = compute_success_prob(mitigated_counts, target_bitstrings)
                    except Exception as exc:
                        print(f"[qiskit_hw_eval] Warning: failed to apply matrix-based readout mitigation on {backend_name}/{label}: {exc}")
                        mitigated_counts = None

                entry.update(
                    {
                        "success_prob": success_prob,
                        "success_prob_raw": success_prob_raw,
                        "counts": counts,
                        "counts_raw": counts_raw,
                        "counts_mitigated": mitigated_counts,
                        "readout_mitigated": bool(mitigated_counts),
                        "readout_calibration": cal_data if cal_data else None,
                        "noise_model": bool(noise_model),
                        "metric": "success_prob",
                    }
                )
            results.append(entry)
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
    randomized_compile_flag: bool = False,
    readout_mitigation: bool = False,
    use_noise_model: bool = False,
    hamiltonian_matrix: Optional[np.ndarray] = None,
    hamiltonian_nqubits: Optional[int] = None,
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
    variants = [("untwirled", False)]
    if randomized_compile_flag:
        variants.append(("twirled", True))
    for name, backend_obj in backend_objs.items():
        combined = []
        for variant, twirl in variants:
            backend_results = evaluate_on_backend(
                circuits,
                backend_name=name,
                backend_obj=backend_obj,
                shots=shots,
                target_bitstrings=target_bitstrings,
                opt_level=opt_level,
                seed=seed + (1 if twirl else 0),
                initial_layout=initial_layout,
                randomized_compile_flag=twirl,
                readout_mitigation=readout_mitigation,
                use_noise_model=use_noise_model,
                hamiltonian_matrix=hamiltonian_matrix,
                hamiltonian_nqubits=hamiltonian_nqubits,
            )
            for res in backend_results:
                res["variant"] = variant
            combined.extend(backend_results)
        all_results[name] = combined

    # Save JSON
    json_path = os.path.join(output_dir, "hardware_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Aggregate plot (mean/std across runs per backend/circuit)
    try:
        from experiments.analysis.plot_hw_fidelity import load_results, plot
        df = load_results(Path(json_path))
        plot(df, Path(output_dir) / "hardware_eval_plot.png")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[qiskit_hw_eval] Warning: failed to generate aggregate plot: {exc}")

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
    parser.add_argument(
        "--randomized-compiling",
        action="store_true",
        dest="randomized_compiling",
        help="Enable Pauli twirling before hardware eval (off by default).",
    )
    parser.add_argument(
        "--readout-mitigation",
        action="store_true",
        help="Enable simple readout error mitigation via calibration and matrix inversion (Ignis).",
    )
    parser.add_argument(
        "--use-noise-model",
        action="store_true",
        help="Use an Aer NoiseModel built from the fake backend properties for gate/readout noise.",
    )
    parser.set_defaults(randomized_compiling=False)
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
        randomized_compile_flag=args.randomized_compiling,
        readout_mitigation=args.readout_mitigation,
        use_noise_model=args.use_noise_model,
    )


if __name__ == "__main__":
    cli()
