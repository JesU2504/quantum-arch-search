#!/usr/bin/env python3
"""
Noise-robust VQE baseline trained with simple adversarial noise sampling.

Supported molecules: H2, HeH+, LiH, BeH2 (same lightweight Hamiltonians as the
TorchQuantum QuantumNAS harness).

At each optimization step we evaluate the clean energy and its depolarized
counterparts for a set of user-provided noise rates, then minimize the
worst-case (max) energy. The learned circuit is exported as TorchQuantum
op_history, QASM, and Cirq JSON for downstream comparison.

Example:
    python -m experiments.adversarial.train_adversarial_vqe \
        --molecule H2 --steps 150 --noise-levels 0.0 0.02 0.05 \
        --out-dir results/adversarial_vqe_h2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torchquantum.plugin import op_history2qiskit

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.quantumnas.train_quantumnas_vqe import (  # noqa: E402
    HardwareEfficientAnsatz,
    energy,
)
from utils.standard_hamiltonians import STANDARD_GEOM, get_standard_hamiltonian  # noqa: E402
from utils.torchquantum_adapter import convert_qasm_file_to_cirq  # noqa: E402


def _dump_qasm(circ) -> str:
    """Serialize qiskit circuit to QASM, using qasm2 if available."""
    try:
        from qiskit import qasm2  # type: ignore
    except Exception:
        return circ.qasm()
    else:
        return qasm2.dumps(circ)


def _mixed_state_energy(hamiltonian: List[Tuple[float, str]], n_qubits: int) -> float:
    """
    Energy of the maximally mixed state for a Pauli-sum Hamiltonian.
    Only all-identity terms contribute to the trace.
    """
    dim = 2 ** n_qubits
    trace = 0.0
    for coeff, pauli in hamiltonian:
        if all(p == "I" for p in pauli):
            trace += coeff * dim
    return trace / dim


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_adversarial_vqe(
    molecule: str,
    noise_levels: Sequence[float],
    n_layers: int = 3,
    steps: int = 200,
    lr: float = 0.1,
    seed: int = 0,
    noise_samples_per_step: int = 0,
) -> Tuple[HardwareEfficientAnsatz, dict, list]:
    info = get_standard_hamiltonian(molecule)
    n_wires, hamiltonian = info["n_qubits"], info["pauli_terms"]
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HardwareEfficientAnsatz(n_wires=n_wires, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    noise_tensor = torch.tensor(noise_levels, device=device, dtype=torch.float32)
    mixed_energy = torch.tensor(
        _mixed_state_energy(hamiltonian, n_wires),
        device=device,
        dtype=torch.float32,
    )

    history = []
    best = {"worst_energy": float("inf"), "clean_energy": None, "step": None}

    for step_idx in range(1, steps + 1):
        opt.zero_grad()
        # record_op=True to enable per-step circuit logging
        qdev = model(record_op=True)
        clean_e = energy(qdev, hamiltonian)
        # Optionally augment with random noise samples each step
        if noise_samples_per_step > 0:
            rand = torch.rand(noise_samples_per_step, device=device) * noise_tensor.max()
            noise_use = torch.cat([noise_tensor, rand])
        else:
            noise_use = noise_tensor
        robust_e = (1 - noise_use) * clean_e + noise_use * mixed_energy
        worst_e = robust_e.max()

        worst_e.backward()
        opt.step()

        if worst_e.item() < best["worst_energy"]:
            best.update(
                {
                    "worst_energy": worst_e.item(),
                    "clean_energy": clean_e.item(),
                    "step": step_idx,
                }
            )

        if step_idx == 1 or step_idx % 20 == 0 or step_idx == steps:
            print(
                f"[step {step_idx:4d}] clean={clean_e.item():.6f} "
                f"worst={worst_e.item():.6f} (noise max={max(noise_levels):.3f})"
            )

        history.append(
            {
                "step": step_idx,
                "clean_energy": clean_e.item(),
                "worst_energy": worst_e.item(),
                "qasm": _dump_qasm(op_history2qiskit(model.n_wires, qdev.op_history)),
            }
        )

    return model, best, history, info


def export_artifacts(
    model: HardwareEfficientAnsatz,
    out_dir: Path,
    molecule: str,
    noise_levels: Iterable[float],
    best: dict,
    history: list,
    ham_info: dict,
    seed: int,
    eval_sweep: List[Tuple[float, float]] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    qdev = model(record_op=True)
    op_history = qdev.op_history
    (out_dir / "op_history_adversarial_vqe.json").write_text(
        json.dumps(op_history, indent=2)
    )

    circ = op_history2qiskit(model.n_wires, op_history)
    qasm_path = out_dir / "circuit_adversarial_vqe.qasm"
    qasm_path.write_text(_dump_qasm(circ))
    convert_qasm_file_to_cirq(qasm_path, out_dir / "circuit_adversarial_vqe.json")

    results = {
        "molecule": molecule,
        "noise_levels": list(noise_levels),
        "best_worst_energy": best["worst_energy"],
        "best_clean_energy": best["clean_energy"],
        "best_step": best["step"],
        "n_wires": model.n_wires,
        "n_layers": model.n_layers,
        "hf_energy": ham_info.get("hf_energy"),
        "fci_energy": ham_info.get("fci_energy"),
        "history": history,
        "seed": seed,
        "eval_sweep": eval_sweep,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved adversarial VQE artifacts to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarially trained VQE (noise-robust) using TorchQuantum."
    )
    parser.add_argument(
        "--molecule",
        choices=sorted(STANDARD_GEOM.keys()),
        default="H2",
        help="Molecule Hamiltonian to target.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05],
        help="Depolarizing noise rates to adversarially train against.",
    )
    parser.add_argument("--n-layers", type=int, default=3, help="Ansatz depth.")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps.")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--noise-samples-per-step",
        type=int,
        default=0,
        help="Extra random noise samples per step (uniform in [0,max(noise_levels)]).",
    )
    parser.add_argument(
        "--eval-noise-levels",
        type=float,
        nargs="+",
        default=None,
        help="Noise levels to evaluate after training (defaults to training list).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Where to store QASM/op_history/results artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    eval_noise = args.eval_noise_levels if args.eval_noise_levels is not None else args.noise_levels
    model, best, history, ham_info = train_adversarial_vqe(
        molecule=args.molecule,
        noise_levels=args.noise_levels,
        n_layers=args.n_layers,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        noise_samples_per_step=args.noise_samples_per_step,
    )
    # Evaluate final model over a sweep of noise levels
    eval_sweep = []
    with torch.no_grad():
        qdev = model(record_op=False)
        clean_e = energy(qdev, ham_info["pauli_terms"]).item()
        mixed_e = _mixed_state_energy(ham_info["pauli_terms"], ham_info["n_qubits"])
        for p in eval_noise:
            eval_sweep.append((p, (1 - p) * clean_e + p * mixed_e))

    export_artifacts(
        model,
        out_dir,
        args.molecule,
        args.noise_levels,
        best,
        history,
        ham_info,
        args.seed,
        eval_sweep,
    )


if __name__ == "__main__":
    main()
