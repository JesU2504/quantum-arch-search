#!/usr/bin/env python3
"""
Minimal TorchQuantum VQE harness that optimizes a small hardware-efficient
ansatz against a predefined molecular Hamiltonian and exports op_history,
QASM, and Cirq JSON artifacts.

Note: Hamiltonians are lightweight placeholders (2-qubit for H2/HeH+,
4-qubit for LiH/BeH2) to keep runs CPU-friendly; swap in accurate integrals
if needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.plugin import op_history2qiskit

import sys
import os

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from utils.torchquantum_adapter import convert_qasm_file_to_cirq  # noqa: E402
from utils.standard_hamiltonians import STANDARD_GEOM  # noqa: E402

def _dump_qasm(circ) -> str:
    try:
        from qiskit import qasm2  # type: ignore
    except Exception:
        return circ.qasm()
    else:
        return qasm2.dumps(circ)

from utils.standard_hamiltonians import get_standard_hamiltonian  # noqa: E402


def pauli_expectation(state: torch.Tensor, pauli: str) -> torch.Tensor:
    op_map = {
        "I": torch.tensor([[1, 0], [0, 1]], dtype=state.dtype, device=state.device),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=state.dtype, device=state.device),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=state.dtype, device=state.device),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=state.dtype, device=state.device),
    }
    op = op_map[pauli[0]]
    for p in pauli[1:]:
        op = torch.kron(op, op_map[p])
    return (state.conj() @ (op @ state)).real


class HardwareEfficientAnsatz(tq.QuantumModule):
    def __init__(self, n_wires: int, n_layers: int):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.theta = torch.nn.Parameter(torch.randn(n_layers, n_wires) * 0.1)

    def forward(self, record_op: bool = False):
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, record_op=record_op)
        for l in range(self.n_layers):
            for i in range(self.n_wires):
                tqf.ry(qdev, wires=i, params=self.theta[l, i])
            if self.n_wires > 1:
                for i in range(self.n_wires):
                    tqf.cnot(qdev, wires=[i, (i + 1) % self.n_wires])
        return qdev


def energy(qdev: tq.QuantumDevice, hamiltonian: List[Tuple[float, str]]) -> torch.Tensor:
    state = qdev.get_states_1d()[0]
    terms = []
    for coeff, pauli in hamiltonian:
        terms.append(coeff * pauli_expectation(state, pauli))
    return torch.stack(terms).sum()


def _set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_vqe(
    molecule: str, out_dir: Path, n_layers: int = 3, lr: float = 0.1, steps: int = 200, seed: int = 0
) -> Tuple[HardwareEfficientAnsatz, float, list]:
    info = get_standard_hamiltonian(molecule)
    n_wires, hamiltonian = info["n_qubits"], info["pauli_terms"]
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HardwareEfficientAnsatz(n_wires=n_wires, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_e = None
    history = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        qdev = model(record_op=False)
        e = energy(qdev, hamiltonian)
        e.backward()
        opt.step()
        if best_e is None or e.item() < best_e:
            best_e = e.item()
        history.append({"step": step, "energy": e.item()})
        if step % 50 == 0 or step == steps:
            print(f"[step {step}] energy={e.item():.6f}, best={best_e:.6f}")
    return model, best_e, history


def export_artifacts(model: HardwareEfficientAnsatz, out_dir: Path, molecule: str, ham_info: Dict[str, object]):
    qdev = model(record_op=True)
    op_history = qdev.op_history
    (out_dir / "op_history.json").write_text(json.dumps(op_history, indent=2))
    circ = op_history2qiskit(model.n_wires, op_history)
    qasm_path = out_dir / "circuit_quantumnas.qasm"
    qasm_path.write_text(_dump_qasm(circ))
    cirq_path = out_dir / "circuit_quantumnas.json"
    convert_qasm_file_to_cirq(qasm_path, cirq_path)
    meta = {
        "molecule": molecule,
        "n_wires": model.n_wires,
        "hf_energy": ham_info.get("hf_energy"),
        "fci_energy": ham_info.get("fci_energy"),
        "pauli_terms": ham_info.get("pauli_terms"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"qasm": str(qasm_path), "cirq_json": str(cirq_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TorchQuantum VQE baseline (quantumnas-style export).")
    p.add_argument("--molecule", choices=sorted(STANDARD_GEOM.keys()), default="H2")
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--out-dir", required=True, help="Output directory for artifacts.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    run(parse_args())


def run(args: argparse.Namespace):
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ham_info = get_standard_hamiltonian(args.molecule)
    model, best_e, history = train_vqe(
        molecule=args.molecule,
        out_dir=out_dir,
        n_layers=args.n_layers,
        lr=args.lr,
        steps=args.steps,
        seed=args.seed,
    )
    export_artifacts(model, out_dir, args.molecule, ham_info)
    extra = {
        "molecule": args.molecule,
        "best_energy": best_e,
        "hf_energy": ham_info["hf_energy"],
        "fci_energy": ham_info["fci_energy"],
        "seed": args.seed,
        "energy_trace": history,
    }
    (out_dir / "results.json").write_text(json.dumps(extra, indent=2))
    print(f"Saved VQE artifacts to {out_dir} (best energy {best_e:.6f})")


if __name__ == "__main__":
    main()
