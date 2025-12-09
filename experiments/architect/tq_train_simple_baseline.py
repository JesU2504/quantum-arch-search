#!/usr/bin/env python3
"""
Lightweight TorchQuantum training to produce GHZ (state prep) and Toffoli (unitary) circuits.

This script avoids heavy QuantumNAS dependencies and exports a Cirq JSON circuit
compatible with the existing robustness analysis.

It uses a small, hand-crafted ansatz trained with Adam. TorchQuantum is
monkey-patched to skip incompatible qiskit imports in modern qiskit versions.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List
import random

import torch
import torch.nn as nn
import numpy as np
import cirq

# --- TorchQuantum import shim (bypass legacy qiskit imports) ---
def _shim_torchquantum_imports():
    import types
    import qiskit  # type: ignore
    # Stub missing aer noise module pieces expected by torchquantum util
    if 'qiskit.providers.aer.noise' not in sys.modules:
        noise_mod = types.ModuleType('qiskit.providers.aer.noise')
        noise_mod.NoiseModel = type('NoiseModel', (), {})
        sys.modules['qiskit.providers.aer.noise'] = noise_mod
    for name in [
        'qiskit.providers.aer',
        'qiskit.providers.aer.noise.device',
        'qiskit.providers.aer.noise.device.parameters',
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    params_mod = sys.modules['qiskit.providers.aer.noise.device.parameters']
    params_mod.gate_error_values = {}
    # Stub torchquantum.plugin.qiskit constant to avoid deep qiskit imports
    plugin_mod = types.ModuleType('torchquantum.plugin')
    qiskit_mod = types.ModuleType('torchquantum.plugin.qiskit')
    qiskit_mod.QISKIT_INCOMPATIBLE_FUNC_NAMES = set()
    sys.modules['torchquantum.plugin'] = plugin_mod
    sys.modules['torchquantum.plugin.qiskit'] = qiskit_mod


_shim_torchquantum_imports()
import torchquantum as tq  # noqa: E402

# --- Target utilities ---
def ghz_target_state(n_qubits: int) -> torch.Tensor:
    """Return |GHZ_n> as a torch complex tensor (length 2^n)."""
    dim = 2 ** n_qubits
    state = torch.zeros(dim, dtype=torch.complex64)
    state[0] = 1 / np.sqrt(2)
    state[-1] = 1 / np.sqrt(2)
    return state


def toffoli_truth_table(n_qubits: int = 3) -> List[int]:
    """Return target basis mapping for Toffoli on n=3 (controls 0,1 -> target 2)."""
    if n_qubits != 3:
        raise ValueError("This simple trainer only supports 3-qubit Toffoli.")
    mapping = []
    for idx in range(8):
        b2 = (idx >> 2) & 1
        b1 = (idx >> 1) & 1
        b0 = idx & 1
        out_b0 = b0
        out_b1 = b1
        out_b2 = b2 ^ (b0 & b1)
        out_idx = (out_b2 << 2) | (out_b1 << 1) | out_b0
        mapping.append(out_idx)
    return mapping


# --- Ansatz definitions ---
class GHZAnsatz(tq.QuantumModule):
    def __init__(self, n_qubits: int = 3, depth: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.ry_layers = nn.ModuleList(
            [nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]) for _ in range(depth)]
        )
        self.cnot_layers = nn.ModuleList(
            [tq.CNOT() for _ in range(n_qubits - 1)]
        )

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for d in range(self.depth):
            for q in range(self.n_qubits):
                self.ry_layers[d][q](qdev, wires=q)
            # Chain entanglement
            for q in range(self.n_qubits - 1):
                self.cnot_layers[q](qdev, wires=[q, q + 1])


class ToffoliAnsatz(tq.QuantumModule):
    def __init__(self, n_qubits: int = 3, depth: int = 3):
        super().__init__()
        if n_qubits != 3:
            raise ValueError("Toffoli ansatz only supports 3 qubits.")
        self.n_qubits = n_qubits
        self.depth = depth
        self.ry_layers = nn.ModuleList(
            [nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]) for _ in range(depth)]
        )
        self.rz_layers = nn.ModuleList(
            [nn.ModuleList([tq.RZ(has_params=True, trainable=True) for _ in range(n_qubits)]) for _ in range(depth)]
        )
        self.cnot_layers = nn.ModuleList(
            [tq.CNOT() for _ in range(n_qubits - 1)]
        )

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for d in range(self.depth):
            for q in range(self.n_qubits):
                self.ry_layers[d][q](qdev, wires=q)
                self.rz_layers[d][q](qdev, wires=q)
            for q in range(self.n_qubits - 1):
                self.cnot_layers[q](qdev, wires=[q, q + 1])


# --- Training helpers ---
def fidelity_to_target(state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return |<target|state>|^2."""
    overlap = torch.vdot(target, state)
    return torch.abs(overlap) ** 2


def run_state_prep(n_qubits: int, epochs: int, lr: float, depth: int, out_dir: Path):
    target = ghz_target_state(n_qubits).to(torch.device("cpu"))
    model = GHZAnsatz(n_qubits=n_qubits, depth=depth)
    qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=1, device="cpu")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        qdev.reset_states(bsz=1)
        model(qdev)
        state = qdev.states.reshape(1, -1)[0]
        loss = 1 - fidelity_to_target(state, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % max(1, epochs // 10) == 0:
            fid = (1 - loss).item()
            print(f"[GHZ] epoch {epoch}/{epochs} fidelity {fid:.6f}")

    return model


def _basis_state_tensor(idx: int, n_qubits: int) -> torch.Tensor:
    dim = 2 ** n_qubits
    vec = torch.zeros(dim, dtype=torch.complex64)
    vec[idx] = 1.0 + 0.0j
    return vec.reshape([2] * n_qubits)


def run_unitary_prep_toffoli(epochs: int, lr: float, depth: int, out_dir: Path):
    n_qubits = 3
    mapping = toffoli_truth_table(n_qubits)
    model = ToffoliAnsatz(n_qubits=n_qubits, depth=depth)
    qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=1, device="cpu")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for idx_in, idx_out in enumerate(mapping):
            qdev.set_states(_basis_state_tensor(idx_in, n_qubits).unsqueeze(0))
            model(qdev)
            state = qdev.states.reshape(1, -1)[0]
            target = _basis_state_tensor(idx_out, n_qubits).reshape(-1)
            loss = 1 - fidelity_to_target(state, target)
            total_loss = total_loss + loss
        total_loss = total_loss / len(mapping)
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if epoch % max(1, epochs // 10) == 0:
            print(f"[Toffoli] epoch {epoch}/{epochs} avg loss {total_loss.item():.6f}")

    return model


# --- Export to Cirq ---
def export_to_cirq(model: tq.QuantumModule, n_qubits: int, out_path: Path):
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    ops = []
    # Traverse layers explicitly to preserve ordering
    if isinstance(model, GHZAnsatz):
        for d in range(model.depth):
            for q in range(model.n_qubits):
                theta = float(model.ry_layers[d][q].params.detach())
                ops.append(cirq.ry(theta).on(qubits[q]))
            for q in range(model.n_qubits - 1):
                ops.append(cirq.CNOT(qubits[q], qubits[q + 1]))
    elif isinstance(model, ToffoliAnsatz):
        for d in range(model.depth):
            for q in range(model.n_qubits):
                theta_ry = float(model.ry_layers[d][q].params.detach())
                theta_rz = float(model.rz_layers[d][q].params.detach())
                ops.append(cirq.ry(theta_ry).on(qubits[q]))
                ops.append(cirq.rz(theta_rz).on(qubits[q]))
            for q in range(model.n_qubits - 1):
                ops.append(cirq.CNOT(qubits[q], qubits[q + 1]))
    circuit = cirq.Circuit(ops)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cirq.to_json(circuit, out_path)
    return circuit


def parse_args():
    p = argparse.ArgumentParser(description="Train simple TorchQuantum baselines and export to Cirq JSON.")
    p.add_argument('--task', choices=['ghz', 'toffoli'], required=True)
    p.add_argument('--epochs', type=int, default=None, help="Training epochs (default: 400 for GHZ, 2000 for Toffoli)")
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--depth', type=int, default=None, help="Ansatz depth (layers of rotations + CNOT chain)")
    p.add_argument('--augment-depth', action='store_true', help="Double the default depth to increase circuit size for fairer comparison.")
    p.add_argument('--max-gates', type=int, default=None, help="Cap on total gate count (defaults to experiments.config.MAX_CIRCUIT_TIMESTEPS when set).")
    p.add_argument('--seed', type=int, default=None, help="Optional RNG seed for reproducibility.")
    p.add_argument('--out-dir', type=str, default='results/quantumnas_simple')
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None
    try:
        from experiments import config as exp_config
        default_max_gates = exp_config.MAX_CIRCUIT_TIMESTEPS
    except Exception:
        default_max_gates = None

    max_gates = args.max_gates if args.max_gates is not None else default_max_gates

    out_dir = Path(args.out_dir).expanduser()
    ghz_depth_default = 4
    toffoli_depth_default = 12
    if args.augment_depth:
        ghz_depth_default *= 2
        toffoli_depth_default *= 2
    ghz_epochs_default = 400
    toffoli_epochs_default = 2000

    def fit_depth(task: str, depth: int) -> int:
        """Adjust depth so total gates <= max_gates (if provided)."""
        if max_gates is None:
            return max(depth, 2)
        if task == 'ghz':
            gates_per_layer = 2 * 3 - 1  # 3 RY + 2 CNOT = 5 for n=3
        else:
            gates_per_layer = 2 * 3 + (3 - 1)  # 6 rotations + 2 CNOT = 8
        max_depth = max(2, max_gates // gates_per_layer)
        return max(2, min(depth, max_depth))

    if args.task == 'ghz':
        depth = args.depth if args.depth is not None else ghz_depth_default
        depth = fit_depth('ghz', depth)
        epochs = args.epochs if args.epochs is not None else ghz_epochs_default
        model = run_state_prep(n_qubits=3, epochs=epochs, lr=args.lr, depth=depth, out_dir=out_dir)
        circuit = export_to_cirq(model, n_qubits=3, out_path=out_dir / 'circuit_quantumnas.json')
        print(f"[GHZ] Saved circuit to {out_dir / 'circuit_quantumnas.json'} (depth={depth}, max_gates={max_gates})")
    else:
        depth = args.depth if args.depth is not None else toffoli_depth_default
        depth = fit_depth('toffoli', depth)
        epochs = args.epochs if args.epochs is not None else toffoli_epochs_default
        model = run_unitary_prep_toffoli(epochs=epochs, lr=args.lr, depth=depth, out_dir=out_dir)
        circuit = export_to_cirq(model, n_qubits=3, out_path=out_dir / 'circuit_quantumnas.json')
        print(f"[Toffoli] Saved circuit to {out_dir / 'circuit_quantumnas.json'} (depth={depth}, max_gates={max_gates})")


if __name__ == "__main__":
    main()
