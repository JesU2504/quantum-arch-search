#!/usr/bin/env python3
"""
Lightweight TorchQuantum classification harness that trains a small quantum
classifier on a 4-class dataset (MNIST or FashionMNIST) and exports
op_history, QASM, and Cirq JSON artifacts for downstream robustness analysis.

This is intentionally minimal and CPU-friendly: it uses a simple
RY-entangling ansatz and maps Z-expectations directly to class logits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.dataset.mnist import MNISTDataset

import sys
import os

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from utils.torchquantum_adapter import convert_qasm_file_to_cirq  # noqa: E402
from torchquantum.plugin import op_history2qiskit  # noqa: E402

DATASET_CHOICES = ["mnist4", "fashionmnist4"]

def _dump_qasm(circ) -> str:
    try:
        from qiskit import qasm2  # type: ignore
    except Exception:
        return circ.qasm()
    else:
        return qasm2.dumps(circ)


def build_loaders(
    dataset: str,
    batch_size: int,
    n_train_samples: int = 512,
    n_valid_samples: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    is_fashion = dataset == "fashionmnist4"
    digits = [0, 1, 2, 3]
    common_kwargs = dict(
        root=str(_repo_root / "data"),
        train_valid_split_ratio=[0.9, 0.1],
        center_crop=28,
        resize=28,
        resize_mode="bilinear",
        binarize=False,
        binarize_threshold=0.5,
        digits_of_interest=digits,
        n_test_samples=None,
        n_valid_samples=n_valid_samples,
        n_train_samples=n_train_samples,
        fashion=is_fashion,
    )
    train_ds = MNISTDataset(split="train", **common_kwargs)
    valid_ds = MNISTDataset(split="valid", **common_kwargs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader


class SimpleQClassifier(tq.QuantumModule):
    def __init__(self, n_wires: int, n_classes: int, n_layers: int):
        super().__init__()
        self.n_wires = n_wires
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.theta = torch.nn.Parameter(torch.randn(n_layers, n_wires) * 0.1)
        self.enc_linear = torch.nn.Linear(28 * 28, n_wires)
        self.head_quantum = torch.nn.Linear(n_wires, n_classes)
        self.head_classical = torch.nn.Linear(28 * 28, n_classes)

    def forward(self, x: torch.Tensor, record_op: bool = False):
        bsz = x.shape[0]
        flat = x.view(bsz, -1)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, record_op=record_op, device=x.device)
        angles = self.enc_linear(flat)

        for i in range(self.n_wires):
            tqf.ry(qdev, wires=i, params=angles[:, i])

        for l in range(self.n_layers):
            for i in range(self.n_wires):
                tqf.ry(qdev, wires=i, params=self.theta[l, i])
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])

        meas = tq.MeasureAll(tq.PauliZ)(qdev)
        logits_q = self.head_quantum(meas[:, : self.n_classes])
        logits_c = self.head_classical(flat)
        logits = logits_q + logits_c
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs, qdev
def train(
    dataset: str,
    out_dir: Path,
    epochs: int = 5,
    batch_size: int = 64,
    n_layers: int = 2,
    lr: float = 1e-2,
    n_wires: int = 4,
) -> Tuple[SimpleQClassifier, float]:
    train_loader, valid_loader = build_loaders(dataset, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleQClassifier(n_wires=n_wires, n_classes=4, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["digit"].to(device)
            opt.zero_grad()
            log_probs, _ = model(images, record_op=False)
            loss = F.nll_loss(log_probs, targets)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch["image"].to(device)
                targets = batch["digit"].to(device)
                log_probs, _ = model(images, record_op=False)
                pred = log_probs.argmax(dim=1)
                correct += (pred == targets).sum().item()
                total += targets.numel()
        acc = correct / max(1, total)
        best_acc = max(best_acc, acc)
        print(f"[epoch {epoch}] valid acc={acc:.4f}, best={best_acc:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "classifier.pt")
    return model, best_acc


def export_artifacts(model: SimpleQClassifier, out_dir: Path):
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 1, 28, 28, device=device)
    _, qdev = model(dummy, record_op=True)
    op_history = qdev.op_history
    (out_dir / "op_history.json").write_text(json.dumps(op_history, indent=2))

    circ = op_history2qiskit(model.n_wires, op_history)
    qasm_path = out_dir / "circuit_quantumnas.qasm"
    qasm_path.write_text(_dump_qasm(circ))
    cirq_path = out_dir / "circuit_quantumnas.json"
    convert_qasm_file_to_cirq(qasm_path, cirq_path)
    return {
        "op_history": str(qasm_path),
        "qasm": str(qasm_path),
        "cirq_json": str(cirq_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TorchQuantum classification baseline (quantumnas-style export).")
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="mnist4")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-wires", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--out-dir", required=True, help="Output directory for artifacts.")
    return p.parse_args()


def main():
    run(parse_args())


def run(args: argparse.Namespace):
    out_dir = Path(args.out_dir).expanduser().resolve()
    model, best_acc = train(
        dataset=args.dataset,
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_layers=args.n_layers,
        lr=args.lr,
        n_wires=args.n_wires,
    )
    export_artifacts(model, out_dir)
    meta = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_layers": args.n_layers,
        "n_wires": args.n_wires,
        "lr": args.lr,
        "best_valid_acc": best_acc,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
