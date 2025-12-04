#!/usr/bin/env python3
"""
QuantumNAS/TorchQuantum entry point with per-task runners.

Supports:
  - classification: experiments.quantumnas.train_quantumnas_classification
  - vqe: experiments.quantumnas.train_quantumnas_vqe
"""
from __future__ import annotations

import argparse
import sys
import importlib.util
from pathlib import Path

# Ensure repo root on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantumNAS task runner.")
    sub = p.add_subparsers(dest="task", required=True)

    p_cls = sub.add_parser("classification", help="Run classification baseline")
    p_cls.add_argument("--dataset", choices=["mnist4", "fashionmnist4"], default="mnist4")
    p_cls.add_argument("--epochs", type=int, default=5)
    p_cls.add_argument("--batch-size", type=int, default=64)
    p_cls.add_argument("--n-layers", type=int, default=2)
    p_cls.add_argument("--n-wires", type=int, default=4)
    p_cls.add_argument("--lr", type=float, default=1e-2)
    p_cls.add_argument("--train-samples", type=int, default=512, help="Number of train samples (None for full).")
    p_cls.add_argument("--valid-samples", type=int, default=256, help="Number of valid samples (None for full).")
    p_cls.add_argument("--out-dir", required=True)

    p_vqe = sub.add_parser("vqe", help="Run VQE baseline")
    p_vqe.add_argument("--molecule", choices=["H2", "HeH+", "LiH", "BeH2"], default="H2")
    p_vqe.add_argument("--n-layers", type=int, default=3)
    p_vqe.add_argument("--lr", type=float, default=0.1)
    p_vqe.add_argument("--steps", type=int, default=200)
    p_vqe.add_argument("--out-dir", required=True)
    p_vqe.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.task == "classification":
        try:
            from experiments.quantumnas.train_quantumnas_classification import run as cls_run
        except ImportError:
            spec = importlib.util.spec_from_file_location(
                "train_quantumnas_classification",
                Path(__file__).parent / "quantumnas" / "train_quantumnas_classification.py",
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            cls_run = mod.run  # type: ignore[attr-defined]
        cls_run(args)
    elif args.task == "vqe":
        try:
            from experiments.quantumnas.train_quantumnas_vqe import run as vqe_run
        except ImportError:
            spec = importlib.util.spec_from_file_location(
                "train_quantumnas_vqe",
                Path(__file__).parent / "quantumnas" / "train_quantumnas_vqe.py",
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            vqe_run = mod.run  # type: ignore[attr-defined]
        vqe_run(args)
    else:
        raise ValueError(f"Unknown task {args.task}")


if __name__ == "__main__":
    main()
