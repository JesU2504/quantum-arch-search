#!/usr/bin/env python3
"""
Run the official QuantumNAS search/training loop for a paper benchmark and
convert the learned circuit into Cirq JSON for downstream evaluation.

This script is intentionally thin: it shells out to the QuantumNAS reference
implementation (e.g., the TorchQuantum release) and standardizes the exported
artifacts into ``results/<run>/quantumnas/circuit_quantumnas.json`` so the
existing robustness and comparison tooling can consume the circuit directly.

Included paper benchmarks
-------------------------
- **Classification:** TorchQuantum QuantumNAS pipeline on the following datasets
  (select via ``--dataset``): ``mnist4`` (default), ``fashionmnist4``,
  ``cifar10_4``, ``gtsrb4``, and ``vowel``.
- **VQE:** molecular Hamiltonians supported by the upstream QuantumNAS CLI and
  exercised in common benchmarks (select via ``--vqe-molecule``): ``H2``
  (default), ``LiH``, ``BeH2``, and ``HeH+``.

Prerequisites
-------------
Install the QuantumNAS/TorchQuantum dependencies in your environment before
running this harness; the script will raise a clear ``ImportError`` if they are
missing.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, "src")
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from experiments.export_tq_op_history import load_op_history  # noqa: E402
from utils.torchquantum_adapter import convert_qasm_file_to_cirq  # noqa: E402


DEFAULT_SEARCH_CMD = "python -m torchquantum.experiment.quantumnas.run_quantumnas"
SUPPORTED_CLASSIFICATION_DATASETS = {
    "mnist4": "QuantumNAS paper benchmark (default).",
    "fashionmnist4": "FashionMNIST 4-class reduction used in prior QuantumNAS experiments.",
    "cifar10_4": "CIFAR-10 reduced to four classes for lightweight benchmarking.",
    "gtsrb4": "German Traffic Sign Recognition Benchmark reduced to four classes.",
    "vowel": "Vowel classification benchmark used in QuantumNAS examples.",
}
SUPPORTED_VQE_MOLECULES = {
    "H2": "Hydrogen molecule ground-state energy benchmark (default).",
    "LiH": "Lithium hydride Hamiltonian used in QuantumNAS VQE examples.",
    "BeH2": "Beryllium hydride Hamiltonian for deeper VQE circuits.",
    "HeH+": "Helium hydride cation benchmark molecule.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QuantumNAS paper benchmark (classification or VQE) and export Cirq JSON."
    )
    parser.add_argument(
        "--task",
        choices=["classification", "vqe"],
        default="classification",
        help="Paper task to run via QuantumNAS.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_CLASSIFICATION_DATASETS.keys()),
        default="mnist4",
        help=(
            "Classification dataset forwarded to the QuantumNAS CLI. "
            "Choices reflect common paper benchmarks and example configs."
        ),
    )
    parser.add_argument(
        "--vqe-molecule",
        choices=sorted(SUPPORTED_VQE_MOLECULES.keys()),
        default="H2",
        help=(
            "Molecule identifier forwarded to the QuantumNAS CLI for VQE benchmarks. "
            "Choices mirror commonly used QuantumNAS/TorchQuantum Hamiltonians."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed forwarded to QuantumNAS.")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional training epoch override forwarded to QuantumNAS if supported by the CLI.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where circuit_quantumnas.json and auxiliary artifacts will be saved.",
    )
    parser.add_argument(
        "--search-cmd",
        default=DEFAULT_SEARCH_CMD,
        help="QuantumNAS search command to execute (string, parsed with shlex.split).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional QuantumNAS config file passed through to the search command.",
    )
    parser.add_argument(
        "--extra-cli-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended verbatim to the QuantumNAS CLI invocation.",
    )
    parser.add_argument(
        "--op-history",
        default=None,
        help="Skip training and import an existing TorchQuantum op_history instead of running the search.",
    )
    parser.add_argument(
        "--qasm",
        default=None,
        help="Skip training and import an existing QuantumNAS-exported QASM file instead of running the search.",
    )
    parser.add_argument(
        "--op-history-flag",
        default="--save-op-history",
        help="Flag name understood by the QuantumNAS CLI to persist op_history.",
    )
    parser.add_argument(
        "--qasm-flag",
        default="--save-qasm",
        help="Flag name understood by the QuantumNAS CLI to persist OpenQASM output.",
    )
    parser.add_argument(
        "--task-flag",
        default="--task",
        help="Flag name used by the QuantumNAS CLI for task selection.",
    )
    parser.add_argument(
        "--seed-flag",
        default="--seed",
        help="Flag name used by the QuantumNAS CLI for seeding.",
    )
    parser.add_argument(
        "--dataset-flag",
        default="--dataset",
        help="Flag name used by the QuantumNAS CLI for classification datasets.",
    )
    parser.add_argument(
        "--vqe-flag",
        default="--molecule",
        help="Flag name used by the QuantumNAS CLI for VQE molecule selection.",
    )
    parser.add_argument(
        "--epoch-flag",
        default="--max-epochs",
        help="Flag name used by the QuantumNAS CLI for epoch override.",
    )
    return parser.parse_args()


def ensure_module_available(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise ImportError(
            f"Required module '{name}' not found. Install the QuantumNAS/TorchQuantum dependencies before running this script."
        )


def run_quantumnas_cli(
    cmd_str: str,
    task: str,
    dataset: str,
    molecule: str,
    seed: int,
    max_epochs: Optional[int],
    op_history_path: Path,
    qasm_path: Path,
    flags: dict,
    config_path: Optional[str],
    extra_args: Optional[Iterable[str]],
) -> subprocess.CompletedProcess:
    ensure_module_available("torchquantum")
    ensure_module_available("torchquantum.experiment")

    cmd: List[str] = shlex.split(cmd_str)
    cmd.extend([flags["task"], task])
    cmd.extend([flags["seed"], str(seed)])
    if config_path:
        cmd.extend(["--config", config_path])
    if task == "classification":
        cmd.extend([flags["dataset"], dataset])
    else:
        cmd.extend([flags["vqe"], molecule])
    if max_epochs is not None:
        cmd.extend([flags["epoch"], str(max_epochs)])
    cmd.extend([flags["op_history"], str(op_history_path)])
    cmd.extend([flags["qasm"], str(qasm_path)])
    if extra_args:
        cmd.extend(extra_args)

    op_history_path.parent.mkdir(parents=True, exist_ok=True)
    qasm_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def convert_op_history_to_qasm(op_history_path: Path, qasm_path: Path) -> None:
    ensure_module_available("torchquantum")
    ensure_module_available("torchquantum.plugin")
    import torchquantum.plugin as tq_plugin

    ops = load_op_history(op_history_path)
    circ = tq_plugin.op_history2qiskit(ops)
    qasm_path.parent.mkdir(parents=True, exist_ok=True)
    qasm_path.write_text(circ.qasm())


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    op_history_path = out_dir / "op_history.json"
    qasm_path = out_dir / "circuit_quantumnas.qasm"
    cirq_out = out_dir / "circuit_quantumnas.json"

    if args.qasm:
        qasm_source = Path(args.qasm).expanduser().resolve()
        qasm_path = qasm_source
        convert_qasm_file_to_cirq(qasm_source, cirq_out)
    else:
        if args.op_history:
            op_history_path = Path(args.op_history).expanduser().resolve()
        else:
            flags = {
                "task": args.task_flag,
                "seed": args.seed_flag,
                "dataset": args.dataset_flag,
                "vqe": args.vqe_flag,
                "epoch": args.epoch_flag,
                "op_history": args.op_history_flag,
                "qasm": args.qasm_flag,
            }
            result = run_quantumnas_cli(
                cmd_str=args.search_cmd,
                task=args.task,
                dataset=args.dataset,
                molecule=args.vqe_molecule,
                seed=args.seed,
                max_epochs=args.max_epochs,
                op_history_path=op_history_path,
                qasm_path=qasm_path,
                flags=flags,
                config_path=args.config,
                extra_args=args.extra_cli_args,
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        if not qasm_path.exists():
            convert_op_history_to_qasm(op_history_path, qasm_path)
        convert_qasm_file_to_cirq(qasm_path, cirq_out)

    metadata = {
        "task": args.task,
        "dataset": args.dataset,
        "vqe_molecule": args.vqe_molecule,
        "seed": args.seed,
        "search_cmd": args.search_cmd,
        "config": args.config,
        "op_history": str(op_history_path),
        "qasm": str(qasm_path),
        "cirq_json": str(cirq_out),
    }
    meta_path = out_dir / "quantumnas_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Saved QuantumNAS circuit to {cirq_out}")


if __name__ == "__main__":
    main()
