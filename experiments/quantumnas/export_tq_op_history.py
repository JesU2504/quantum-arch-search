#!/usr/bin/env python3
"""
Convert a TorchQuantum op_history (list of ops) to QASM and Cirq JSON.

This is useful when QuantumNAS/VQE runs in TorchQuantum export their learned
circuits as op histories. We convert to Qiskit, dump QASM, and then to Cirq
JSON for downstream robustness analysis.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add repository root to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

try:
    import torchquantum as tq
    from torchquantum.plugin import op_history2qiskit
except ImportError as exc:
    tq = None
    _tq_import_error = exc

from utils.torchquantum_adapter import convert_qasm_file_to_cirq


def parse_args():
    p = argparse.ArgumentParser(description="Convert TorchQuantum op_history to QASM and Cirq JSON.")
    p.add_argument('--op-history', required=True, help="Path to op_history file (JSON or Torch saved list).")
    p.add_argument('--qasm-out', default=None, help="Output QASM path (default: op_history.qasm alongside input).")
    p.add_argument('--cirq-out', default=None, help="Output Cirq JSON path (default: op_history.json alongside input).")
    return p.parse_args()


def load_op_history(path: Path):
    """Load op_history from JSON or Torch saved file."""
    # Try JSON
    try:
        return json.loads(path.read_text())
    except Exception:
        pass
    # Try torch.load
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load op_history from {path}: {exc}")


def main():
    if tq is None:
        raise ImportError(
            "torchquantum is required for op_history export. "
            f"Install torchquantum and retry. Underlying import error: {_tq_import_error}"
        )

    args = parse_args()
    op_path = Path(args.op_history).expanduser().resolve()
    ops = load_op_history(op_path)

    circ = op_history2qiskit(ops)
    qasm_out = Path(args.qasm_out).expanduser().resolve() if args.qasm_out else op_path.with_suffix(".qasm")
    qasm_out.parent.mkdir(parents=True, exist_ok=True)
    qasm_out.write_text(circ.qasm())
    print(f"Saved QASM to {qasm_out}")

    cirq_out = Path(args.cirq_out).expanduser().resolve() if args.cirq_out else op_path.with_suffix(".json")
    convert_qasm_file_to_cirq(qasm_out, cirq_out)
    print(f"Saved Cirq JSON to {cirq_out}")


if __name__ == "__main__":
    main()
