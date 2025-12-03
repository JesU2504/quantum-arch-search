#!/usr/bin/env python3
"""
Convert a TorchQuantum/Qiskit-exported OpenQASM circuit to Cirq JSON.

Usage:
  python experiments/import_quantumnas_qasm.py --qasm path/to/circuit.qasm --out results/run_x/quantumnas/circuit_quantumnas.json

This is intended for integrating TorchQuantum QuantumNAS/VQE outputs into the
existing robustness analysis, which consumes Cirq JSON circuits.
"""

import argparse
from pathlib import Path
import sys

# Add repository root to sys.path for standalone execution
import os
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from utils.torchquantum_adapter import convert_qasm_file_to_cirq


def parse_args():
    p = argparse.ArgumentParser(description="Import OpenQASM circuit into Cirq JSON for analysis.")
    p.add_argument('--qasm', required=True, help="Path to OpenQASM 2.0 file exported from TorchQuantum/Qiskit.")
    p.add_argument('--out', default=None, help="Output path for Cirq JSON. Default: replace .qasm with .json in-place.")
    return p.parse_args()


def main():
    args = parse_args()
    qasm_path = Path(args.qasm).expanduser().resolve()
    if not qasm_path.exists():
        raise FileNotFoundError(f"QASM file not found: {qasm_path}")
    out_path = Path(args.out).expanduser().resolve() if args.out else None
    saved = convert_qasm_file_to_cirq(qasm_path, out_path)
    print(f"Saved Cirq circuit to {saved}")


if __name__ == "__main__":
    main()
