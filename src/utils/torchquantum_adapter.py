"""
Utilities to bridge TorchQuantum / Qiskit outputs into Cirq for evaluation.

TorchQuantum examples (including QuantumNAS and VQE) often export or can export
circuits as Qiskit QuantumCircuit objects or OpenQASM strings. Our evaluation
stack uses Cirq, so these helpers convert Qiskit/QASM to Cirq and persist in
the repo's JSON format for robustness analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cirq

try:
    from cirq.contrib.qasm_import import circuit_from_qasm
except Exception as _exc:
    circuit_from_qasm = None  # type: ignore[assignment]
    _cirq_qasm_import_error = _exc
else:
    _cirq_qasm_import_error = None


def qasm_to_cirq(qasm_str: str) -> cirq.Circuit:
    """
    Convert an OpenQASM 2.0 string to a Cirq Circuit.

    Requires cirq.contrib.qasm_import to be available.
    """
    if circuit_from_qasm is None:
        raise ImportError(
            "cirq.contrib.qasm_import is unavailable; cannot import QASM. "
            f"Underlying error: {_cirq_qasm_import_error}"
        )
    return circuit_from_qasm(qasm_str)


def save_cirq_circuit(circuit: cirq.Circuit, out_path: Path) -> None:
    """
    Save a Cirq circuit to JSON using cirq.read_json compatibility.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cirq.to_json(circuit, out_path)


def convert_qasm_file_to_cirq(qasm_path: Path, out_path: Optional[Path] = None) -> Path:
    """
    Read a QASM file and save as Cirq JSON circuit.

    Args:
        qasm_path: Path to an OpenQASM 2.0 file.
        out_path: Optional output path for the Cirq JSON. If None, uses the
            same stem with .json extension in the same directory.

    Returns:
        Path to the written Cirq JSON file.
    """
    if out_path is None:
        out_path = qasm_path.with_suffix(".json")
    qasm_str = qasm_path.read_text()
    circuit = qasm_to_cirq(qasm_str)
    save_cirq_circuit(circuit, out_path)
    return out_path
