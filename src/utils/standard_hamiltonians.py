"""
Load molecular Hamiltonians from pre-generated JSON files.

This implementation is Qiskit-free and cluster-safe.
Hamiltonians must exist in: <repo_root>/hamiltonians/*.json

We also expose STANDARD_GEOM as lightweight metadata (used by some envs),
but it is not used to build the Hamiltonians here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np

# ---------------------------------------------------------------------
# Lightweight geometry metadata (for envs that import STANDARD_GEOM)
# ---------------------------------------------------------------------

STANDARD_GEOM: Dict[str, Dict[str, object]] = {
    "H2": {
        "atom": "H 0 0 0; H 0 0 0.735",
        "charge": 0,
        "spin": 0,
        "basis": "sto3g",
    },
    "HeH+": {
        # Bond length ~1.4632 Å from common benchmarks
        "atom": "H 0 0 0; He 0 0 1.4632",
        "charge": 1,
        "spin": 0,
        "basis": "sto3g",
    },
    "LiH": {
        # Standard geometry ~1.6 Å; 2e,2o active space was used when generating the JSON
        "atom": "Li 0 0 0; H 0 0 1.6",
        "charge": 0,
        "spin": 0,
        "basis": "sto3g",
    },
    "BeH2": {
        # Linear BeH2: H at ±1.326 Å; 4e,4o active space was used when generating the JSON
        "atom": "Be 0 0 0; H 0 0 -1.326; H 0 0 1.326",
        "charge": 0,
        "spin": 0,
        "basis": "sto3g",
    },
}

# ---------------------------------------------------------------------
# Pauli matrices (pure NumPy)
# ---------------------------------------------------------------------

_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_string_matrix(label: str) -> np.ndarray:
    """
    Build the matrix for a Pauli string like 'IZXZ'.
    Leftmost character corresponds to the most significant qubit.
    """
    mat = np.array([[1]], dtype=complex)
    for p in label:
        mat = np.kron(mat, _PAULI[p])
    return mat


def _matrix_from_pauli_terms(
    pauli_terms: List[Tuple[float, str]]
) -> np.ndarray:
    """
    Construct full Hamiltonian matrix from Pauli terms.
    """
    if not pauli_terms:
        raise ValueError("Empty pauli_terms list.")

    n_qubits = len(pauli_terms[0][1])
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    for coeff, label in pauli_terms:
        if len(label) != n_qubits:
            raise ValueError(
                f"Inconsistent Pauli string length: {label}"
            )
        H += coeff * _pauli_string_matrix(label)

    return H


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_standard_hamiltonian(molecule: str) -> Dict[str, object]:
    """
    Load Hamiltonian from JSON.

    Expected JSON format:
    {
      "n_qubits": int,
      "pauli_terms": [[coeff, "PAULISTRING"], ...],
      "hf_energy": float,
      "fci_energy": float
    }

    Returns:
    {
      "n_qubits": int,
      "pauli_terms": List[(coeff, label)],
      "matrix": np.ndarray,
      "hf_energy": float,
      "fci_energy": float
    }
    """
    # Normalize molecule name -> filename
    if molecule == "HeH+":
        filename = "HeHp.json"
    else:
        filename = f"{molecule}.json"

    # repo_root/src/utils/standard_hamiltonians.py -> repo_root
    repo_root = Path(__file__).resolve().parents[2]
    ham_path = repo_root / "hamiltonians" / filename

    if not ham_path.exists():
        raise FileNotFoundError(
            f"Hamiltonian file not found:\n  {ham_path}\n"
            f"Expected files under:\n  {repo_root / 'hamiltonians'}"
        )

    raw = json.loads(ham_path.read_text())

    pauli_terms = [(float(c), str(p)) for c, p in raw["pauli_terms"]]
    matrix = _matrix_from_pauli_terms(pauli_terms)

    # FCI energy: load if present, otherwise compute
    if "fci_energy" in raw:
        fci_energy = float(raw["fci_energy"])
    else:
        fci_energy = float(np.min(np.linalg.eigvalsh(matrix)))

    return {
        "n_qubits": int(raw["n_qubits"]),
        "pauli_terms": pauli_terms,
        "matrix": matrix,
        "hf_energy": float(raw.get("hf_energy", np.nan)),
        "fci_energy": fci_energy,
    }
