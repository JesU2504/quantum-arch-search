"""
Utility to generate standard molecular Hamiltonians (H2, HeH+, LiH, BeH2)
using Qiskit Nature + PySCF with reasonable active-space reductions.

Returns both Pauli-sum terms and full matrix for downstream consumers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer


STANDARD_GEOM = {
    "H2": {
        "atom": "H 0 0 0; H 0 0 0.735",
        "charge": 0,
        "spin": 0,
        "basis": "sto3g",
        "transformers": [],
    },
    "HeH+": {
        # Bond length ~1.4632 Å from common benchmarks
        "atom": "H 0 0 0; He 0 0 1.4632",
        "charge": 1,
        "spin": 0,
        "basis": "sto3g",
        "transformers": [],
    },
    "LiH": {
        # Standard geometry ~1.6 Å; freeze core + 2e,2o active space
        "atom": "Li 0 0 0; H 0 0 1.6",
        "charge": 0,
        "spin": 0,
        "basis": "sto3g",
        "transformers": [
            FreezeCoreTransformer(),
            ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2),
        ],
    },
    "BeH2": {
        # Linear BeH2: H at ±1.326 Å; freeze core + 4e,4o active space
        "atom": "Be 0 0 0; H 0 0 -1.326; H 0 0 1.326",
        "charge": 0,
        "spin": 0,
        "basis": "sto3g",
        "transformers": [
            FreezeCoreTransformer(),
            ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=4),
        ],
    },
}


def _pauli_terms_from_sparse_op(op) -> List[Tuple[float, str]]:
    """Convert a SparsePauliOp to (coeff, label) list with real coefficients."""
    terms: List[Tuple[float, str]] = []
    for label, coeff in zip(op.paulis, op.coeffs):
        terms.append((float(np.real(coeff)), label.to_label()))
    return terms


@lru_cache(maxsize=None)
def get_standard_hamiltonian(molecule: str) -> Dict[str, object]:
    """
    Build a standard molecular Hamiltonian for the given molecule.

    Returns dict with:
      - n_qubits: int
      - pauli_terms: List[(coeff, pauli_str)]
      - matrix: np.ndarray
      - hf_energy: float
      - fci_energy: float (exact ground energy from matrix diag)
    """
    if molecule not in STANDARD_GEOM:
        raise ValueError(f"Unsupported molecule {molecule}. Supported: {', '.join(STANDARD_GEOM.keys())}")

    spec = STANDARD_GEOM[molecule]
    problem = PySCFDriver(
        atom=spec["atom"],
        charge=spec["charge"],
        spin=spec["spin"],
        basis=spec["basis"],
    ).run()

    # Apply transformers (freeze core / active space)
    for transformer in spec["transformers"]:
        problem = transformer.transform(problem)

    mapper = ParityMapper()
    tapered_mapper = problem.get_tapered_mapper(mapper)
    qubit_op = tapered_mapper.map(problem.hamiltonian.second_q_op())

    # Add nuclear repulsion so returned energies are total energies
    nuc = getattr(problem.hamiltonian, "nuclear_repulsion_energy", 0.0)
    if abs(nuc) > 0:
        from qiskit.quantum_info import SparsePauliOp

        shift = SparsePauliOp.from_list([("I" * qubit_op.num_qubits, nuc)])
        qubit_op = qubit_op + shift

    matrix = qubit_op.to_matrix()
    eigvals = np.linalg.eigvalsh(matrix)

    return {
        "n_qubits": qubit_op.num_qubits,
        "pauli_terms": _pauli_terms_from_sparse_op(qubit_op),
        "matrix": matrix,
        "hf_energy": float(problem.reference_energy),
        "fci_energy": float(np.min(eigvals)),
    }
