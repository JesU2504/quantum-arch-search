"""
Stage 7.2 - VQE physics check.

See ExpPlan.md, Part 7.2:
  - Test: Initialize VQEArchitectEnv with a dummy (identity) circuit.
  - Verify: Energy matches Hartree-Fock (≈ -1.117 Ha for H2 at eq.)
    or a random-state energy, but not the exact FCI ground state yet.
  - Why: Validates Hamiltonian mapping and expectation computation.

This test validates the VQE environment correctly computes molecular
energies using quantum circuits.

TODO: Implement full test once VQEArchitectEnv class is complete.
"""

import numpy as np


# Reference energies for H2 at equilibrium (~0.74 Å)
H2_HARTREE_FOCK_ENERGY = -1.117  # Ha (approximate)
H2_FCI_ENERGY = -1.137  # Ha (approximate)
CHEMICAL_ACCURACY = 0.0016  # 1.6 mHa


def test_vqe_returns_reasonable_energy():
    """
    Test that VQEArchitectEnv returns physically reasonable energy.

    For an identity (empty) circuit, the energy should be close to
    the Hartree-Fock energy, not the exact ground state.
    """
    # TODO: Import VQEArchitectEnv
    # TODO: Create environment for H2 molecule
    # TODO: Initialize with empty/identity circuit
    # TODO: Compute energy
    # TODO: Assert energy is above FCI (not exact ground state)
    # TODO: Assert energy is close to Hartree-Fock
    pass


def test_vqe_energy_not_exact_ground_state():
    """
    Test that dummy circuit doesn't achieve exact ground state.

    An identity circuit should not achieve the FCI ground state energy.
    This verifies the Hamiltonian is non-trivial.
    """
    # TODO: Create VQEArchitectEnv with dummy circuit
    # TODO: Compute energy
    # TODO: Assert |energy - FCI| > CHEMICAL_ACCURACY
    pass


def test_vqe_hamiltonian_construction():
    """
    Test that Hamiltonian is correctly constructed.

    The molecular Hamiltonian should be Hermitian and have
    the expected number of terms.
    """
    # TODO: Create VQEArchitectEnv
    # TODO: Access the constructed Hamiltonian
    # TODO: Verify Hermiticity
    # TODO: Verify term count is reasonable
    pass


def test_vqe_reference_energies():
    """
    Test that reference energies are provided correctly.

    The environment should provide both Hartree-Fock and FCI
    reference energies for comparison.
    """
    # TODO: Create VQEArchitectEnv
    # TODO: Get reference energies
    # TODO: Verify both HF and FCI energies are present
    # TODO: Verify FCI < HF (ground state is lower)
    pass


def test_vqe_circuit_improves_energy():
    """
    Test that adding gates can improve energy.

    A well-chosen circuit should achieve lower energy than
    the identity circuit.
    """
    # TODO: Create VQEArchitectEnv
    # TODO: Compute identity circuit energy
    # TODO: Apply some gates (e.g., parameterized ansatz)
    # TODO: Verify energy decreased
    pass


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("Running Stage 7.2 tests...")
    test_vqe_returns_reasonable_energy()
    test_vqe_energy_not_exact_ground_state()
    test_vqe_hamiltonian_construction()
    test_vqe_reference_energies()
    test_vqe_circuit_improves_energy()
    print("All Stage 7.2 tests passed (or skipped with TODO).")
