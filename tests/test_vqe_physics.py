"""
Stage 7.2 - VQE physics check.

See ExpPlan.md, Part 7.2:
  - Test: Initialize VQEArchitectEnv with a dummy (identity) circuit.
  - Verify: Energy matches Hartree-Fock (≈ -1.117 Ha for H2 at eq.)
    or a random-state energy, but not the exact FCI ground state yet.
  - Why: Validates Hamiltonian mapping and expectation computation.

This test validates the VQE environment correctly computes molecular
energies using quantum circuits.
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
    from src.envs import VQEArchitectEnv

    # Create environment for H2 molecule
    env = VQEArchitectEnv(molecule="H2", bond_distance=0.74)

    # Compute energy with identity circuit (None means |00> state)
    energy = env.compute_energy(circuit=None)

    # Assert energy is above FCI (not exact ground state)
    # The |00> state should not achieve the exact ground state energy
    assert energy > H2_FCI_ENERGY, (
        f"Energy {energy:.4f} Ha should be above FCI {H2_FCI_ENERGY:.4f} Ha"
    )

    # Assert energy is reasonably close to Hartree-Fock
    # The |00> state represents the HF reference for our H2 Hamiltonian
    # We use a tolerance of 0.01 Ha (10 mHa) to account for small numerical
    # differences. This is much tighter than the 0.02 Ha gap between HF and FCI.
    hf_tolerance = 0.01  # Ha (10 mHa)
    assert abs(energy - H2_HARTREE_FOCK_ENERGY) < hf_tolerance, (
        f"Energy {energy:.4f} Ha should be within {hf_tolerance} Ha "
        f"of Hartree-Fock {H2_HARTREE_FOCK_ENERGY:.4f} Ha"
    )


def test_vqe_energy_not_exact_ground_state():
    """
    Test that dummy circuit doesn't achieve exact ground state.

    An identity circuit should not achieve the FCI ground state energy.
    This verifies the Hamiltonian is non-trivial.
    """
    from src.envs import VQEArchitectEnv

    # Create environment for H2 molecule
    env = VQEArchitectEnv(molecule="H2", bond_distance=0.74)

    # Compute energy with identity circuit (None means |00> state)
    energy = env.compute_energy(circuit=None)

    # Assert |energy - FCI| > CHEMICAL_ACCURACY
    # The identity circuit (|00> state) should NOT achieve the FCI ground state
    assert abs(energy - H2_FCI_ENERGY) > CHEMICAL_ACCURACY, (
        f"Identity circuit energy {energy:.4f} Ha is too close to FCI "
        f"{H2_FCI_ENERGY:.4f} Ha (within {CHEMICAL_ACCURACY} Ha chemical accuracy)"
    )


def test_vqe_hamiltonian_construction():
    """
    Test that Hamiltonian is correctly constructed.

    The molecular Hamiltonian should be Hermitian and have
    the expected number of terms.
    """
    from src.envs import VQEArchitectEnv

    # Create environment for H2 molecule
    env = VQEArchitectEnv(molecule="H2", bond_distance=0.74)

    # Access the constructed Hamiltonian
    H = env.hamiltonian

    # Verify Hamiltonian is a 2D matrix (4x4 for 2 qubits)
    assert H.ndim == 2, f"Hamiltonian should be 2D, got {H.ndim}D"
    expected_dim = 2 ** env.n_qubits
    assert H.shape == (expected_dim, expected_dim), (
        f"Hamiltonian shape should be ({expected_dim}, {expected_dim}), "
        f"got {H.shape}"
    )

    # Verify Hermiticity: H = H†
    H_dagger = np.conj(H.T)
    assert np.allclose(H, H_dagger), (
        "Hamiltonian must be Hermitian (H = H†)"
    )

    # Verify the Hamiltonian has real eigenvalues (consequence of Hermiticity)
    eigenvalues = np.linalg.eigvalsh(H)
    assert np.all(np.isreal(eigenvalues)), (
        "Hamiltonian eigenvalues must be real"
    )


def test_vqe_reference_energies():
    """
    Test that reference energies are provided correctly.

    The environment should provide both Hartree-Fock and FCI
    reference energies for comparison.
    """
    from src.envs import VQEArchitectEnv

    # Create environment for H2 molecule
    env = VQEArchitectEnv(molecule="H2", bond_distance=0.74)

    # Get reference energies
    ref_energies = env.get_reference_energies()

    # Verify both HF and FCI energies are present
    assert "hartree_fock" in ref_energies, (
        "Reference energies must include 'hartree_fock'"
    )
    assert "fci" in ref_energies, (
        "Reference energies must include 'fci'"
    )

    hf_energy = ref_energies["hartree_fock"]
    fci_energy = ref_energies["fci"]

    # Verify energies are finite numbers
    assert np.isfinite(hf_energy), "Hartree-Fock energy must be finite"
    assert np.isfinite(fci_energy), "FCI energy must be finite"

    # Verify FCI < HF (ground state is lower than Hartree-Fock)
    assert fci_energy < hf_energy, (
        f"FCI energy {fci_energy:.4f} Ha must be lower than "
        f"Hartree-Fock {hf_energy:.4f} Ha"
    )


def test_vqe_circuit_improves_energy():
    """
    Test that adding gates can improve energy.

    A well-chosen circuit should achieve lower energy than
    the identity circuit.
    """
    import cirq
    from src.envs import VQEArchitectEnv

    # Create environment for H2 molecule
    env = VQEArchitectEnv(molecule="H2", bond_distance=0.74)

    # Compute identity circuit energy
    identity_energy = env.compute_energy(circuit=None)

    # Create an ansatz circuit that can reach the ground state.
    # For H2, the ground state is (|01> - |10>)/sqrt(2), which requires
    # specific entangling operations to achieve from the |00> reference.
    qubits = cirq.LineQubit.range(env.n_qubits)
    ansatz = cirq.Circuit([
        cirq.H(qubits[0]),           # Create superposition on qubit 0
        cirq.X(qubits[1]),           # Flip qubit 1
        cirq.CZ(qubits[0], qubits[1]),  # Apply controlled-Z for phase
        cirq.CNOT(qubits[0], qubits[1]),  # Entangle qubits
    ])

    # Compute energy with the ansatz circuit
    ansatz_energy = env.compute_energy(circuit=ansatz)

    # Verify energy decreased - the ansatz should achieve lower energy
    # than the identity circuit (which gives the Hartree-Fock energy)
    assert ansatz_energy < identity_energy, (
        f"Ansatz energy {ansatz_energy:.4f} Ha should be lower than "
        f"identity energy {identity_energy:.4f} Ha"
    )


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("Running Stage 7.2 tests...")
    test_vqe_returns_reasonable_energy()
    test_vqe_energy_not_exact_ground_state()
    test_vqe_hamiltonian_construction()
    test_vqe_reference_energies()
    test_vqe_circuit_improves_energy()
    print("All Stage 7.2 tests passed.")
