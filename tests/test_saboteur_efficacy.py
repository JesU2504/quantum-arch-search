"""
Stage 7.1 - Saboteur efficacy check.

See ExpPlan.md, Part 7.1:
  - Test: Load a perfect GHZ circuit. Let the Saboteur act for 1 step
    with max budget.
  - Verify: Fidelity must drop significantly below 1.0.
  - Why: If fidelity stays 1.0, noise injection is broken and the
    Architect trains against a phantom.

This test validates that the Saboteur environment correctly injects
noise into quantum circuits.

TODO: Implement full test once Saboteur class is complete.
"""

import numpy as np


def test_saboteur_drops_ghz_fidelity():
    """
    Test that Saboteur reduces GHZ circuit fidelity.

    Steps:
    1. Create a perfect GHZ circuit
    2. Compute initial fidelity (should be ~1.0)
    3. Apply Saboteur attack with maximum budget
    4. Verify fidelity drops significantly (< 0.95)

    This ensures the noise injection mechanism is working correctly.
    """
    # TODO: Import Saboteur and GHZ utilities
    # TODO: Create perfect GHZ circuit for n_qubits=4
    # TODO: Compute initial fidelity against target GHZ state
    # TODO: Create Saboteur environment with max attack budget
    # TODO: Take one step with maximum error rates
    # TODO: Assert fidelity dropped below threshold
    pass


def test_saboteur_respects_attack_budget():
    """
    Test that Saboteur respects the max_concurrent_attacks budget.

    The Saboteur should only attack a limited number of gates per step,
    even if the action requests attacks on more gates.
    """
    # TODO: Create circuit with many gates
    # TODO: Create Saboteur with small attack budget
    # TODO: Request attacks on all gates
    # TODO: Verify only budget-limited attacks applied
    pass


def test_saboteur_zero_noise_preserves_fidelity():
    """
    Test that zero-noise action preserves circuit fidelity.

    When the Saboteur chooses error_rate=0.0 for all gates,
    the circuit fidelity should remain unchanged.
    """
    # TODO: Create perfect GHZ circuit
    # TODO: Apply Saboteur action with all zeros (no noise)
    # TODO: Verify fidelity remains ~1.0
    pass


def test_saboteur_observation_shape():
    """
    Test that Saboteur produces correct observation shape.

    The observation should be a dict with:
      - 'projected_state': shape (2 * n_qubits,)
      - 'gate_structure': shape (max_gates,)
    """
    # TODO: Create Saboteur environment
    # TODO: Reset and get observation
    # TODO: Verify observation structure and shapes
    pass


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("Running Stage 7.1 tests...")
    test_saboteur_drops_ghz_fidelity()
    test_saboteur_respects_attack_budget()
    test_saboteur_zero_noise_preserves_fidelity()
    test_saboteur_observation_shape()
    print("All Stage 7.1 tests passed (or skipped with TODO).")
