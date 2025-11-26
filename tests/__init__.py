"""
Tests package for Quantum Architecture Search.

See ExpPlan.md, Part 7 (Verification & smoke testing).

This package contains verification tests for Stage 7:
  - test_saboteur_efficacy.py: Stage 7.1 - Saboteur drops GHZ fidelity
  - test_vqe_physics.py: Stage 7.2 - VQE returns correct reference energies
  - test_parallelism_overhead.py: Stage 7.3 - DummyVecEnv vs SubprocVecEnv
"""
