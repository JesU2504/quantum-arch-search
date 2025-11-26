"""
Stage 7.3 - Parallelism overhead check.

See ExpPlan.md, Part 7.3:
  - Test: Run 1000 steps with DummyVecEnv (serial) vs SubprocVecEnv
    (parallel, n_envs=4).
  - Verify: Parallel FPS ≈ 3× serial (accounting for overhead).
  - Why: If parallel is slower, env serialization is too heavy
    (common with Cirq objects).

This test validates that parallel training is beneficial for the
quantum architecture search environments.

TODO: Implement full test once parallel utilities are complete.
"""

import time


# Expected minimum speedup for parallel to be beneficial
MIN_EXPECTED_SPEEDUP = 2.0  # Parallel should be at least 2x faster
N_STEPS = 1000
N_ENVS = 4


def test_parallel_is_faster_than_serial():
    """
    Test that SubprocVecEnv is faster than DummyVecEnv.

    Runs 1000 steps with both serial and parallel environments
    and verifies that parallel achieves meaningful speedup.
    """
    # TODO: Import create_vec_env and benchmark utilities
    # TODO: Define environment factory function
    # TODO: Measure serial (DummyVecEnv) FPS
    # TODO: Measure parallel (SubprocVecEnv) FPS
    # TODO: Assert speedup >= MIN_EXPECTED_SPEEDUP
    pass


def test_serialization_overhead():
    """
    Test that environment serialization is not prohibitive.

    Cirq objects can have heavy serialization overhead which
    may make SubprocVecEnv slower than serial execution.
    """
    # TODO: Create environment with Cirq circuits
    # TODO: Measure time to pickle/unpickle environment
    # TODO: Verify overhead is acceptable (< 10ms per call)
    pass


def test_parallel_results_consistency():
    """
    Test that parallel and serial produce consistent results.

    Both execution modes should produce statistically similar
    rewards and fidelities over many episodes.
    """
    # TODO: Run N episodes with DummyVecEnv
    # TODO: Run N episodes with SubprocVecEnv
    # TODO: Compare mean rewards (should be similar)
    pass


def test_benchmark_reports_fps():
    """
    Test that benchmark_parallelism returns correct metrics.

    The benchmark function should return:
      - serial_fps: Frames per second with DummyVecEnv
      - parallel_fps: Frames per second with SubprocVecEnv
      - speedup: parallel_fps / serial_fps
    """
    # TODO: Import benchmark_parallelism
    # TODO: Run benchmark
    # TODO: Verify all expected keys in result
    # TODO: Verify speedup = parallel_fps / serial_fps
    pass


def measure_fps_manual(env, n_steps):
    """
    Helper function to manually measure FPS.

    Args:
        env: The environment (or vec_env) to benchmark.
        n_steps: Number of steps to run.

    Returns:
        Frames per second.
    """
    # TODO: Reset environment
    # TODO: Start timer
    # TODO: Run n_steps with random actions
    # TODO: Calculate and return FPS
    pass


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("Running Stage 7.3 tests...")
    test_parallel_is_faster_than_serial()
    test_serialization_overhead()
    test_parallel_results_consistency()
    test_benchmark_reports_fps()
    print("All Stage 7.3 tests passed (or skipped with TODO).")
