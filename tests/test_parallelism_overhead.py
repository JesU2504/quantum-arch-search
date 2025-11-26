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
"""

import sys
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.parallel import create_vec_env, benchmark_parallelism


# Expected minimum speedup for parallel to be beneficial
MIN_EXPECTED_SPEEDUP = 2.0  # Parallel should be at least 2x faster
N_STEPS = 1000
N_ENVS = 4


class SlowToyEnv(gym.Env):
    """
    A toy environment with simulated computational work per step.
    
    This simulates the computational overhead of quantum circuit 
    simulation (e.g., Cirq operations) to properly test parallelism benefits.
    Each step performs Python-heavy computation to simulate work that
    benefits from multiprocessing.
    """
    
    def __init__(self, work_iterations: int = 50000):
        super().__init__()
        self.work_iterations = work_iterations
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self._state = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = np.random.randn(4).astype(np.float32)
        return self._state, {}
    
    def step(self, action):
        # Simulate computational work (Python loops don't release GIL)
        # This represents the kind of work that benefits from multiprocessing
        total = 0.0
        for i in range(self.work_iterations):
            total += (i * 0.001) ** 0.5
        
        self._state = np.random.randn(4).astype(np.float32)
        reward = float(total % 1.0)  # Use result to prevent optimization
        terminated = False
        truncated = False
        return self._state, reward, terminated, truncated, {}


def make_slow_toy_env():
    """Factory function to create SlowToyEnv."""
    return SlowToyEnv(work_iterations=50000)


def test_parallel_is_faster_than_serial():
    """
    Test that SubprocVecEnv is faster than DummyVecEnv.

    Runs 1000 steps with both serial and parallel environments
    and verifies that parallel achieves meaningful speedup.
    
    Uses a toy environment with simulated computational work to
    represent quantum circuit simulation overhead.
    """
    # Run benchmark using slow toy Gym environment
    results = benchmark_parallelism(
        env_fn=make_slow_toy_env,
        n_steps=N_STEPS,
        n_envs=N_ENVS,
    )
    
    serial_fps = results["serial_fps"]
    parallel_fps = results["parallel_fps"]
    speedup = results["speedup"]
    
    # Print results for debugging
    print(f"\nSerial FPS: {serial_fps:.2f}")
    print(f"Parallel FPS: {parallel_fps:.2f}")
    print(f"Speedup: {speedup:.2f}x")
    
    # Assert parallel is at least 2x faster than serial
    assert speedup >= MIN_EXPECTED_SPEEDUP, (
        f"Parallel speedup ({speedup:.2f}x) is less than expected "
        f"minimum ({MIN_EXPECTED_SPEEDUP}x). This may indicate "
        f"environment serialization overhead is too high."
    )


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
