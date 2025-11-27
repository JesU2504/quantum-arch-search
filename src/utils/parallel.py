"""
Parallelism utilities for Quantum Architecture Search.

See ExpPlan.md, Implementation notes:
  - Parallelization: Use SubprocVecEnv in Stable-Baselines3 to utilize
    all CPU cores (essential for 5-10 seeds)

See ExpPlan.md, Part 7.3 (Parallelism overhead check):
  - Test: Run 1000 steps with DummyVecEnv (serial) vs SubprocVecEnv (parallel, n_envs=4)
  - Verify: Parallel FPS ≈ 3× serial (accounting for overhead)
  - Why: If parallel is slower, env serialization is too heavy
    (common with Cirq objects)
"""

import time
from typing import Callable, Dict, Optional

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def create_vec_env(
    env_fn: Callable,
    n_envs: int = 4,
    use_subproc: bool = True,
    **kwargs
):
    """
    Create a vectorized environment for parallel training.

    Args:
        env_fn: Factory function that creates an environment.
        n_envs: Number of parallel environments.
        use_subproc: If True, use SubprocVecEnv; else DummyVecEnv.
        **kwargs: Additional arguments for the vec env.

    Returns:
        A vectorized environment (SubprocVecEnv or DummyVecEnv).
    """
    env_fns = [env_fn for _ in range(n_envs)]
    if use_subproc:
        return SubprocVecEnv(env_fns, **kwargs)
    else:
        return DummyVecEnv(env_fns)


def benchmark_parallelism(
    env_fn: Callable,
    n_steps: int = 1000,
    n_envs: int = 4,
) -> Dict[str, float]:
    """
    Benchmark serial vs parallel environment performance.

    See ExpPlan.md, Part 7.3 for expected results.

    Args:
        env_fn: Factory function that creates an environment.
        n_steps: Number of steps to benchmark.
        n_envs: Number of parallel environments.

    Returns:
        Dict with benchmark results:
          - serial_fps: Frames per second with DummyVecEnv
          - parallel_fps: Frames per second with SubprocVecEnv
          - speedup: parallel_fps / serial_fps
    """
    # Benchmark serial (DummyVecEnv)
    serial_env = create_vec_env(env_fn, n_envs=n_envs, use_subproc=False)
    serial_fps = measure_fps(serial_env, n_steps)
    serial_env.close()
    
    # Benchmark parallel (SubprocVecEnv)
    parallel_env = create_vec_env(env_fn, n_envs=n_envs, use_subproc=True)
    parallel_fps = measure_fps(parallel_env, n_steps)
    parallel_env.close()
    
    speedup = parallel_fps / serial_fps if serial_fps > 0 else 0.0
    
    return {
        "serial_fps": serial_fps,
        "parallel_fps": parallel_fps,
        "speedup": speedup,
    }


def measure_fps(vec_env, n_steps: int) -> float:
    """
    Measure frames per second for a vectorized environment.

    Args:
        vec_env: The vectorized environment.
        n_steps: Number of steps to run.

    Returns:
        Frames per second.
    """
    import numpy as np
    
    vec_env.reset()
    n_envs = vec_env.num_envs
    
    start_time = time.perf_counter()
    for _ in range(n_steps):
        # Sample actions for all environments at once using numpy
        actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
        vec_env.step(actions)
    elapsed = time.perf_counter() - start_time
    
    total_frames = n_steps * n_envs
    return total_frames / elapsed


class PickleSafeEnvWrapper:
    """
    Wrapper to make environments pickle-safe for SubprocVecEnv.

    Some environments (especially with Cirq objects) may have
    serialization issues. This wrapper provides a pickle-safe
    interface.

    See Implementation notes in ExpPlan.md regarding Cirq object
    serialization overhead.
    """

    def __init__(self, env_fn: Callable):
        """
        Initialize the wrapper.

        Args:
            env_fn: Factory function for the environment.
        """
        self.env_fn = env_fn
        self.env = None

    def __call__(self):
        """Create and return the environment."""
        if self.env is None:
            self.env = self.env_fn()
        return self.env
