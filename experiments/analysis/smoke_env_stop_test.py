"""Smoke test to verify STOP action works and episodes can end early.

Usage: run from repository root.
"""
import sys
from math import sqrt
import numpy as np

# Make repo importable
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qas_gym.envs.qas_env import QuantumArchSearchEnv
import cirq


def make_target_state(n_qubits):
    # Simple |00...0> target
    dim = 2 ** n_qubits
    vec = np.zeros((dim,), dtype=complex)
    vec[0] = 1.0
    return vec


def run_once(n_qubits=3, max_timesteps=15, seed=0):
    target = make_target_state(n_qubits)
    env = QuantumArchSearchEnv(target=target,
                               fidelity_threshold=0.9999,
                               reward_penalty=-0.1,
                               max_timesteps=max_timesteps,
                               qubits=cirq.LineQubit.range(n_qubits))
    obs, info = env.reset()
    print('Action space size:', env.action_space.n, 'STOP index:', env.stop_action_index)

    rng = np.random.default_rng(seed)
    for step in range(30):
        # With some probability choose the STOP action to test early stop
        if rng.random() < 0.3:
            action = env.stop_action_index
        else:
            action = rng.integers(low=0, high=env.stop_action_index)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step={step}, action={action}, reward={reward:.4f}, terminated={terminated}, total_gates={info.get('total_gates')}, ops={info.get('operation_count')}")
        if terminated:
            print('Terminated reason info:', info.get('stop_reason', 'fidelity_or_budget'))
            break


if __name__ == '__main__':
    run_once()
