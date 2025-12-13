import numpy as np
import cirq

from src.qas_gym.envs.qas_env import QuantumArchSearchEnv


def make_state_vec(n_qubits, ones=False):
    dim = 2 ** n_qubits
    vec = np.zeros((dim,), dtype=complex)
    if ones:
        vec[-1] = 1.0
    else:
        vec[0] = 1.0
    return vec


def test_stop_success_reward():
    target = make_state_vec(2, ones=False)  # |00> target
    env = QuantumArchSearchEnv(
        target=target,
        fidelity_threshold=0.9,
        reward_penalty=-0.1,
        max_timesteps=15,
        qubits=cirq.LineQubit.range(2),
        stop_success_bonus=0.42,
        stop_failure_penalty=-0.123,
        per_step_penalty=-0.01,
    )
    obs, info = env.reset()
    # Immediately choose STOP -> empty circuit has fidelity 1.0 -> success
    obs, reward, terminated, truncated, info = env.step(env.stop_action_index)
    assert terminated
    assert abs(reward - 0.42) < 1e-8


def test_stop_failure_reward():
    target = make_state_vec(2, ones=True)  # |11> target; empty circuit fidelity 0
    env = QuantumArchSearchEnv(
        target=target,
        fidelity_threshold=0.9,
        reward_penalty=-0.1,
        max_timesteps=15,
        qubits=cirq.LineQubit.range(2),
        stop_success_bonus=0.5,
        stop_failure_penalty=-0.321,
        per_step_penalty=-0.01,
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.stop_action_index)
    assert terminated
    assert abs(reward - (-0.321)) < 1e-8
