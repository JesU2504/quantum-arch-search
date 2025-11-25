import cirq
import numpy as np
from qas_gym.envs.qas_env import QuantumArchSearchEnv
from qas_gym.utils import *


class NoisyNQubitEnv(QuantumArchSearchEnv):
    def __init__(
        self,
        target: np.ndarray,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        n_qubits = int(np.log2(len(target)))
        qubits = cirq.LineQubit.range(n_qubits)
        state_observables = get_default_observables(qubits)
        action_gates = get_default_gates(qubits)
        self.error_rate = error_rate
        super().__init__(target=target,
                         fidelity_threshold=fidelity_threshold,
                         reward_penalty=reward_penalty,
                         max_timesteps=max_timesteps,
                         qubits=qubits,
                         state_observables=state_observables,
                         action_gates=action_gates)

    def _get_obs(self):
        circuit = self._get_cirq().with_noise(cirq.depolarize(self.error_rate))
        return super()._get_obs_from_circuit(circuit)

    def get_fidelity(self, circuit):
        noisy_circuit = circuit.with_noise(cirq.depolarize(self.error_rate))
        return super().get_fidelity(noisy_circuit)


class NoisyTwoQubitEnv(NoisyNQubitEnv):
    def __init__(
        self,
        target: np.ndarray = get_bell_state(),
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        assert len(target) == 4, 'Target must be of size 4'
        super().__init__(target=target,
                         fidelity_threshold=fidelity_threshold,
                         reward_penalty=reward_penalty,
                         max_timesteps=max_timesteps,
                         error_rate=error_rate)


class NoisyThreeQubitEnv(NoisyNQubitEnv):
    def __init__(
        self,
        target: np.ndarray = get_ghz_state(3),
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        assert len(target) == 8, 'Target must be of size 8'
        super().__init__(target=target,
                         fidelity_threshold=fidelity_threshold,
                         reward_penalty=reward_penalty,
                         max_timesteps=max_timesteps,
                         error_rate=error_rate)
