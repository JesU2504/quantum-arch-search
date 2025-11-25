import cirq
import numpy as np
from typing import Optional
from qas_gym.envs.qas_env import QuantumArchSearchEnv
from qas_gym.utils import *



class BasicNQubitEnv(QuantumArchSearchEnv):
    def __init__(self,
                 target: np.ndarray,
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        # Derive qubits and default resources
        n_qubits = int(np.log2(len(target)))
        qubits = self.get_qubits(n_qubits)
        state_observables = get_default_observables(qubits)
        action_gates = get_default_gates(qubits)
        # Correct argument ordering by using keywords to avoid silent mis-binding
        super().__init__(target=target,
                         fidelity_threshold=fidelity_threshold,
                         reward_penalty=reward_penalty,
                         max_timesteps=max_timesteps,
                         qubits=qubits,
                         state_observables=state_observables,
                         action_gates=action_gates)

    def get_qubits(self, n_qubits):
        return cirq.LineQubit.range(n_qubits)


class BasicTwoQubitEnv(BasicNQubitEnv):
    def __init__(self,
                 target: Optional[np.ndarray] = None,
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        target = self.get_target_state(target)
        super().__init__(target=target,
                         fidelity_threshold=fidelity_threshold,
                         reward_penalty=reward_penalty,
                         max_timesteps=max_timesteps)

    def get_target_state(self, target):
        if target is None:
            return get_bell_state()
        assert len(target) == 4, 'Target must be of size 4'
        return target


class BasicThreeQubitEnv(BasicNQubitEnv):
    def __init__(self,
                 target: Optional[np.ndarray] = None,
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        target = self.get_target_state(target)
        super().__init__(target=target,
                         fidelity_threshold=fidelity_threshold,
                         reward_penalty=reward_penalty,
                         max_timesteps=max_timesteps)

    def get_target_state(self, target):
        if target is None:
            return get_ghz_state(3)
        assert len(target) == 8, 'Target must be of size 8'
        return target
