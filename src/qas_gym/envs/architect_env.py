import numpy as np
import cirq
from .qas_env import QuantumArchSearchEnv

class ArchitectEnv(QuantumArchSearchEnv):
    """
    An environment for the architect agent that uses a sophisticated reward function
    based on the paper "QML Architecture Search via Deep Reinforcement Learning".
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_fidelity = 0.0

    def reset(self, seed=None, options=None):
        self.previous_fidelity = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # Get the result from the parent environment
        observation, _, terminated, truncated, info = super().step(action)
        
        fidelity = info.get('fidelity', 0.0)
        num_gates = len(self.circuit_gates)
        
        y_l = fidelity
        y_target = self.fidelity_threshold
        l = num_gates
        L = self.max_timesteps

        if y_l >= y_target and l < L:
            # Success condition
            reward = 0.2 * (y_l / y_target) * (L - l)
        elif l == L and y_l < y_target:
            # Failure condition
            reward = -0.2 * ((y_target - y_l) / y_target) * l
        else:
            # Shaping condition
            shaped_reward = (y_l - self.previous_fidelity) / (self.previous_fidelity + 1e-6) - 0.01 * l
            reward = np.clip(shaped_reward, -1.5, 1.5)

        self.previous_fidelity = fidelity

        return observation, reward, terminated, truncated, info


class AdversarialArchitectEnv(ArchitectEnv):
    """Evaluate robustness by applying a multi-gate noise vector from a saboteur agent.

    Differences vs previous version:
      * Attacks the final episode circuit (not a possibly stale champion snapshot).
      * Uses multi-gate action emitted by SaboteurMultiGateEnv (vector of indices).
      * Terminal reward replaced by fidelity under attack for credit alignment.
    """

    def __init__(self, saboteur_agent=None, saboteur_max_error_level=None, **kwargs):
        super().__init__(**kwargs)
        self.saboteur_agent = saboteur_agent
        self.saboteur_max_error_level = saboteur_max_error_level

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated:
            # Use final circuit constructed this episode
            final_circuit = self._get_cirq()
            if final_circuit is None or not final_circuit.all_operations():
                info['fidelity_under_attack'] = 0.0
                return obs, 0.0, terminated, truncated, info

            if self.saboteur_agent is not None:
                from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv

                ops = list(final_circuit.all_operations())
                sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(final_circuit, n_qubits=len(self.qubits))
                try:
                    sab_action, _ = self.saboteur_agent.predict(sab_obs, deterministic=True)
                except Exception:
                    import numpy as _np
                    sab_action = _np.zeros(len(ops), dtype=int)

                # Build noisy circuit by appending depolarizing channels per gate
                noisy_ops = []
                all_rates = SaboteurMultiGateEnv.all_error_rates
                max_idx = len(all_rates) - 1
                for i, op in enumerate(ops):
                    noisy_ops.append(op)
                    idx = int(sab_action[i]) if i < len(sab_action) else 0
                    idx = max(0, min(idx, max_idx))
                    error_rate = all_rates[idx]
                    for q in op.qubits:
                        noisy_ops.append(cirq.DepolarizingChannel(error_rate).on(q))
                noisy_circuit = cirq.Circuit(noisy_ops)
                fidelity_under_attack = self.get_fidelity(noisy_circuit)
                info['fidelity_under_attack'] = fidelity_under_attack
                reward = fidelity_under_attack
            else:
                info['fidelity_under_attack'] = self.get_fidelity(final_circuit)

        return obs, reward, terminated, truncated, info
