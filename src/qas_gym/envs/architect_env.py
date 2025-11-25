import numpy as np
import cirq
from .qas_env import QuantumArchSearchEnv
# We must import the Saboteur class to access the static helper and error rates
from .saboteur_env import SaboteurMultiGateEnv

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
            
            # Guard clause for empty or invalid circuits
            if final_circuit is None or not final_circuit.all_operations():
                info['fidelity_under_attack'] = 0.0
                return obs, 0.0, terminated, truncated, info

            if self.saboteur_agent is not None:
                # 1. Generate the correct Dict observation using the static helper
                # This ensures keys ('projected_state', 'gate_structure') and shapes match exactly.
                sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(
                    final_circuit, 
                    n_qubits=len(self.qubits),
                    max_gates=self.max_timesteps # MUST match the Saboteur's max_gates
                )
        
                # 2. Get the Saboteur's action (indices for error rates)
                # This line was previously causing the indentation error
                sab_action, _ = self.saboteur_agent.predict(sab_obs, deterministic=True)

                # 3. Apply the noise
                ops = list(final_circuit.all_operations()) 
                noisy_ops = []
                all_rates = SaboteurMultiGateEnv.all_error_rates
                max_rate_idx = len(all_rates) - 1
                
                for i, op in enumerate(ops):
                    noisy_ops.append(op)
                    
                    # Safety check: Ensure we don't go out of bounds of the agent's output
                    if i < len(sab_action):
                        idx = int(sab_action[i])
                        # Clip index to be safe
                        idx = max(0, min(idx, max_rate_idx))
                        error_rate = all_rates[idx]
                        
                        # Apply depolarizing noise to all qubits in this gate
                        for q in op.qubits:
                            noisy_ops.append(cirq.DepolarizingChannel(error_rate).on(q))
                            
                noisy_circuit = cirq.Circuit(noisy_ops)
                
                # Calculate fidelity under attack
                fidelity_under_attack = self.get_fidelity(noisy_circuit)
                
                info['fidelity_under_attack'] = fidelity_under_attack
                
                # REWARD OVERRIDE: The Architect is now judged solely on robustness
                reward = fidelity_under_attack
            else:
                # If no saboteur, fallback to standard fidelity
                info['fidelity_under_attack'] = self.get_fidelity(final_circuit)

        return obs, reward, terminated, truncated, info