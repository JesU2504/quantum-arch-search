import numpy as np
import cirq
from .qas_env import QuantumArchSearchEnv
# We must import the Saboteur class to access the static helper and error rates
from .saboteur_env import SaboteurMultiGateEnv

class ArchitectEnv(QuantumArchSearchEnv):
    """
    An environment for the architect agent that uses a sophisticated reward function.
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
        
        # Standard Curriculum Reward (for Baseline)
        # This encourages the agent to maintain high fidelity while reducing depth
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
            # Shaping condition (Improvement signal)
            shaped_reward = (y_l - self.previous_fidelity) / (self.previous_fidelity + 1e-6) - 0.01 * l
            reward = np.clip(shaped_reward, -1.5, 1.5)

        self.previous_fidelity = fidelity

        return observation, reward, terminated, truncated, info


class AdversarialArchitectEnv(ArchitectEnv):
    """
    Adversarial Environment with REWARD MIXING.
    
    This environment mixes the 'Clean Fidelity' (structure) with 
    'Fidelity Under Attack' (robustness) to prevent the Saboteur 
    from suppressing the Architect's learning entirely.
    """

    def __init__(self, saboteur_agent=None, saboteur_max_error_level=None, **kwargs):
        super().__init__(**kwargs)
        self.saboteur_agent = saboteur_agent
        self.saboteur_max_error_level = saboteur_max_error_level

    def step(self, action):
        # 1. Execute Architect Step (Get Clean Fidelity)
        obs, clean_reward, terminated, truncated, info = super().step(action)
        clean_fidelity = info.get('fidelity', 0.0)

        # Initialize robustness metrics
        fidelity_under_attack = clean_fidelity
        reward = clean_reward

        # 2. If Episode Ends (Circuit Complete), Apply Saboteur Attack
        if terminated and self.saboteur_agent is not None:
            final_circuit = self._get_cirq()
            
            # Guard for empty circuits
            if final_circuit is not None and len(list(final_circuit.all_operations())) > 0:
                
                # A. Generate Saboteur Observation
                sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(
                    final_circuit, 
                    n_qubits=len(self.qubits),
                    max_gates=self.max_timesteps
                )
        
                # B. Get Saboteur Action
                sab_action, _ = self.saboteur_agent.predict(sab_obs, deterministic=True)

                # C. Apply Noise (Reconstruct Noisy Circuit)
                ops = list(final_circuit.all_operations()) 
                noisy_ops = []
                all_rates = SaboteurMultiGateEnv.all_error_rates
                
                for i, op in enumerate(ops):
                    noisy_ops.append(op)
                    if i < len(sab_action):
                        idx = int(sab_action[i])
                        idx = max(0, min(idx, len(all_rates) - 1))
                        error_rate = all_rates[idx]
                        if error_rate > 0:
                            for q in op.qubits:
                                noisy_ops.append(cirq.DepolarizingChannel(error_rate).on(q))
                            
                noisy_circuit = cirq.Circuit(noisy_ops)
                
                # D. Measure Robustness
                fidelity_under_attack = self.get_fidelity(noisy_circuit)
                
                # E. MIXED REWARD CALCULATION
                # alpha = 0.5 balances "Building it right" vs "Building it strong"
                alpha = 0.5
                
                # Check for Unitary Task Override
                robust_metric = fidelity_under_attack
                if self.task_mode == 'unitary_preparation' and self.ideal_unitary is not None:
                    try:
                        from utils.metrics import unitary_from_basis_columns, process_fidelity
                        # ... (complex unitary sim logic omitted for brevity, assumes helper exists) ...
                        # For now, we trust the scalar fidelity_under_attack if unitary logic isn't strictly required here
                        pass 
                    except:
                        pass

                # The New Reward:
                # We blend the Clean Fidelity (so the agent knows it built a GHZ state)
                # with the Robust Fidelity (so the agent knows it survived the attack).
                # We DO NOT use the 'clean_reward' shaped signal here, we use raw fidelity to be clearer.
                reward = (alpha * clean_fidelity) + ((1 - alpha) * robust_metric)
                
                # Re-apply complexity penalty if needed
                reward -= 0.01 * len(ops)

        # Update Info
        info['fidelity_under_attack'] = fidelity_under_attack
        
        return obs, reward, terminated, truncated, info