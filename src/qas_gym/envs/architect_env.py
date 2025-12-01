import numpy as np
import cirq
from .qas_env import QuantumArchSearchEnv
# We must import the Saboteur class to access the static helper and error rates
from .saboteur_env import SaboteurMultiGateEnv

class ArchitectEnv(QuantumArchSearchEnv):
    """
    An environment for the architect agent that uses a sophisticated reward function.
    """
    def __init__(self, complexity_penalty_weight: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.previous_fidelity = 0.0
        # NEW: store λ for the sweep
        self.complexity_penalty_weight = complexity_penalty_weight


    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.previous_fidelity = 0.0
        return obs, info

    def step(self, action):
        """
        Step with enhanced reward to balance fidelity and complexity.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        
        fidelity = info.get('fidelity', 0.0)
        l = info.get('gate_count', 0)

        # Example: penalize gate count lightly while rewarding fidelity improvements
        delta_fidelity = fidelity - self.previous_fidelity
        self.previous_fidelity = fidelity

        # Base reward: improvement in fidelity
        reward = delta_fidelity

        # Penalty for excessive complexity
        reward -= self.complexity_penalty_weight * l

        info['shaped_reward'] = reward
        return obs, reward, terminated, truncated, info


class AdversarialArchitectEnv(ArchitectEnv):
    """
    Adversarial Environment with REWARD MIXING.
    
    This environment mixes the 'Clean Fidelity' (structure) with 
    'Fidelity Under Attack' (robustness) to prevent the Saboteur 
    from suppressing the Architect's learning entirely.
    """

    def __init__(
        self,
        saboteur_agent=None,
        saboteur_max_error_level=None,
        total_training_steps: int = 100_000,
        saboteur_budget: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.saboteur_agent = saboteur_agent
        self.saboteur_max_error_level = saboteur_max_error_level

        # Global step counter for annealing alpha
        self.total_training_steps = float(total_training_steps)
        self.global_step = 0

        # Start mostly clean (structure), end fully robustness-focused
        self.alpha_start = 0.6
        self.alpha_end = 0.0

        # Attack budget (number of gates that can be attacked per episode)
        self.saboteur_budget = saboteur_budget

    def step(self, action):
        # 1. Execute Architect Step (Get Clean Fidelity & curriculum reward)
        obs, clean_reward, terminated, truncated, info = super().step(action)
        clean_fidelity = info.get('fidelity', 0.0)

        # Advance global step (used to anneal alpha)
        self.global_step += 1

        # Default: no attack, pure curriculum reward
        fidelity_under_attack = clean_fidelity
        reward = clean_reward

        # 2. If Episode Ends (Circuit Complete), Apply Saboteur Attack
        if terminated and self.saboteur_agent is not None:
            final_circuit = self._get_cirq()

            # Guard for empty circuits
            if final_circuit is not None and len(list(final_circuit.all_operations())) > 0:
                # A. Build saboteur observation from the final circuit
                sab_obs = SaboteurMultiGateEnv._get_obs(
                    SaboteurMultiGateEnv(
                        architect_circuit=final_circuit,
                        target_state=self.target,
                        max_circuit_timesteps=self.max_timesteps,
                        n_qubits=len(self.qubits),
                    ),
                    final_circuit,
                )

                # B. Saboteur policy (deterministic: approximate worst-case)
                sab_action, _ = self.saboteur_agent.predict(sab_obs, deterministic=True)

                # C. Apply Noise with budgeted top-K attack
                ops = list(final_circuit.all_operations())
                noisy_ops = []
                all_rates = SaboteurMultiGateEnv.all_error_rates

                import numpy as _np
                valid_gate_count = min(len(ops), self.max_timesteps)
                raw_action = _np.array(sab_action[:valid_gate_count], dtype=int)

                budget = min(self.saboteur_budget, valid_gate_count)
                effective_action = _np.zeros_like(raw_action)
                if budget > 0:
                    top_k_indices = _np.argsort(raw_action)[-budget:]
                    effective_action[top_k_indices] = raw_action[top_k_indices]

                for i, op in enumerate(ops):
                    noisy_ops.append(op)
                    if i < len(effective_action):
                        idx = int(effective_action[i])
                        idx = max(0, min(idx, len(all_rates) - 1))
                        error_rate = all_rates[idx]
                        if error_rate > 0:
                            for q in op.qubits:
                                noisy_ops.append(cirq.DepolarizingChannel(error_rate).on(q))

                noisy_circuit = cirq.Circuit(noisy_ops)

                # D. Measure Robustness
                fidelity_under_attack = self.get_fidelity(noisy_circuit)

                # E. MIXED REWARD CALCULATION WITH ANNEALED ALPHA
                # alpha goes from alpha_start -> alpha_end over training
                t = min(1.0, self.global_step / self.total_training_steps)
                alpha = self.alpha_start + t * (self.alpha_end - self.alpha_start)

                robust_metric = fidelity_under_attack

                if self.task_mode == 'unitary_preparation' and self.ideal_unitary is not None:
                    try:
                        from qas_gym.utils import process_fidelity
                        # if you want unitary robustness, compute it here:
                        # robust_metric = process_fidelity(...)
                        pass
                    except Exception:
                        pass

                # Final reward: blend structure (clean) and robustness (attacked)
                reward = alpha * clean_fidelity + (1.0 - alpha) * robust_metric
                # IMPORTANT: avoid double-penalizing complexity here;
                # ArchitectEnv already has a -0.01 * l term in its shaping.
                # If you want extra penalty, reduce it substantially, e.g.:
                # reward -= 0.001 * len(ops)

        # Update Info
        info['fidelity_under_attack'] = fidelity_under_attack

        return obs, reward, terminated, truncated, info
