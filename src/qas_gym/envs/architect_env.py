import numpy as np
import cirq
from .qas_env import QuantumArchSearchEnv
# We must import the Saboteur class to access the static helper and error rates
from .saboteur_env import SaboteurMultiGateEnv
from src.utils.metrics import state_energy

try:
    from utils.standard_hamiltonians import get_standard_hamiltonian  # type: ignore
except Exception:
    get_standard_hamiltonian = None

class ArchitectEnv(QuantumArchSearchEnv):
    """
    An environment for the architect agent that uses a sophisticated reward function.
    """
    def __init__(
        self,
        complexity_penalty_weight: float = 0.01,
        hamiltonian_matrix=None,
        hamiltonian_name: str | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Track best metric seen within an episode for improvement-based shaping
        self.best_episode_metric = 0.0
        # NEW: store λ for the sweep
        self.complexity_penalty_weight = complexity_penalty_weight
        self.hamiltonian_matrix = hamiltonian_matrix
        self.hamiltonian_name = hamiltonian_name
        if self.hamiltonian_matrix is None and hamiltonian_name and get_standard_hamiltonian:
            try:
                info = get_standard_hamiltonian(hamiltonian_name)
                self.hamiltonian_matrix = info.get("matrix")
            except Exception:
                self.hamiltonian_matrix = None


    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.best_episode_metric = 0.0
        return obs, info

    def step(self, action):
        """
        Step with enhanced reward to balance fidelity and complexity.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        
        gate_count = info.get('total_gates', 0)

        # VQE reward override: reward = -energy with complexity penalty
        if self.task_mode == 'vqe' and self.hamiltonian_matrix is not None:
            circuit = self._get_cirq()
            sim = cirq.Simulator()
            result = sim.simulate(circuit, qubit_order=self.qubits)
            state_vec = result.final_state_vector
            energy = state_energy(state_vec, self.hamiltonian_matrix)
            reward = -energy - self.complexity_penalty_weight * gate_count
            info['energy'] = energy
            info['shaped_reward'] = reward
            return obs, reward, terminated, truncated, info

        # Use process fidelity when available (unitary mode), otherwise state fidelity
        metric = info.get('process_fidelity', info.get('fidelity', 0.0))

        # In unitary_preparation, when the environment provides process fidelity at
        # termination, pass it through directly so PPO optimizes the correct metric.
        if terminated and 'process_fidelity' in info:
            reward = info['process_fidelity']
            info['shaped_reward'] = reward
            return obs, reward, terminated, truncated, info

        # Reward only positive improvements over the best fidelity seen this episode.
        # This avoids penalizing temporary drops (e.g., when building GHZ requires
        # an intermediate decrease) and works for any target/task metric.
        improvement = max(0.0, metric - self.best_episode_metric)
        if metric > self.best_episode_metric:
            self.best_episode_metric = metric
        reward = improvement

        # Penalty for excessive complexity
        reward -= self.complexity_penalty_weight * gate_count

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
        saboteur_budget_fraction: float | None = 0.2,
        saboteur_start_budget_scale: float = 0.3,
        saboteur_error_rates=None,
        saboteur_noise_family: str = "depolarizing",
        saboteur_noise_kwargs: dict | None = None,
        alpha_start: float = 0.6,
        alpha_end: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.saboteur_agent = saboteur_agent
        self.saboteur_max_error_level = saboteur_max_error_level

        # Global step counter for annealing alpha
        self.total_training_steps = float(total_training_steps)
        self.global_step = 0

        # Start mostly clean (structure), end fully robustness-focused
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

        # Attack budget (number of gates that can be attacked per episode)
        self.saboteur_budget = saboteur_budget
        # Fractional budget relative to circuit length (if set)
        self.saboteur_budget_fraction = saboteur_budget_fraction
        # Start the ramp at a smaller fraction for stability (1.0 disables ramp)
        self.saboteur_start_budget_scale = saboteur_start_budget_scale
        # Noise configuration shared with the saboteur policy
        rates = list(saboteur_error_rates) if saboteur_error_rates is not None else SaboteurMultiGateEnv.all_error_rates
        if saboteur_max_error_level is not None and saboteur_max_error_level > 0:
            rates = rates[:saboteur_max_error_level]
        if len(rates) == 0:
            rates = SaboteurMultiGateEnv.all_error_rates
        self.saboteur_error_rates = rates
        self.saboteur_noise_family = saboteur_noise_family
        self.saboteur_noise_kwargs = saboteur_noise_kwargs.copy() if saboteur_noise_kwargs is not None else {}

    def step(self, action):
        # 1. Execute Architect Step (Get Clean Fidelity & curriculum reward)
        obs, clean_reward, terminated, truncated, info = super().step(action)
        clean_fidelity = info.get('fidelity', 0.0)
        clean_process_fidelity = info.get('process_fidelity', None)

        # Advance global step (used to anneal alpha)
        self.global_step += 1

        # Default: no attack, pure curriculum reward
        fidelity_under_attack = clean_fidelity
        process_fidelity_under_attack = None
        reward = clean_reward

        # 2. If Episode Ends (Circuit Complete), Apply Saboteur Attack
        if terminated and self.saboteur_agent is not None:
            import numpy as _np
            final_circuit = self._get_cirq()

            # Guard for empty circuits
            if final_circuit is not None and len(list(final_circuit.all_operations())) > 0:
                # A. Build saboteur observation from the final circuit
                saboteur_target = self.target
                if saboteur_target is None:
                    saboteur_target = _np.zeros(2 ** len(self.qubits), dtype=complex)
                sab_env = SaboteurMultiGateEnv(
                    architect_circuit=final_circuit,
                    target_state=saboteur_target,
                    max_circuit_timesteps=self.max_timesteps,
                    n_qubits=len(self.qubits),
                    error_rates=self.saboteur_error_rates,
                    noise_family=self.saboteur_noise_family,
                    noise_kwargs=self.saboteur_noise_kwargs,
                    max_concurrent_attacks=self.saboteur_budget if self.saboteur_budget is not None else self.max_timesteps,
                )
                sab_obs = SaboteurMultiGateEnv._get_obs(sab_env, final_circuit)

                # B. Saboteur policy (deterministic: approximate worst-case)
                sab_action, _ = self.saboteur_agent.predict(sab_obs, deterministic=True)

                # C. Apply Noise with budgeted top-K attack
                ops = list(final_circuit.all_operations())
                valid_gate_count = min(len(ops), self.max_timesteps)
                raw_action = _np.array(sab_action[:valid_gate_count], dtype=int)

                # Length-aware, ramped budget: fraction of gates with optional cap and ramp-up
                target_budget = self.saboteur_budget if self.saboteur_budget is not None else valid_gate_count
                if self.saboteur_budget_fraction is not None:
                    frac_budget = int(_np.ceil(self.saboteur_budget_fraction * valid_gate_count))
                    frac_budget = max(1, frac_budget)
                    target_budget = frac_budget if target_budget is None else min(target_budget, frac_budget)
                progress = min(1.0, self.global_step / self.total_training_steps) if self.total_training_steps > 0 else 1.0
                budget_scale = self.saboteur_start_budget_scale + progress * (1.0 - self.saboteur_start_budget_scale)
                budget = int(_np.ceil(target_budget * budget_scale)) if target_budget is not None else valid_gate_count
                budget = max(0, min(budget, valid_gate_count))

                noisy_circuit, applied_rates, _ = SaboteurMultiGateEnv.build_noisy_circuit(
                    circuit=final_circuit,
                    action=raw_action,
                    error_rates=self.saboteur_error_rates,
                    noise_family=self.saboteur_noise_family,
                    max_concurrent_attacks=budget,
                    max_gates=self.max_timesteps,
                    noise_kwargs=self.saboteur_noise_kwargs,
                )
                info["mean_error_rate"] = float(_np.mean(applied_rates)) if applied_rates else 0.0

                # D. Measure Robustness (state fidelity)
                fidelity_under_attack = self.get_fidelity(noisy_circuit)

                # E. MIXED REWARD CALCULATION WITH ANNEALED ALPHA
                # alpha goes from alpha_start -> alpha_end over training
                t = min(1.0, self.global_step / self.total_training_steps)
                alpha = self.alpha_start + t * (self.alpha_end - self.alpha_start)

                robust_metric = fidelity_under_attack

                if self.task_mode == 'unitary_preparation' and self.ideal_unitary is not None:
                    try:
                        from utils.metrics import unitary_from_basis_columns, process_fidelity
                        # Compute process fidelity for the noisy circuit
                        dim = 2 ** len(self.qubits)
                        columns = []
                        sim = cirq.Simulator()
                        for idx in range(dim):
                            init_bits = [(idx >> (len(self.qubits) - 1 - b)) & 1 for b in range(len(self.qubits))]
                            prep_ops = [cirq.X(self.qubits[b]) for b, bit in enumerate(init_bits) if bit == 1]
                            test_circuit = cirq.Circuit()
                            test_circuit.append(prep_ops)
                            test_circuit += noisy_circuit
                            result = sim.simulate(test_circuit, qubit_order=self.qubits)
                            out_state = result.final_state_vector
                            columns.append(out_state)
                        U_noisy = unitary_from_basis_columns(columns)
                        process_fidelity_under_attack = process_fidelity(self.ideal_unitary, U_noisy)
                        # Use process fidelity metrics for mixing if available
                        robust_metric = process_fidelity_under_attack
                        if clean_process_fidelity is not None:
                            clean_fidelity_metric = clean_process_fidelity
                        else:
                            clean_fidelity_metric = clean_fidelity
                    except Exception:
                        clean_fidelity_metric = clean_process_fidelity if clean_process_fidelity is not None else clean_fidelity
                else:
                    clean_fidelity_metric = clean_process_fidelity if clean_process_fidelity is not None else clean_fidelity

                # Final reward: blend structure (clean) and robustness (attacked)
                reward = alpha * clean_fidelity_metric + (1.0 - alpha) * robust_metric
                # IMPORTANT: avoid double-penalizing complexity here;
                # ArchitectEnv already has a -0.01 * l term in its shaping.
                # If you want extra penalty, reduce it substantially, e.g.:
                # reward -= 0.001 * len(ops)

        # Update Info
        info['fidelity_under_attack'] = fidelity_under_attack
        if process_fidelity_under_attack is not None:
            info['process_fidelity_under_attack'] = process_fidelity_under_attack

        return obs, reward, terminated, truncated, info
