import gymnasium as gym
import numpy as np
import cirq
from gymnasium import spaces


def _robust_expectations(simulator, circuit, qubits):
    """Compute <X>, <Y> robustly to seed observations."""
    observables = [cirq.X(q) for q in qubits] + [cirq.Y(q) for q in qubits]
    try:
        return simulator.simulate_expectation_values(circuit, observables=observables, qubit_order=qubits)
    except Exception:
        result = simulator.simulate(circuit, qubit_order=qubits)
        rho = result.final_density_matrix
        rho = 0.5 * (rho + rho.conj().T)  # enforce Hermiticity
        vals = []
        for obs in observables:
            obs_mat = cirq.Circuit(obs).unitary(qubit_order=qubits)
            vals.append(np.real(np.trace(rho @ obs_mat)))
        return vals


class CoherentSaboteurEnv(gym.Env):
    """
    Saboteur that injects coherent over-rotation noise (Rx) after attacked gates.
    Action: per-gate discrete angle index; budgeted top-k attacks.
    Reward: increase damage (1-fid) minus penalty proportional to mean angle.
    """
    metadata = {'render_modes': []}
    all_error_angles = [0.0, 0.01, 0.02, 0.05, 0.10]  # radians

    def __init__(self, architect_circuit, target_state, max_circuit_timesteps=20,
                 episode_length=1, lambda_penalty=0.5, max_concurrent_attacks=3, **kwargs):
        super().__init__()
        self.target_state = target_state
        self.max_gates = max_circuit_timesteps
        self.n_qubits = kwargs.get('n_qubits', int(np.log2(len(target_state))))
        self.simulator = cirq.DensityMatrixSimulator()
        self.current_circuit = architect_circuit
        self.episode_length = episode_length
        self.current_step = 0
        self.lambda_penalty = lambda_penalty
        self.max_concurrent_attacks = max_concurrent_attacks

        self.num_levels = len(self.all_error_angles)
        self.action_space = spaces.MultiDiscrete([self.num_levels] * self.max_gates)
        self.observation_space = spaces.Dict({
            "projected_state": spaces.Box(low=-1.0, high=1.0, shape=(2 * self.n_qubits,), dtype=np.float32),
            "gate_structure": spaces.Box(low=0, high=10, shape=(self.max_gates,), dtype=np.int32)
        })

    def set_circuit(self, circuit: cirq.Circuit):
        self.current_circuit = circuit

    def _encode_gate(self, op):
        if isinstance(op.gate, cirq.XPowGate): return 1
        if isinstance(op.gate, cirq.YPowGate): return 2
        if isinstance(op.gate, cirq.ZPowGate): return 3
        if isinstance(op.gate, cirq.HPowGate): return 4
        if isinstance(op.gate, cirq.CNotPowGate): return 5
        if isinstance(op.gate, cirq.Rx): return 7
        if isinstance(op.gate, cirq.Ry): return 8
        if isinstance(op.gate, cirq.Rz): return 9
        return 6

    def _get_obs(self, circuit=None):
        circuit = circuit or self.current_circuit
        all_qubits = cirq.LineQubit.range(self.n_qubits)
        if circuit is None or not circuit.all_operations():
            state_obs = np.zeros((2 * self.n_qubits,), dtype=np.float32)
        else:
            state_obs = np.array(_robust_expectations(self.simulator, circuit, all_qubits)).real.astype(np.float32)
        structure = np.zeros((self.max_gates,), dtype=np.int32)
        ops = list(circuit.all_operations()) if circuit else []
        for i, op in enumerate(ops[:self.max_gates]):
            structure[i] = self._encode_gate(op)
        return {
            "projected_state": state_obs,
            "gate_structure": structure,
        }

    @staticmethod
    def create_observation_from_circuit(circuit: cirq.Circuit, n_qubits: int, max_circuit_timesteps: int | None = None):
        max_steps = max_circuit_timesteps if max_circuit_timesteps is not None else 20
        dummy_target = np.zeros(2 ** n_qubits, dtype=complex)
        env = CoherentSaboteurEnv(
            architect_circuit=circuit,
            target_state=dummy_target,
            max_circuit_timesteps=max_steps,
            n_qubits=n_qubits,
        )
        return env._get_obs(circuit)

    def step(self, action):
        if self.current_circuit is None:
            return self._get_obs(), 0.0, True, False, {}

        ops = list(self.current_circuit.all_operations())
        valid_gate_count = min(len(ops), self.max_gates)
        effective_action = np.zeros(valid_gate_count, dtype=int)
        if valid_gate_count > 0:
            raw_action = action[:valid_gate_count]
            budget = min(self.max_concurrent_attacks, valid_gate_count)
            top_k_indices = np.argsort(raw_action)[-budget:]
            effective_action[top_k_indices] = raw_action[top_k_indices]

        noisy_ops = []
        applied_angles = []
        for i in range(valid_gate_count):
            op = ops[i]
            noisy_ops.append(op)
            angle_idx = int(effective_action[i])
            angle_idx = max(0, min(angle_idx, self.num_levels - 1))
            angle = self.all_error_angles[angle_idx]
            if angle > 0:
                applied_angles.append(angle)
                for q in op.qubits:
                    noisy_ops.append(cirq.rx(angle).on(q))

        noisy_circuit = cirq.Circuit(noisy_ops)
        all_qubits = cirq.LineQubit.range(self.n_qubits)
        result = self.simulator.simulate(noisy_circuit, qubit_order=all_qubits)
        rho = result.final_density_matrix
        fidelity = np.real(np.vdot(self.target_state, rho @ self.target_state))

        mean_angle = np.mean(applied_angles) if applied_angles else 0.0
        reward = (1.0 - fidelity) - self.lambda_penalty * mean_angle

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        info = {'fidelity': fidelity, 'mean_angle': mean_angle}
        return self._get_obs(self.current_circuit), reward, terminated, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(self.current_circuit), {}


class CoherentAdversarialArchitectEnv(gym.Env):
    """
    Drop-in adversarial architect environment that uses CoherentSaboteurEnv
    for over-rotation attacks instead of depolarizing noise. Keeps the same
    reward mixing as AdversarialArchitectEnv but applies Rx noise.
    """
    def __init__(
        self,
        base_arch_env,
        saboteur_agent=None,
        total_training_steps: int = 100_000,
        saboteur_budget: int = 3,
        saboteur_budget_fraction: float | None = 0.2,
        saboteur_start_budget_scale: float = 0.3,
    ):
        # Wrap an ArchitectEnv instance to avoid modifying existing classes.
        self.arch_env = base_arch_env
        self.saboteur_agent = saboteur_agent
        self.total_training_steps = float(total_training_steps)
        self.global_step = 0
        self.alpha_start = 0.6
        self.alpha_end = 0.0
        self.saboteur_budget = saboteur_budget
        self.saboteur_budget_fraction = saboteur_budget_fraction
        self.saboteur_start_budget_scale = saboteur_start_budget_scale
        self.max_timesteps = base_arch_env.max_timesteps
        self.qubits = base_arch_env.qubits
        self.target = base_arch_env.target
        self.task_mode = getattr(base_arch_env, "task_mode", "state_preparation")
        self.ideal_unitary = getattr(base_arch_env, "ideal_unitary", None)
        # Expose spaces for SB3 compatibility
        self.observation_space = base_arch_env.observation_space
        self.action_space = base_arch_env.action_space

    def reset(self, seed=None, options=None):
        return self.arch_env.reset(seed=seed, options=options)

    def _get_cirq(self):
        return self.arch_env._get_cirq()

    def get_fidelity(self, circuit):
        return self.arch_env.get_fidelity(circuit)

    def step(self, action):
        obs, clean_reward, terminated, truncated, info = self.arch_env.step(action)
        clean_fidelity = info.get('fidelity', 0.0)
        clean_process_fidelity = info.get('process_fidelity', None)

        self.global_step += 1
        fidelity_under_attack = clean_fidelity
        process_fidelity_under_attack = None
        reward = clean_reward

        if terminated and self.saboteur_agent is not None:
            final_circuit = self._get_cirq()
            if final_circuit is not None and len(list(final_circuit.all_operations())) > 0:
                sab_obs = CoherentSaboteurEnv.create_observation_from_circuit(
                    final_circuit,
                    n_qubits=len(self.qubits),
                    max_circuit_timesteps=self.max_timesteps,
                )
                sab_action, _ = self.saboteur_agent.predict(sab_obs, deterministic=True)

                ops = list(final_circuit.all_operations())
                noisy_ops = []
                angles = CoherentSaboteurEnv.all_error_angles
                import numpy as _np
                valid_gate_count = min(len(ops), self.max_timesteps)
                raw_action = _np.array(sab_action[:valid_gate_count], dtype=int)

                target_budget = self.saboteur_budget if self.saboteur_budget is not None else valid_gate_count
                if self.saboteur_budget_fraction is not None:
                    frac_budget = int(_np.ceil(self.saboteur_budget_fraction * valid_gate_count))
                    frac_budget = max(1, frac_budget)
                    target_budget = frac_budget if target_budget is None else min(target_budget, frac_budget)
                progress = min(1.0, self.global_step / self.total_training_steps) if self.total_training_steps > 0 else 1.0
                budget_scale = self.saboteur_start_budget_scale + progress * (1.0 - self.saboteur_start_budget_scale)
                budget = int(_np.ceil(target_budget * budget_scale))
                budget = max(1, min(budget, valid_gate_count))

                effective_action = _np.zeros_like(raw_action)
                if budget > 0:
                    top_k_indices = _np.argsort(raw_action)[-budget:]
                    effective_action[top_k_indices] = raw_action[top_k_indices]

                for i, op in enumerate(ops):
                    noisy_ops.append(op)
                    if i < len(effective_action):
                        idx = int(effective_action[i])
                        idx = max(0, min(idx, len(angles) - 1))
                        angle = angles[idx]
                        if angle > 0:
                            for q in op.qubits:
                                noisy_ops.append(cirq.rx(angle).on(q))

                noisy_circuit = cirq.Circuit(noisy_ops)
                fidelity_under_attack = self.get_fidelity(noisy_circuit)

                # Anneal alpha between clean and robust metrics
                t = min(1.0, self.global_step / self.total_training_steps)
                alpha = self.alpha_start + t * (self.alpha_end - self.alpha_start)
                robust_metric = fidelity_under_attack

                if self.task_mode == 'unitary_preparation' and self.ideal_unitary is not None:
                    try:
                        from utils.metrics import unitary_from_basis_columns, process_fidelity
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
                        robust_metric = process_fidelity_under_attack
                        clean_fidelity_metric = clean_process_fidelity if clean_process_fidelity is not None else clean_fidelity
                    except Exception:
                        clean_fidelity_metric = clean_process_fidelity if clean_process_fidelity is not None else clean_fidelity
                else:
                    clean_fidelity_metric = clean_process_fidelity if clean_process_fidelity is not None else clean_fidelity

                reward = alpha * clean_fidelity_metric + (1.0 - alpha) * robust_metric

        info['fidelity_under_attack'] = fidelity_under_attack
        if process_fidelity_under_attack is not None:
            info['process_fidelity_under_attack'] = process_fidelity_under_attack
        return obs, reward, terminated, truncated, info
