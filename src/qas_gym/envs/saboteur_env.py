import gymnasium as gym
import numpy as np
import cirq
from gymnasium import spaces

class SaboteurMultiGateEnv(gym.Env):
    all_error_rates = [0.001, 0.005, 0.01, 0.05]

    def __init__(self, architect_circuit, target_state, max_circuit_timesteps=20, discrete=True, episode_length=1, lambda_penalty=0.5, **kwargs):
        super().__init__()
        self.architect_circuit = architect_circuit
        self.target_state = target_state
        self.n_qubits = kwargs.get('n_qubits', int(np.log2(len(target_state))))
        self.qubits = sorted(list(architect_circuit.all_qubits()))
        self.simulator = cirq.DensityMatrixSimulator()
        self.lambda_penalty = lambda_penalty
        self.max_circuit_timesteps = max_circuit_timesteps
        self.discrete = discrete
        self.max_error_level = kwargs.get('max_error_level', len(self.all_error_rates))
        self.max_error_level = max(1, min(self.max_error_level, len(self.all_error_rates)))
        self.num_gates = len(list(self.architect_circuit.all_operations()))
        if self.discrete:
            self.action_space = spaces.MultiDiscrete([self.max_error_level] * self.num_gates)
        else:
            self.action_space = spaces.Box(low=min(self.all_error_rates), high=max(self.all_error_rates), shape=(self.num_gates,), dtype=np.float32)
        self._step_counter = 0
        self.episode_length = episode_length
        self.current_step = 0
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * self.n_qubits,), dtype=np.float32)

    def set_circuit(self, circuit: cirq.Circuit):
        self.architect_circuit = circuit
        self.qubits = sorted(list(self.architect_circuit.all_qubits()))
        new_num_gates = len(list(self.architect_circuit.all_operations()))
        if new_num_gates != self.num_gates:
            pass  # print(f"[SaboteurMultiGateEnv] WARNING: Number of gates changed from {self.num_gates} to {new_num_gates}. Action space will be updated.")
        self.num_gates = new_num_gates
        if self.discrete:
            self.action_space = spaces.MultiDiscrete([self.max_error_level] * self.num_gates)
        else:
            self.action_space = spaces.Box(low=min(self.all_error_rates), high=max(self.all_error_rates), shape=(self.num_gates,), dtype=np.float32)

    def _get_obs(self, circuit=None):
        all_qubits = list(cirq.LineQubit.range(self.n_qubits))
        target_circuit = circuit if circuit is not None else self.architect_circuit
        if not target_circuit.all_operations():
            return np.zeros((2 * self.n_qubits,), dtype=np.float32)
        obs = self.simulator.simulate_expectation_values(
            target_circuit,
            observables=[cirq.X(q) for q in all_qubits] + [cirq.Y(q) for q in all_qubits],
            qubit_order=all_qubits
        )
        return np.array(obs).real.astype(np.float32)

    def step(self, action):
        ops = list(self.architect_circuit.all_operations())
        num_gates = len(ops)
        if num_gates == 0:
            obs = self._get_obs()
            fidelity = 0.0
            reward = 1.0 - fidelity
            terminated = True
            truncated = False
            info = {'fidelity': fidelity, 'noisy_circuit': self.architect_circuit}
            return obs, reward, terminated, truncated, info

        noisy_circuit = cirq.Circuit()
        per_gate_error = []
        for i, op in enumerate(ops):
            noisy_circuit.append(op)
            if self.discrete:
                idx = int(action[i])
                idx = max(0, min(idx, self.max_error_level - 1))
                error_rate = self.all_error_rates[idx]
            else:
                error_rate = float(action[i])
                error_rate = max(min(error_rate, max(self.all_error_rates)), min(self.all_error_rates))
            per_gate_error.append(error_rate)
            for q in op.qubits:
                noisy_circuit.append(cirq.DepolarizingChannel(error_rate).on(q))
        # print(f"[SaboteurMultiGateEnv] Step: per-gate error rates: {per_gate_error}")
        # print(f"[SaboteurMultiGateEnv] Step: noisy circuit after noise application:\n{noisy_circuit}")

        all_qubits = list(cirq.LineQubit.range(self.n_qubits))
        result = self.simulator.simulate(noisy_circuit, qubit_order=all_qubits)
        final_density_matrix = result.final_density_matrix
        fidelity = np.abs(np.vdot(self.target_state, final_density_matrix @ self.target_state))
        # print(f"[SaboteurMultiGateEnv] Fidelity after attack: {fidelity:.6f}")
        # New reward: (1 - fidelity) - lambda_penalty * mean(error_rate)
        mean_error = float(np.mean(per_gate_error)) if per_gate_error else 0.0
        reward = (1.0 - fidelity) - self.lambda_penalty * mean_error
        # print(f"[SaboteurMultiGateEnv] Step: reward computed (scaled): {reward}")

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False
        info = {'fidelity': fidelity, 'noisy_circuit': noisy_circuit,
                'per_gate_error': per_gate_error}

        obs = self._get_obs(noisy_circuit)

        if terminated:
            self.current_step = 0

        return obs, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        # No need to call super().reset() for gym.Env
        obs = self._get_obs()
        info = {}
        return obs, info

    @staticmethod
    def create_observation_from_circuit(circuit: cirq.Circuit, n_qubits: int) -> np.ndarray:
        """
        A static method to create an observation from a circuit, which can be
        used for analysis without creating a full environment instance.
        """
        if not circuit.all_qubits():
            return np.zeros((2 * n_qubits,), dtype=np.float32)

        all_qubits = cirq.LineQubit.range(n_qubits)
        simulator = cirq.DensityMatrixSimulator()
        observables = [cirq.X(q) for q in all_qubits] + [cirq.Y(q) for q in all_qubits]
        obs_values = simulator.simulate_expectation_values(circuit, observables, qubit_order=all_qubits)
        return np.array(obs_values).real.astype(np.float32)


# Alias for compatibility with architect_env.py
SaboteurEnv = SaboteurMultiGateEnv