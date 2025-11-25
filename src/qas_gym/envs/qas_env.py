import sys
from contextlib import closing
from qas_gym.utils import get_default_gates, get_default_observables, fidelity_pure_target
from io import StringIO
from typing import List, Optional

import cirq
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class QuantumArchSearchEnv(gym.Env):
    metadata = {'render_modes': ['ansi', 'human'], 'render_fps': 4}

    def __init__(
            self,
            target: np.ndarray,
            fidelity_threshold: float,
            reward_penalty: float,
            max_timesteps: int,
            qubits: Optional[List[cirq.LineQubit]] = None,
            state_observables: Optional[List[cirq.GateOperation]] = None,
            action_gates: Optional[List[cirq.GateOperation]] = None,
            complexity_penalty_weight=0.0
    ):
        super(QuantumArchSearchEnv, self).__init__()

        self.target = target
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.complexity_penalty_weight = complexity_penalty_weight
        self._max_episode_steps = max_timesteps

        # --- Initialize Qubits, Observables, and Gates ---
        # This logic must come before defining the observation and action spaces.
        if qubits is None:
            n_qubits = int(np.log2(len(target)))
            qubits = cirq.LineQubit.range(n_qubits)
        if state_observables is None:
            state_observables = get_default_observables(qubits)
        action_gates = action_gates if action_gates is not None else get_default_gates(qubits)

        self.qubits = qubits
        self.state_observables = state_observables
        self.champion_circuit = None
        self.best_fidelity = -1.0
        self.previous_final_fidelity = 0.0 # For reward shaping

        self.action_gates = action_gates
        self.target_density = np.outer(target, np.conj(target))
        self.simulator = cirq.DensityMatrixSimulator()
        self.observation_space = spaces.Box(low=-1.,
                                            high=1.,
                                            shape=(len(state_observables),),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(n=len(action_gates))

    def __str__(self):
        return f"QuantumArchSearch-v0(Qubits={len(self.qubits)}, Target={self.target}, " \
               f"Gates={', '.join(gate.__str__() for gate in self.action_gates)}, " \
               f"Observables={', '.join(gate.__str__() for gate in self.state_observables)})"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.circuit_gates = []
        self.previous_final_fidelity = 0.0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_cirq(self):
        return cirq.Circuit(self.circuit_gates)

    def _get_obs(self):
        circuit = self._get_cirq()
        # Add a tiny bit of noise to break symmetries and stabilize training
        stable_circuit = circuit.with_noise(cirq.depolarize(1e-6))
        return self._get_obs_from_circuit(stable_circuit)

    def _get_obs_from_circuit(self, circuit_to_obs):
        result = self.simulator.simulate(circuit_to_obs, qubit_order=self.qubits)
        final_density_matrix = 0.5 * (
                result.final_density_matrix + np.conj(result.final_density_matrix).T)
        obs = [np.real(np.trace(            
            cirq.Circuit(ob).unitary(qubit_order=self.qubits) @ final_density_matrix))
            for ob in self.state_observables]
        return np.array(obs).astype(np.float32)

    def get_fidelity(self, circuit):
        """Unified fidelity computation using fidelity_pure_target helper.

        The target is assumed pure throughout this project; use the canonical
        inner-product form for consistency with saboteur evaluation.
        """
        return fidelity_pure_target(circuit, self.target, self.qubits)

    def get_circuit_complexity(self, circuit):
        return len(circuit)

    def step(self, action):
        # The action from the agent can be a numpy array, so we must convert it to a scalar int for indexing.
        action_idx = int(action)
        action_gate = self.action_gates[action_idx]
        reward_penalty = 0.0
        last_op_on_qubits = next((gate for gate in reversed(self.circuit_gates)
                                  if gate.qubits == action_gate.qubits), None)

        if last_op_on_qubits and action_gate == cirq.inverse(last_op_on_qubits):
            reward_penalty = -0.1

        self.circuit_gates.append(action_gate)
        circuit = self._get_cirq()
        observation = self._get_obs()

        # The fidelity is calculated on the clean circuit at each step for reward shaping.
        # The final, post-sabotage fidelity will be handled in the external training loop.
        current_fidelity = self.get_fidelity(circuit)

        terminated = (current_fidelity >= self.fidelity_threshold) or \
                     (self.get_circuit_complexity(circuit) >= self.max_timesteps)
        truncated = False

        # --- Reward Shaping for Architect ---
        # This reward encourages building a high-fidelity circuit.
        # The primary reward for robustness will come from the training loop.
        fidelity_delta = current_fidelity - self.previous_final_fidelity
        reward = (0.1 * fidelity_delta) + reward_penalty  # Small shaping reward
        if terminated:
            reward -= self.complexity_penalty_weight * self.get_circuit_complexity(circuit)

        info = {'fidelity': current_fidelity, 'circuit': circuit}
        if current_fidelity > self.best_fidelity:
            self.best_fidelity = current_fidelity
            self.champion_circuit = circuit
            info['is_champion'] = True
            info['champion_fidelity'] = current_fidelity

        # Update the previous final fidelity for the next step's reward shaping
        self.previous_final_fidelity = current_fidelity

        return observation, reward, terminated, truncated, info

    def render(self, mode='human', circuit=None):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        circuit = circuit or self._get_cirq()
        outfile.write(f'\n{circuit}\n')

        if mode != 'human':
            with closing(outfile):
                # Only StringIO (ansi mode) supports getvalue; degrade gracefully.
                return outfile.getvalue() if isinstance(outfile, StringIO) else ''