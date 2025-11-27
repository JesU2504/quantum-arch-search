"""
AdversarialArchitectEnv: Architect environment with adversarial evaluation.

See ExpPlan.md, Part 1-3 and Part 5 (Computational overhead).
This environment trains the architect against a Saboteur agent, creating
an "ensemble robustness" effect without manual hyperparameter tuning.

The adversarial approach acts as a parameter-free, dynamic regularizer
that outperforms "Static Penalty" QAS methods (see Research goal in ExpPlan.md).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cirq

from src.utils.metrics import compute_fidelity, count_cnots, count_gates
from src.envs.saboteur import Saboteur


class AdversarialArchitectEnv(gym.Env):
    """
    Adversarial architect environment for robust circuit design.

    Evaluates the architect's circuits against a Saboteur agent that
    injects noise at strategic points. This creates circuits that are
    inherently robust to various noise types.

    Attributes:
        saboteur_agent: The trained Saboteur agent for adversarial evaluation.
        target_state: The target quantum state to prepare.
        max_timesteps: Maximum number of gates allowed in the circuit.
        n_qubits: Number of qubits in the circuit.
        qubits: Cirq LineQubit objects.
        circuit: The current quantum circuit being built.
        current_step: Current timestep in the episode.

    See Also:
        ExpPlan.md - Part 2 (Robustness to distribution shift)
        ExpPlan.md - Part 5 (Computational overhead)
    """

    # Gate types available for the agent
    GATE_TYPES = ["I", "H", "X", "RZ", "CNOT"]

    def __init__(
        self,
        saboteur_agent=None,
        target_state=None,
        max_timesteps=20,
        n_qubits=4,
        default_noise_rate=0.01,
        **kwargs
    ):
        """
        Initialize the AdversarialArchitectEnv.

        Args:
            saboteur_agent: Pre-trained or co-evolving Saboteur agent.
            target_state: Target quantum state vector.
            max_timesteps: Maximum gates per episode (default: 20).
            n_qubits: Number of qubits (default: 4).
            default_noise_rate: Default depolarizing noise rate when no saboteur (default: 0.01).
            **kwargs: Additional arguments for gym.Env.
        """
        super().__init__()
        self.saboteur_agent = saboteur_agent
        self.target_state = target_state
        self.max_timesteps = max_timesteps
        self.n_qubits = n_qubits
        self.default_noise_rate = default_noise_rate

        # Initialize qubits
        self.qubits = cirq.LineQubit.range(n_qubits)

        # Initialize circuit and simulator
        self.circuit = cirq.Circuit()
        self.simulator = cirq.Simulator()
        self.dm_simulator = cirq.DensityMatrixSimulator()
        self.current_step = 0

        # Define observation space: current state vector (real + imag parts)
        state_dim = 2 ** n_qubits
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * state_dim,),
            dtype=np.float32,
        )

        # Define action space:
        # Action = (gate_type, qubit1, qubit2)
        n_gate_types = len(self.GATE_TYPES)
        self.action_space = spaces.MultiDiscrete(
            [n_gate_types, n_qubits, n_qubits]
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        # Reset circuit to empty state
        self.circuit = cirq.Circuit()
        self.current_step = 0

        # Return initial observation
        observation = self._get_observation()
        info = {"cnot_count": 0, "total_gates": 0}
        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Gate selection action (gate_type, qubit1, qubit2).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        gate_type_idx, qubit1_idx, qubit2_idx = action
        gate_type = self.GATE_TYPES[gate_type_idx]

        # Apply selected gate to circuit
        if gate_type == "I":
            pass
        elif gate_type == "H":
            self.circuit.append(cirq.H(self.qubits[qubit1_idx]))
        elif gate_type == "X":
            self.circuit.append(cirq.X(self.qubits[qubit1_idx]))
        elif gate_type == "RZ":
            self.circuit.append(cirq.rz(np.pi / 4)(self.qubits[qubit1_idx]))
        elif gate_type == "CNOT":
            if qubit1_idx != qubit2_idx:
                self.circuit.append(
                    cirq.CNOT(self.qubits[qubit2_idx], self.qubits[qubit1_idx])
                )

        self.current_step += 1

        # Check if episode should terminate
        truncated = self.current_step >= self.max_timesteps

        # Compute clean fidelity
        clean_fidelity = 0.0
        if self.target_state is not None:
            clean_fidelity = compute_fidelity(
                self.circuit, self.target_state, self.qubits
            )

        # On termination, compute adversarial fidelity
        adversarial_fidelity = clean_fidelity
        if truncated or clean_fidelity > 0.99:
            adversarial_fidelity = self.get_fidelity_under_attack(self.circuit)

        # Compute complexity metrics
        gate_count = count_gates(self.circuit)
        cnot_count = count_cnots(self.circuit)

        # Adversarial reward: use fidelity under attack
        # No lambda penalty - adversarial training acts as dynamic regularizer
        reward = adversarial_fidelity

        # Check termination conditions
        terminated = clean_fidelity > 0.99

        # Get observation and info
        observation = self._get_observation()
        info = {
            "fidelity": clean_fidelity,
            "adversarial_fidelity": adversarial_fidelity,
            "cnot_count": cnot_count,
            "total_gates": gate_count,
            "depth": len(self.circuit),
        }

        return observation, reward, terminated, truncated, info

    def set_saboteur(self, saboteur_agent):
        """
        Set or update the Saboteur agent.

        Args:
            saboteur_agent: New Saboteur agent for adversarial evaluation.
        """
        self.saboteur_agent = saboteur_agent

    def get_fidelity_under_attack(self, circuit):
        """
        Compute fidelity after Saboteur noise injection.

        Args:
            circuit: The quantum circuit to evaluate.

        Returns:
            Fidelity value after adversarial noise injection.
        """
        if self.target_state is None:
            return 0.0

        if self.saboteur_agent is None:
            # No saboteur - apply default depolarizing noise
            return self._compute_fidelity_with_default_noise(circuit)

        # Generate Saboteur observation from circuit
        obs = Saboteur.create_observation_from_circuit(
            circuit, self.n_qubits, max_gates=self.max_timesteps
        )

        # Get Saboteur action
        action, _ = self.saboteur_agent.predict(obs, deterministic=True)

        # Apply noise to circuit using Saboteur
        saboteur = Saboteur(
            target_circuit=circuit,
            target_state=self.target_state,
            qubits=self.qubits,
            max_gates=self.max_timesteps,
        )
        noisy_circuit, _ = saboteur.apply_noise(circuit, action)

        # Compute fidelity with density matrix simulator (for noisy circuit)
        dm_result = self.dm_simulator.simulate(noisy_circuit, qubit_order=self.qubits)
        noisy_dm = dm_result.final_density_matrix

        # Compute fidelity: F = <target|rho|target>
        fidelity = np.real(
            np.dot(np.conj(self.target_state), np.dot(noisy_dm, self.target_state))
        )
        return float(fidelity)

    def _compute_fidelity_with_default_noise(self, circuit):
        """
        Compute fidelity with default depolarizing noise.

        Args:
            circuit: The quantum circuit to evaluate.

        Returns:
            Fidelity value after default noise.
        """
        # Apply depolarizing noise to all qubits
        noisy_circuit = circuit.copy()
        for q in self.qubits:
            noisy_circuit.append(cirq.depolarize(p=self.default_noise_rate).on(q))

        # Compute fidelity with density matrix simulator
        dm_result = self.dm_simulator.simulate(noisy_circuit, qubit_order=self.qubits)
        noisy_dm = dm_result.final_density_matrix

        fidelity = np.real(
            np.dot(np.conj(self.target_state), np.dot(noisy_dm, self.target_state))
        )
        return float(fidelity)

    def render(self, mode="human"):
        """
        Render the current circuit state.

        Args:
            mode: Rendering mode ('human' or 'ansi').
        """
        circuit_str = str(self.circuit)
        if mode == "human":
            print(circuit_str)
        return circuit_str

    def get_circuit(self):
        """
        Get the current quantum circuit.

        Returns:
            The current Cirq circuit object.
        """
        return self.circuit

    def _get_observation(self):
        """
        Get the current observation (state vector).

        Returns:
            Numpy array of shape (2 * 2^n_qubits,) representing real and imag parts.
        """
        if len(self.circuit) == 0:
            dim = 2 ** self.n_qubits
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0
        else:
            result = self.simulator.simulate(self.circuit, qubit_order=self.qubits)
            state = result.final_state_vector

        obs = np.concatenate([state.real, state.imag]).astype(np.float32)
        return obs
