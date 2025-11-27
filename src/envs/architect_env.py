r"""
ArchitectEnv: Base environment for the architect agent.

See ExpPlan.md, Part 1 (Hyperparameter sensitivity) and the implementation notes.
This environment is used in Experiment 1.1 (Lambda sweep) as a baseline with
static penalty $R = F - \lambda C$.

The architect agent builds quantum circuits by selecting gates to achieve
a target quantum state with high fidelity while minimizing circuit depth.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cirq

from src.utils.metrics import compute_fidelity, count_cnots, count_gates


class ArchitectEnv(gym.Env):
    """
    Base environment for the architect agent in quantum architecture search.

    The architect agent builds quantum circuits by selecting gates to achieve
    a target quantum state with high fidelity while minimizing circuit depth.

    Attributes:
        target_state: The target quantum state to prepare.
        lambda_penalty: Weight for the circuit complexity penalty.
        max_timesteps: Maximum number of gates allowed in the circuit.
        n_qubits: Number of qubits in the circuit.
        qubits: Cirq LineQubit objects.
        circuit: The current quantum circuit being built.
        current_step: Current timestep in the episode.

    See Also:
        ExpPlan.md - Experiment 1.1 (Lambda sweep)
    """

    # Gate types available for the agent
    # Index 0: Identity (no-op), 1-3: single-qubit gates, 4+: two-qubit gates
    GATE_TYPES = ["I", "H", "X", "RZ", "CNOT"]

    def __init__(
        self,
        target_state=None,
        lambda_penalty=0.01,
        max_timesteps=20,
        n_qubits=4,
        **kwargs
    ):
        """
        Initialize the ArchitectEnv.

        Args:
            target_state: Target quantum state vector.
            lambda_penalty: Complexity penalty weight (default: 0.01).
            max_timesteps: Maximum gates per episode (default: 20).
            n_qubits: Number of qubits (default: 4).
            **kwargs: Additional arguments for gym.Env.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.target_state = target_state
        self.lambda_penalty = lambda_penalty
        self.max_timesteps = max_timesteps

        # Initialize qubits
        self.qubits = cirq.LineQubit.range(n_qubits)

        # Initialize circuit and simulator
        self.circuit = cirq.Circuit()
        self.simulator = cirq.Simulator()
        self.current_step = 0

        # Define observation space: current state vector (real + imag parts)
        # Observation is the current quantum state represented as 2 * 2^n_qubits floats
        state_dim = 2 ** n_qubits
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * state_dim,),
            dtype=np.float32,
        )

        # Define action space:
        # Action = (gate_type, qubit1, qubit2)
        # - gate_type: 0=I, 1=H, 2=X, 3=RZ, 4=CNOT
        # - qubit1: target qubit (0 to n_qubits-1)
        # - qubit2: control qubit for CNOT (0 to n_qubits-1), ignored for single-qubit gates
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

        # Return initial observation (|0...0> state)
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
            # Identity - no operation
            pass
        elif gate_type == "H":
            self.circuit.append(cirq.H(self.qubits[qubit1_idx]))
        elif gate_type == "X":
            self.circuit.append(cirq.X(self.qubits[qubit1_idx]))
        elif gate_type == "RZ":
            # Use a fixed rotation angle (pi/4) for simplicity
            self.circuit.append(cirq.rz(np.pi / 4)(self.qubits[qubit1_idx]))
        elif gate_type == "CNOT":
            # Ensure control and target are different
            if qubit1_idx != qubit2_idx:
                self.circuit.append(
                    cirq.CNOT(self.qubits[qubit2_idx], self.qubits[qubit1_idx])
                )

        self.current_step += 1

        # Compute fidelity
        fidelity = 0.0
        if self.target_state is not None:
            fidelity = compute_fidelity(self.circuit, self.target_state, self.qubits)

        # Compute complexity (gate count as proxy for circuit cost C)
        gate_count = count_gates(self.circuit)
        cnot_count = count_cnots(self.circuit)

        # Compute reward as R = F - lambda * C (per ExpPlan.md)
        reward = fidelity - self.lambda_penalty * gate_count

        # Check termination conditions
        terminated = fidelity > 0.99  # Success condition
        truncated = self.current_step >= self.max_timesteps

        # Get observation and info
        observation = self._get_observation()
        info = {
            "fidelity": fidelity,
            "cnot_count": cnot_count,
            "total_gates": gate_count,
            "depth": len(self.circuit),
        }

        return observation, reward, terminated, truncated, info

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
            # Empty circuit - return |0...0> state
            dim = 2 ** self.n_qubits
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0
        else:
            result = self.simulator.simulate(self.circuit, qubit_order=self.qubits)
            state = result.final_state_vector

        # Flatten real and imaginary parts
        obs = np.concatenate([state.real, state.imag]).astype(np.float32)
        return obs
