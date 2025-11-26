"""
Saboteur: Noise injection agent for robustness testing.

See ExpPlan.md, Part 7.1 (Saboteur efficacy check).
The Saboteur acts as an adversary that injects noise into quantum circuits
to test and improve their robustness.

Stage 7.1 verification test:
  - Load a perfect GHZ circuit
  - Let the Saboteur act for 1 step with max budget
  - Verify: Fidelity must drop significantly below 1.0
  - If fidelity stays 1.0, noise injection is broken

TODO: Implement the following:
  - Multi-gate noise injection strategy
  - Attack budget constraint
  - Error rate selection mechanism
  - Integration with depolarizing noise channels
"""

import gymnasium as gym
from gymnasium import spaces
import cirq
import numpy as np


class Saboteur(gym.Env):
    """
    Saboteur environment for adversarial noise injection.

    The Saboteur learns to inject noise at strategic locations in a
    quantum circuit to minimally degrade its fidelity while staying
    within an attack budget.

    Attributes:
        target_circuit: The quantum circuit to attack.
        target_state: The target quantum state for fidelity computation.
        max_concurrent_attacks: Maximum gates to attack per step.
        error_rates: Available noise levels for injection.

    See Also:
        ExpPlan.md - Part 7.1 (Saboteur efficacy check)
    """

    # Available error rates for noise injection
    # 0.0 = no attack, higher values = stronger noise
    ERROR_RATES = [0.0, 0.001, 0.005, 0.01]
    MAX_ERROR_RATE = 0.1  # Maximum error rate for depolarizing noise

    def __init__(
        self,
        target_circuit=None,
        target_state=None,
        qubits=None,
        max_gates=20,
        max_concurrent_attacks=2,
        **kwargs
    ):
        """
        Initialize the Saboteur environment.

        Args:
            target_circuit: The circuit to attack.
            target_state: Target state for fidelity computation.
            qubits: Qubits used in the circuit.
            max_gates: Maximum gates in the circuit (for action space).
            max_concurrent_attacks: Attack budget per step.
            **kwargs: Additional arguments for gym.Env.
        """
        super().__init__()
        self.target_circuit = target_circuit
        self.target_state = target_state
        self.qubits = qubits
        self.max_gates = max_gates
        self.max_concurrent_attacks = max_concurrent_attacks

        # Determine n_qubits from circuit or qubits
        if qubits is not None:
            self.n_qubits = len(qubits)
        elif target_circuit is not None:
            self.n_qubits = len(list(target_circuit.all_qubits()))
        else:
            self.n_qubits = 4  # Default

        # Define observation space
        self.observation_space = spaces.Dict({
            'projected_state': spaces.Box(
                low=-1.0, high=1.0,
                shape=(2 * self.n_qubits,),
                dtype=np.float32
            ),
            'gate_structure': spaces.Box(
                low=0, high=len(self.ERROR_RATES),
                shape=(self.max_gates,),
                dtype=np.int32
            )
        })

        # Define action space: one error level per gate
        self.action_space = spaces.MultiDiscrete(
            [len(self.ERROR_RATES)] * self.max_gates
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        """
        Execute a noise injection attack.

        Args:
            action: Per-gate error level indices.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Apply noise with budget constraint
        noisy_circuit, _ = self.apply_noise(self.target_circuit, action)
        observation = self._get_observation()
        reward = 0.0  # Placeholder
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def set_circuit(self, circuit):
        """
        Update the target circuit for attack.

        Args:
            circuit: New circuit to attack.
        """
        self.target_circuit = circuit

    def apply_noise(self, circuit, action):
        """
        Apply noise to circuit based on action, respecting attack budget.

        Args:
            circuit: The quantum circuit.
            action: Per-gate error level indices.

        Returns:
            Tuple of (noisy_circuit, num_attacks) where num_attacks is the
            number of gates that had noise applied (respecting budget).
        """
        if circuit is None:
            raise ValueError("No circuit provided")

        noisy_circuit = circuit.copy()
        qubits = self.qubits if self.qubits is not None else list(
            circuit.all_qubits()
        )

        # Get all operations from circuit
        operations = list(circuit.all_operations())

        # Count attacks requested (error_level > 0)
        attack_indices = []
        for i, error_level in enumerate(action):
            if i < len(operations) and error_level > 0:
                attack_indices.append(i)

        # Apply budget constraint: only attack up to max_concurrent_attacks
        attacks_to_apply = attack_indices[:self.max_concurrent_attacks]
        num_attacks = len(attacks_to_apply)

        # Apply noise after operations that are under attack
        for idx in attacks_to_apply:
            if idx < len(operations):
                op = operations[idx]
                error_level = action[idx]
                error_rate = self.ERROR_RATES[error_level]
                if error_rate > 0:
                    # Apply depolarizing noise to qubits involved in this gate
                    for q in op.qubits:
                        noisy_circuit.append(cirq.depolarize(p=error_rate).on(q))

        return noisy_circuit, num_attacks

    def apply_max_noise(self, error_rate=None):
        """
        Apply maximum depolarizing noise to all qubits after the circuit.

        This is a simplified attack method for testing purposes.
        Applies depolarizing noise channel to all qubits in the circuit.

        Args:
            error_rate: Error rate for depolarizing noise.
                       If None, uses MAX_ERROR_RATE.

        Returns:
            Tuple of (noisy_circuit, qubits) where noisy_circuit has
            depolarizing noise applied after all operations.
        """
        if self.target_circuit is None:
            raise ValueError("No target circuit set")

        if error_rate is None:
            error_rate = self.MAX_ERROR_RATE

        # Create a copy of the circuit with noise
        noisy_circuit = self.target_circuit.copy()

        # Apply depolarizing noise to all qubits
        qubits = self.qubits if self.qubits is not None else list(
            self.target_circuit.all_qubits()
        )

        # Add depolarizing channel after the circuit
        noise_ops = [cirq.depolarize(p=error_rate).on(q) for q in qubits]
        noisy_circuit.append(noise_ops)

        return noisy_circuit, qubits

    def _get_observation(self):
        """
        Generate observation dict for current state.

        Returns:
            Dict with 'projected_state' and 'gate_structure'.
        """
        return self.create_observation_from_circuit(
            self.target_circuit,
            self.n_qubits,
            self.max_gates
        )

    @staticmethod
    def create_observation_from_circuit(circuit, n_qubits, max_gates=20):
        """
        Create observation dict from a circuit (static helper).

        Args:
            circuit: The quantum circuit.
            n_qubits: Number of qubits.
            max_gates: Maximum gates for padding.

        Returns:
            Dict with 'projected_state' and 'gate_structure'.
        """
        # Compute projected_state: Pauli Z expectation values (real + imag parts)
        # For simplicity, use random values for now (can be extended later)
        if circuit is not None:
            qubits = list(circuit.all_qubits())
            if len(qubits) > 0:
                # Simulate circuit to get state
                simulator = cirq.Simulator()
                result = simulator.simulate(circuit, qubit_order=qubits)
                state = result.final_state_vector

                # Compute Z expectation for each qubit
                projected = []
                for i in range(n_qubits):
                    if i < len(qubits):
                        # Compute <Z_i> = sum(|psi_j|^2 * z_j) where z_j = +1/-1
                        dim = 2 ** len(qubits)
                        z_exp = 0.0
                        for j in range(dim):
                            # Check bit i of j
                            bit = (j >> (len(qubits) - 1 - i)) & 1
                            z_val = 1 - 2 * bit  # 0 -> +1, 1 -> -1
                            z_exp += np.abs(state[j]) ** 2 * z_val
                        projected.append(np.real(z_exp))
                        projected.append(np.imag(z_exp))
                    else:
                        projected.extend([0.0, 0.0])
                projected_state = np.array(projected[:2 * n_qubits], dtype=np.float32)
            else:
                projected_state = np.zeros(2 * n_qubits, dtype=np.float32)
        else:
            projected_state = np.zeros(2 * n_qubits, dtype=np.float32)

        # Encode gate_structure: gate type indices (0 = empty)
        gate_structure = np.zeros(max_gates, dtype=np.int32)
        if circuit is not None:
            operations = list(circuit.all_operations())
            for i, op in enumerate(operations[:max_gates]):
                # Simple encoding: 1 for single-qubit, 2 for two-qubit
                gate_structure[i] = 1 if len(op.qubits) == 1 else 2

        return {
            'projected_state': projected_state,
            'gate_structure': gate_structure
        }
