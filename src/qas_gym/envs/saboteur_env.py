import gymnasium as gym
import numpy as np
import cirq
from gymnasium import spaces

class SaboteurMultiGateEnv(gym.Env):
    """
    An adversarial environment where an agent (Saboteur) adds noise to a quantum circuit
    constructed by an Architect.
    
    IMPROVEMENTS:
    1. Budget Constraint: Saboteur can only attack 'max_concurrent_attacks' gates per step.
    2. Rates: Lowered max error rate to prevent 'carpet bombing' the circuit.
    """
    # MODIFICATION 1: Reduced rates. removed 0.05 to prevent instant circuit death.
    # Added 0.0 as an explicit "No Attack" option.
    all_error_rates = [0.0, 0.001, 0.005, 0.01]

    def __init__(self, architect_circuit, target_state, max_circuit_timesteps=20, 
                 discrete=True, episode_length=1, lambda_penalty=0.5, **kwargs):
        super().__init__()
        self.target_state = target_state
        self.max_gates = max_circuit_timesteps 
        self.n_qubits = kwargs.get('n_qubits', int(np.log2(len(target_state))))
        self.simulator = cirq.DensityMatrixSimulator()
        
        self.current_circuit = architect_circuit
        self.episode_length = episode_length
        self.current_step = 0
        
        # MODIFICATION 2: Define the Attack Budget
        self.max_concurrent_attacks = 2

        # --- Fixed Action Space (Padding) ---
        self.num_error_levels = len(self.all_error_rates)
        self.action_space = spaces.MultiDiscrete([self.num_error_levels] * self.max_gates)

        # --- Enriched Observation Space (Dict) ---
        self.observation_space = spaces.Dict({
            "projected_state": spaces.Box(low=-1.0, high=1.0, shape=(2 * self.n_qubits,), dtype=np.float32),
            "gate_structure": spaces.Box(low=0, high=10, shape=(self.max_gates,), dtype=np.int32)
        })

    def set_circuit(self, circuit: cirq.Circuit):
        """Updates the target circuit without changing observation/action shapes."""
        self.current_circuit = circuit

    def _encode_gate(self, op):
        """Helper to map a Cirq Operation to an integer ID."""
        if isinstance(op.gate, cirq.XPowGate): return 1
        if isinstance(op.gate, cirq.YPowGate): return 2
        if isinstance(op.gate, cirq.ZPowGate): return 3
        if isinstance(op.gate, cirq.HPowGate): return 4
        if isinstance(op.gate, cirq.CNotPowGate): return 5
        return 6 

    def _get_obs(self, circuit=None):
        target_circuit = circuit if circuit is not None else self.current_circuit
        
        # 1. Quantum State Component
        if target_circuit is None or not target_circuit.all_operations():
            state_obs = np.zeros((2 * self.n_qubits,), dtype=np.float32)
            structure_obs = np.zeros((self.max_gates,), dtype=np.int32)
        else:
            all_qubits = cirq.LineQubit.range(self.n_qubits)
            obs_vals = self.simulator.simulate_expectation_values(
                target_circuit,
                observables=[cirq.X(q) for q in all_qubits] + [cirq.Y(q) for q in all_qubits],
                qubit_order=all_qubits
            )
            state_obs = np.array(obs_vals).real.astype(np.float32)

            # 2. Structure Component
            structure_obs = np.zeros((self.max_gates,), dtype=np.int32)
            for i, op in enumerate(target_circuit.all_operations()):
                if i >= self.max_gates: break
                structure_obs[i] = self._encode_gate(op)
        
        return {
            "projected_state": state_obs,
            "gate_structure": structure_obs
        }

    def step(self, action):
        if self.current_circuit is None:
            return self._get_obs(), 0.0, True, False, {}

        ops = list(self.current_circuit.all_operations())
        noisy_ops = []
        applied_errors = []

        # --- BUDGET CONSTRAINT LOGIC ---
        # 1. Identify the 'bids' (actions) for each gate.
        # 2. Keep only the Top-K highest bids (strongest attacks).
        # 3. Mask the rest to 0 (No Attack).
        
        # Ensure we don't crash if circuit is smaller than budget
        valid_gate_count = min(len(ops), self.max_gates)
        effective_action = np.zeros(valid_gate_count, dtype=int)

        if valid_gate_count > 0:
            raw_action = action[:valid_gate_count]
            # Get indices of the top K actions
            # argsort sorts ascending, so we take the last K indices
            budget = min(self.max_concurrent_attacks, valid_gate_count)
            top_k_indices = np.argsort(raw_action)[-budget:]
            
            # Apply the actions ONLY at these indices
            effective_action[top_k_indices] = raw_action[top_k_indices]

        # --- EXECUTE NOISY CIRCUIT ---
        for i in range(valid_gate_count):
            op = ops[i]
            noisy_ops.append(op)
            
            error_idx = effective_action[i]
            # Clip safety
            error_idx = max(0, min(error_idx, self.num_error_levels - 1))
            
            rate = self.all_error_rates[error_idx]
            
            # Only append noise channel if rate > 0
            if rate > 0.0:
                applied_errors.append(rate)
                for q in op.qubits:
                    noisy_ops.append(cirq.DepolarizingChannel(rate).on(q))

        noisy_circuit = cirq.Circuit(noisy_ops)
        
        # --- FIDELITY CALCULATION ---
        all_qubits = cirq.LineQubit.range(self.n_qubits)
        result = self.simulator.simulate(noisy_circuit, qubit_order=all_qubits)
        rho = result.final_density_matrix
        fidelity = np.real(np.vdot(self.target_state, rho @ self.target_state))
        
        # Reward: Minimizing fidelity
        reward = 1.0 - fidelity

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        
        info = {
            'fidelity': fidelity, 
            'mean_error_rate': np.mean(applied_errors) if applied_errors else 0.0
        }
        
        # Return observation of the CLEAN circuit (Saboteur sees what it's attacking)
        return self._get_obs(self.current_circuit), reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    @staticmethod
    def create_observation_from_circuit(circuit: cirq.Circuit, n_qubits: int, max_gates: int = 20) -> dict:
        """Static helper for the ArchitectEnv to generate observations."""
        all_qubits = cirq.LineQubit.range(n_qubits)
        simulator = cirq.DensityMatrixSimulator()
        if not circuit.all_operations():
             state_obs = np.zeros((2 * n_qubits,), dtype=np.float32)
        else:
            obs_vals = simulator.simulate_expectation_values(
                circuit,
                observables=[cirq.X(q) for q in all_qubits] + [cirq.Y(q) for q in all_qubits],
                qubit_order=all_qubits
            )
            state_obs = np.array(obs_vals).real.astype(np.float32)

        structure_obs = np.zeros((max_gates,), dtype=np.int32)
        for i, op in enumerate(circuit.all_operations()):
            if i >= max_gates: break
            val = 6
            if isinstance(op.gate, cirq.XPowGate): val = 1
            elif isinstance(op.gate, cirq.YPowGate): val = 2
            elif isinstance(op.gate, cirq.ZPowGate): val = 3
            elif isinstance(op.gate, cirq.HPowGate): val = 4
            elif isinstance(op.gate, cirq.CNotPowGate): val = 5
            structure_obs[i] = val
            
        return {
            "projected_state": state_obs,
            "gate_structure": structure_obs
        }

# Alias
SaboteurEnv = SaboteurMultiGateEnv


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
                low=0, high=2,  # 0 = empty, 1 = single-qubit, 2 = two-qubit
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
                        # Store Z expectation value twice for shape compatibility
                        projected.append(np.real(z_exp))
                        projected.append(0.0)  # Padding for 2*n_qubits shape
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