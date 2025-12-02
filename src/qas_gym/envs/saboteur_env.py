import gymnasium as gym
import numpy as np
import cirq
from gymnasium import spaces

def robust_measure_observables(simulator, circuit, qubits):
    """
    Robustly compute expectation values for X and Y observables.
    Falls back to manual simulation if the density matrix is numerically unstable.
    """
    observables = [cirq.X(q) for q in qubits] + [cirq.Y(q) for q in qubits]
    
    try:
        # Try the fast, strict method first
        return simulator.simulate_expectation_values(
            circuit,
            observables=observables,
            qubit_order=qubits
        )
    except Exception:
        # Fallback: Simulate full density matrix and clean it
        result = simulator.simulate(circuit, qubit_order=qubits)
        rho = result.final_density_matrix
        
        # Force Hermiticity (clean numerical noise)
        rho = 0.5 * (rho + rho.conj().T)
        
        # Calculate Expectation <O> = Tr(rho * O)
        vals = []
        for obs in observables:
            # Get the matrix for the observable in the full Hilbert space
            # Note: This involves matrix multiplication which is slower but safe
            obs_mat = cirq.Circuit(obs).unitary(qubit_order=qubits)
            exp_val = np.real(np.trace(rho @ obs_mat))
            vals.append(exp_val)
        return vals


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
        
        # Penalty coefficient for using strong noise
        self.lambda_penalty = lambda_penalty
        
        # MODIFICATION 2: Define the Attack Budget
        self.max_concurrent_attacks = 3

        # --- Fixed Action Space (Padding) ---
        self.num_error_levels = len(self.all_error_rates)
        self.action_space = spaces.MultiDiscrete([
            self.num_error_levels] * self.max_gates)

        # --- Enriched Observation Space (Dict) ---
        self.observation_space = spaces.Dict({
            "projected_state": spaces.Box(low=-1.0, high=1.0, shape=(2 * self.n_qubits,), dtype=np.float32),
            "gate_structure": spaces.Box(low=0, high=10, shape=(self.max_gates,), dtype=np.int32)
        })

    def set_circuit(self, circuit: cirq.Circuit):
        """Updates the target circuit without changing observation/action shapes."""
        self.current_circuit = circuit

    def _encode_gate(self, op):
        """Helper to map a Cirq Operation to an integer ID.
        
        Gate encoding:
            0: Empty/Unknown
            1: X gate (cirq.XPowGate)
            2: Y gate (cirq.YPowGate)
            3: Z gate (cirq.ZPowGate)
            4: H gate (cirq.HPowGate)
            5: CNOT gate (cirq.CNotPowGate)
            6: Other single-qubit gates
            7: Rx rotation gate
            8: Ry rotation gate
            9: Rz rotation gate
        """
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
        """
        Build a robust observation for the Saboteur given the (clean) circuit.
        
        Returns:
            A dict with:
                - projected_state: <X_i>, <Y_i> for each qubit i
                - gate_structure: integer-encoded gate type for each timestep
        Now uses robust measurement to prevent crashes on invalid density matrices.
        """
        if circuit is None:
            circuit = self.current_circuit

        n_qubits = self.n_qubits
        all_qubits = cirq.LineQubit.range(n_qubits)
        simulator = self.simulator
        
        # 1. Projected State Component (Robust)
        if not circuit.all_operations():
            state_obs = np.zeros((2 * n_qubits,), dtype=np.float32)
        else:
            # Uses the helper function defined above
            obs_vals = robust_measure_observables(simulator, circuit, all_qubits)
            state_obs = np.array(obs_vals).real.astype(np.float32)

        # 2. Gate Structure Component (Encoding)
        structure = np.zeros((self.max_gates,), dtype=np.int32)
        ops = list(circuit.all_operations())
        for i, op in enumerate(ops[:self.max_gates]):
            structure[i] = self._encode_gate(op)

        return {
            "projected_state": state_obs,
            "gate_structure": structure,
        }

    @staticmethod
    def create_observation_from_circuit(circuit: cirq.Circuit, n_qubits: int, max_circuit_timesteps: int | None = None):
        """
        Convenience helper to build the saboteur observation for an arbitrary circuit
        without requiring a full environment to be set up by the caller.
        """
        max_steps = max_circuit_timesteps if max_circuit_timesteps is not None else 20
        # Dummy target_state is unused for observation; length drives qubit count if not passed.
        dummy_target = np.zeros(2 ** n_qubits, dtype=complex)
        env = SaboteurMultiGateEnv(
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
        
        # Reward: trade off damage vs noise usage
        mean_err = np.mean(applied_errors) if applied_errors else 0.0
        reward = (1.0 - fidelity) - self.lambda_penalty * mean_err

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        
        info = {
            'fidelity': fidelity, 
            'mean_error_rate': mean_err
        }
        
        # Return observation of the CLEAN circuit (Saboteur sees what it's attacking)
        return self._get_obs(self.current_circuit), reward, terminated, False, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.
        """
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(self.current_circuit), {}

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
    """
    metadata = {'render_modes': []}

    def __init__(self, target_circuit, target_state, max_concurrent_attacks=2):
        super().__init__()

        self.target_circuit = target_circuit
        self.target_state = target_state
        self.max_concurrent_attacks = max_concurrent_attacks

        # Define action space (for each gate: choose an error rate index)
        self.num_gates = len(list(target_circuit.all_operations()))
        self.num_error_levels = len(self.all_error_rates)
        self.action_space = spaces.MultiDiscrete([self.num_error_levels] * self.num_gates)

        # Observation space: Fidelity and optionally other features
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        # For simplicity, we reuse the MultiGate environment's step logic
        env = SaboteurMultiGateEnv(self.target_circuit, self.target_state)
        obs, reward, terminated, truncated, info = env.step(action)
        return np.array([info['fidelity']], dtype=np.float32), reward, terminated, truncated, info
