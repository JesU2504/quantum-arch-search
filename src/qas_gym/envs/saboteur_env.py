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