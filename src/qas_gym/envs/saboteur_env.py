import gymnasium as gym
import numpy as np
import cirq
from gymnasium import spaces

# Default error rates for noise channels (kept for backward compatibility)
DEFAULT_ERROR_RATES = [0.0, 0.005, 0.01, 0.02, 0.05]
SUPPORTED_NOISE_FAMILIES = {"depolarizing", "amplitude_damping", "coherent_overrotation", "readout", "bitflip"}


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
    # MODIFICATION 1: Error rates (includes stronger levels; keep 0.0 as "No Attack").
    all_error_rates = DEFAULT_ERROR_RATES.copy()

    def __init__(
        self,
        architect_circuit,
        target_state,
        max_circuit_timesteps=20,
        discrete=True,
        episode_length=1,
        lambda_penalty=0.5,
        error_rates=None,
        noise_family: str = "depolarizing",
        noise_kwargs: dict | None = None,
        max_concurrent_attacks: int = 5,
        **kwargs
    ):
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
        
        # MODIFICATION 2: Define the Attack Budget (configurable)
        self.max_concurrent_attacks = max_concurrent_attacks

        # Noise configuration
        self.error_rates = list(error_rates) if error_rates is not None else self.all_error_rates
        self.all_error_rates = self.error_rates  # keep alias for backward compatibility
        self.noise_family = noise_family
        self.noise_kwargs = noise_kwargs.copy() if noise_kwargs is not None else {}
        if self.noise_family not in SUPPORTED_NOISE_FAMILIES:
            raise ValueError(f"Unsupported noise_family={noise_family}. Supported: {sorted(SUPPORTED_NOISE_FAMILIES)}")

        # --- Fixed Action Space (Padding) ---
        self.num_error_levels = len(self.error_rates)
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
    def create_observation_from_circuit(
        circuit: cirq.Circuit,
        n_qubits: int,
        max_circuit_timesteps: int | None = None,
        **env_kwargs,
    ):
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
            **env_kwargs,
        )
        return env._get_obs(circuit)

    @staticmethod
    def _noise_ops_for(rate: float, op: cirq.Operation, noise_family: str, noise_kwargs: dict):
        """Return a list of noise operations to apply for a given rate and noise family."""
        if rate <= 0:
            return []
        family = noise_family.lower()
        if family == "depolarizing":
            return [cirq.DepolarizingChannel(rate).on(q) for q in op.qubits]
        if family == "amplitude_damping":
            gamma = min(max(rate, 0.0), 1.0)
            try:
                channel = cirq.amplitude_damp(gamma)
                return [channel.on(q) for q in op.qubits]
            except Exception:
                try:
                    return [cirq.AmplitudeDampingChannel(gamma).on(q) for q in op.qubits]
                except Exception:
                    # Fallback to depolarizing if amplitude damping channel is unavailable
                    return [cirq.DepolarizingChannel(gamma).on(q) for q in op.qubits]
        if family in {"bitflip", "readout"}:
            p = min(max(rate, 0.0), 1.0)
            return [cirq.BitFlipChannel(p).on(q) for q in op.qubits]
        if family == "coherent_overrotation":
            angle_scale = noise_kwargs.get("coherent_angle_scale", np.pi)
            axis = noise_kwargs.get("coherent_axis", "z")
            angle = rate * angle_scale
            gate_fn = {"x": cirq.rx, "y": cirq.ry}.get(axis.lower(), cirq.rz)
            return [gate_fn(angle).on(q) for q in op.qubits]
        # Fallback to depolarizing if an unknown family somehow slips through validation
        return [cirq.DepolarizingChannel(rate).on(q) for q in op.qubits]

    @classmethod
    def build_noisy_circuit(
        cls,
        circuit: cirq.Circuit,
        action,
        error_rates,
        noise_family: str,
        max_concurrent_attacks: int,
        max_gates: int | None = None,
        noise_kwargs: dict | None = None,
    ):
        """Apply the saboteur action to a circuit and return the noisy circuit.

        Returns (noisy_circuit, applied_error_rates, effective_action).
        """
        ops = list(circuit.all_operations())
        max_gates = max_gates if max_gates is not None else len(ops)
        valid_gate_count = min(len(ops), max_gates)
        effective_action = np.zeros(valid_gate_count, dtype=int)

        if valid_gate_count > 0:
            raw_action = np.array(action[:valid_gate_count], dtype=int)
            budget = min(max_concurrent_attacks, valid_gate_count)
            top_k_indices = np.argsort(raw_action)[-budget:]
            effective_action[top_k_indices] = raw_action[top_k_indices]

        noisy_ops = []
        applied_rates = []
        rates = np.array(error_rates, dtype=float)
        for i in range(valid_gate_count):
            op = ops[i]
            noisy_ops.append(op)

            idx = int(np.clip(effective_action[i], 0, len(rates) - 1))
            rate = float(rates[idx])
            noise_ops = cls._noise_ops_for(rate, op, noise_family, noise_kwargs or {})
            if noise_ops:
                applied_rates.append(rate)
                noisy_ops.extend(noise_ops)

        noisy_circuit = cirq.Circuit(noisy_ops)
        return noisy_circuit, applied_rates, effective_action

    def step(self, action):
        if self.current_circuit is None:
            return self._get_obs(), 0.0, True, False, {}

        noisy_circuit, applied_errors, _ = self.build_noisy_circuit(
            circuit=self.current_circuit,
            action=action,
            error_rates=self.error_rates,
            noise_family=self.noise_family,
            max_concurrent_attacks=self.max_concurrent_attacks,
            max_gates=self.max_gates,
            noise_kwargs=self.noise_kwargs,
        )
        
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

    def __init__(
        self,
        target_circuit,
        target_state,
        qubits=None,
        max_concurrent_attacks=2,
        max_gates=None,
        error_rates=None,
        noise_family: str = "depolarizing",
        noise_kwargs: dict | None = None,
        lambda_penalty: float = 0.5,
    ):
        super().__init__()

        self.target_circuit = target_circuit
        self.target_state = target_state
        self.max_concurrent_attacks = max_concurrent_attacks
        self.qubits = qubits if qubits is not None else sorted(target_circuit.all_qubits())
        self.max_gates = max_gates if max_gates is not None else len(list(target_circuit.all_operations()))
        self.error_rates = list(error_rates) if error_rates is not None else DEFAULT_ERROR_RATES.copy()
        self.noise_family = noise_family
        self.noise_kwargs = noise_kwargs.copy() if noise_kwargs is not None else {}
        self.lambda_penalty = lambda_penalty

        # Backing SaboteurMultiGateEnv for observations/stepping
        self._multi_env = SaboteurMultiGateEnv(
            architect_circuit=self.target_circuit,
            target_state=self.target_state,
            max_circuit_timesteps=self.max_gates,
            n_qubits=len(self.qubits),
            max_concurrent_attacks=self.max_concurrent_attacks,
            error_rates=self.error_rates,
            noise_family=self.noise_family,
            noise_kwargs=self.noise_kwargs,
            lambda_penalty=self.lambda_penalty,
        )

        # Define action/observation spaces to mirror the underlying env
        self.action_space = self._multi_env.action_space
        self.observation_space = self._multi_env.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._multi_env.set_circuit(self.target_circuit)
        return self._multi_env.reset(seed=seed, options=options)

    def step(self, action):
        self._multi_env.set_circuit(self.target_circuit)
        return self._multi_env.step(action)

    def apply_noise(self, circuit=None, action=None):
        """Apply a specific noise action to the circuit and return (noisy_circuit, num_attacks)."""
        if circuit is None:
            circuit = self.target_circuit
        if action is None:
            action = [0] * self.max_gates
        noisy_circuit, applied_rates, _ = SaboteurMultiGateEnv.build_noisy_circuit(
            circuit=circuit,
            action=action,
            error_rates=self.error_rates,
            noise_family=self.noise_family,
            max_concurrent_attacks=self.max_concurrent_attacks,
            max_gates=self.max_gates,
            noise_kwargs=self.noise_kwargs,
        )
        return noisy_circuit, len(applied_rates)

    def apply_max_noise(self):
        """Apply the strongest available noise to every gate (respecting the budget)."""
        max_idx = len(self.error_rates) - 1
        action = [max_idx] * self.max_gates
        return self.apply_noise(self.target_circuit, action)
