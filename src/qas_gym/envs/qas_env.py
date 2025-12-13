import sys
from contextlib import closing
from ..utils import (
    get_default_gates, get_default_observables, fidelity_pure_target,
    is_rotation_gate, get_rotation_gate_info, create_rotation_gate,
    count_rotation_gates, serialize_circuit_with_rotations
)
from io import StringIO
from typing import List, Optional, Dict, Any

import cirq
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class QuantumArchSearchEnv(gym.Env):
    """
    Quantum Architecture Search Environment.
    
    This environment supports both fixed Clifford/T gates and parameterized
    rotation gates (Rx, Ry, Rz) for circuit construction.
    
    When `include_rotations=True`, the action space includes rotation gates
    that are initially set with a default angle. The rotation angles can be
    modified using `set_rotation_angle()` or by providing angles through
    the action using a hybrid action space.
    
    Attributes:
        target: Target quantum state vector.
        fidelity_threshold: Fidelity threshold for successful circuit.
        max_timesteps: Maximum gates per episode.
        qubits: List of qubits for the circuit.
        action_gates: List of available gate operations.
        circuit_gates: Current circuit gate sequence.
        rotation_params: Dictionary mapping gate index to rotation angle.
        include_rotations: Whether rotation gates are included.
    
    Action Space:
        Discrete: Each action corresponds to a gate in `action_gates`.
        When rotation gates are included:
        - Actions for fixed gates apply the gate as-is
        - Actions for rotation gates use the current rotation angle
          (default or set via `set_rotation_angle`)
    
    See Also:
        VQEArchitectEnv - For VQE-specific architecture search with full
                         rotation angle optimization.
    """
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
            complexity_penalty_weight=0.0,
            include_rotations: bool = False,
            default_rotation_angle: float = np.pi / 4,
            task_mode: Optional[str] = None,
        ideal_unitary: Optional[np.ndarray] = None,
        # Reward shaping for STOP behavior and per-step cost
        stop_success_bonus: float = 0.1,
        stop_failure_penalty: float = -0.05,
        per_step_penalty: float = -0.01,
    ):
        """
        Initialize the Quantum Architecture Search Environment.
        
        Args:
            target: Target quantum state vector.
            fidelity_threshold: Fidelity threshold for termination.
            reward_penalty: Penalty for invalid actions.
            max_timesteps: Maximum gates per episode.
            qubits: Optional list of qubits (auto-determined from target if None).
            state_observables: Optional list of observables for observation.
            action_gates: Optional custom list of gate operations.
            complexity_penalty_weight: Weight for circuit complexity penalty.
            include_rotations: If True, include Rx, Ry, Rz gates in action space.
                This enables more expressive circuits suitable for VQE-like tasks.
            default_rotation_angle: Default angle for rotation gates (radians).
        """
        super(QuantumArchSearchEnv, self).__init__()

        self.target = target
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.complexity_penalty_weight = complexity_penalty_weight
        self._max_episode_steps = max_timesteps
        self.include_rotations = include_rotations
        self.default_rotation_angle = default_rotation_angle
        self.task_mode = task_mode or 'state_preparation'
        self.ideal_unitary = ideal_unitary
        # Reward shaping params
        self.stop_success_bonus = stop_success_bonus
        self.stop_failure_penalty = stop_failure_penalty
        self.per_step_penalty = per_step_penalty

        # --- Initialize Qubits, Observables, and Gates ---
        # This logic must come before defining the observation and action spaces.
        if qubits is None:
            n_qubits = int(np.log2(len(target)))
            qubits = cirq.LineQubit.range(n_qubits)
        if state_observables is None:
            state_observables = get_default_observables(qubits)
        action_gates = action_gates if action_gates is not None else get_default_gates(
            qubits, include_rotations=include_rotations
        )

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
        # Add a special STOP action so the architect can choose to finish
        # the circuit before reaching max_timesteps. The STOP action index
        # equals len(action_gates). When chosen, no gate is appended and
        # the environment terminates (agent-controlled early stopping).
        self.stop_action_index = len(action_gates)
        self.action_space = spaces.Discrete(n=len(action_gates) + 1)

        # Track rotation parameters for the current episode
        self.rotation_params: Dict[int, float] = {}
        
        # Build rotation gate mapping for quick lookup
        self._rotation_action_indices = []
        for i, gate_op in enumerate(self.action_gates):
            if is_rotation_gate(gate_op.gate):
                self._rotation_action_indices.append(i)

    def __str__(self):
        gates_str = ', '.join(gate.__str__() for gate in self.action_gates)
        obs_str = ', '.join(ob.__str__() for ob in self.state_observables)
        return (
            f"QuantumArchSearch-v0(Qubits={len(self.qubits)}, Target={self.target}, "
            f"Gates={gates_str}, Observables={obs_str}, IncludesRotations={self.include_rotations})"
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.circuit_gates = []
        self.rotation_params = {}
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
        """
        Return the circuit complexity measured as the total number of
        operations (gates) in the circuit. Previously this returned
        the number of Cirq moments (len(circuit)), which can be smaller
        than the actual gate count when multiple operations share a
        moment. Using the operation count makes `max_timesteps` a strict
        cap on how many gates an agent may add.
        """
        return len(list(circuit.all_operations()))
    
    def set_rotation_angle(self, gate_index: int, angle: float):
        """
        Set the rotation angle for a specific gate in the current circuit.
        
        This method allows external agents or optimization routines to
        modify rotation angles for parameterized gates.
        
        Args:
            gate_index: Index of the gate in the circuit (0-based).
            angle: New rotation angle in radians.
        
        Raises:
            IndexError: If gate_index is out of bounds.
            ValueError: If the gate at gate_index is not a rotation gate.
        """
        if gate_index < 0 or gate_index >= len(self.circuit_gates):
            raise IndexError(f"Gate index {gate_index} out of bounds for circuit with {len(self.circuit_gates)} gates")
        
        gate_op = self.circuit_gates[gate_index]
        if not is_rotation_gate(gate_op.gate):
            raise ValueError(f"Gate at index {gate_index} is not a rotation gate: {gate_op}")
        
        # Get gate info and create new gate with updated angle
        gate_info = get_rotation_gate_info(gate_op)
        new_gate = create_rotation_gate(gate_info['type'], gate_op.qubits[0], angle)
        self.circuit_gates[gate_index] = new_gate
        self.rotation_params[gate_index] = angle
    
    def get_rotation_angles(self) -> Dict[int, float]:
        """
        Get all rotation angles for the current circuit.
        
        Returns:
            Dictionary mapping gate index to rotation angle for all
            rotation gates in the circuit.
        """
        angles = {}
        for i, gate_op in enumerate(self.circuit_gates):
            gate_info = get_rotation_gate_info(gate_op)
            if gate_info is not None:
                angles[i] = gate_info['angle']
        return angles
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current circuit.
        
        Returns:
            Dictionary with circuit statistics including:
            - total_gates: Total number of gates
            - rotation_counts: Count of each rotation gate type
            - cnot_count: Number of CNOT gates
            - rotation_angles: Angles for all rotation gates
            - serialized: Full serialization with rotation parameters
        """
        circuit = self._get_cirq()
        rotation_counts = count_rotation_gates(circuit)
        cnot_count = sum(1 for op in circuit.all_operations() 
                        if isinstance(op.gate, cirq.CNotPowGate))
        
        return {
            'total_gates': len(list(circuit.all_operations())),
            'rotation_counts': rotation_counts,
            'cnot_count': cnot_count,
            'rotation_angles': self.get_rotation_angles(),
            'serialized': serialize_circuit_with_rotations(circuit)
        }

    def step(self, action):
        # The action from the agent can be a numpy array, so we must convert it to a scalar int for indexing.
        action_idx = int(action)

        # If the agent selected the STOP action, end the episode immediately
        # without appending another gate. This enables variable-length
        # circuits shorter than max_timesteps.
        if action_idx == self.stop_action_index:
            circuit = self._get_cirq()
            observation = self._get_obs()
            current_fidelity = self.get_fidelity(circuit)
            terminated = True
            truncated = False

            # Terminal reward for agent-initiated stop: positive bonus if
            # the fidelity meets threshold, negative penalty otherwise.
            if current_fidelity >= self.fidelity_threshold:
                reward = float(self.stop_success_bonus)
            else:
                reward = float(self.stop_failure_penalty)

            circuit_info = self.get_circuit_info()
            info = {
                'fidelity': current_fidelity,
                'circuit': circuit,
                'rotation_counts': circuit_info['rotation_counts'],
                'rotation_angles': circuit_info['rotation_angles'],
                'total_gates': circuit_info['total_gates'],
                'cnot_count': circuit_info['cnot_count'],
                'stop_reason': 'agent_stop'
            }
            info['moment_count'] = len(circuit)
            info['operation_count'] = len(list(circuit.all_operations()))
            # Debug print to indicate an agent-triggered stop
            print(f"[Env Terminated] agent chose STOP: moments={info['moment_count']}, ops={info['operation_count']}, reward={reward}")
            # Update previous fidelity for reward shaping consistency
            self.previous_final_fidelity = current_fidelity
            return observation, reward, terminated, truncated, info

        # Normal gate action
        action_gate = self.action_gates[action_idx]
        reward_penalty = 0.0
        last_op_on_qubits = next((gate for gate in reversed(self.circuit_gates)
                                  if gate.qubits == action_gate.qubits), None)

        if last_op_on_qubits and action_gate == cirq.inverse(last_op_on_qubits):
            reward_penalty = -0.1

        # For rotation gates, we can optionally sample a new angle
        # Currently, we use the default angle from action_gates
        # This can be extended to support hybrid action spaces in the future
        gate_to_add = action_gate
        
        # Track rotation parameters
        if is_rotation_gate(action_gate.gate):
            gate_info = get_rotation_gate_info(action_gate)
            if gate_info:
                self.rotation_params[len(self.circuit_gates)] = gate_info['angle']

        self.circuit_gates.append(gate_to_add)
        circuit = self._get_cirq()
        observation = self._get_obs()

        # The fidelity is calculated on the clean circuit at each step for reward shaping.
        # In unitary_preparation, we still compute state fidelity for shaping but override the
        # terminal reward with process fidelity against the ideal unitary.
        current_fidelity = self.get_fidelity(circuit)

        terminated = (current_fidelity >= self.fidelity_threshold) or \
                     (self.get_circuit_complexity(circuit) >= self.max_timesteps)
        truncated = False

        # --- Reward Shaping for Architect ---
        # This reward encourages building a high-fidelity circuit.
        # In unitary mode, terminal reward is replaced by unitary process fidelity if available.
        fidelity_delta = current_fidelity - self.previous_final_fidelity
        reward = (0.1 * fidelity_delta) + reward_penalty  # Small shaping reward
        # Per-step complexity / time penalty to encourage shorter circuits
        # This is applied only when the agent actually appends a gate.
        reward += float(self.per_step_penalty)
        if terminated:
            reward -= self.complexity_penalty_weight * self.get_circuit_complexity(circuit)
            # Override terminal reward in unitary mode using process fidelity
            if self.task_mode == 'unitary_preparation' and self.ideal_unitary is not None:
                try:
                    # Assemble learned unitary by simulating all basis inputs
                    dim = 2 ** len(self.qubits)
                    n_qubits = len(self.qubits)
                    columns = []
                    sim = cirq.Simulator()
                    for idx in range(dim):
                        # Cirq uses big-endian ordering: qubit 0 is MSB of state index
                        init_bits = [(idx >> (n_qubits - 1 - b)) & 1 for b in range(n_qubits)]
                        prep_ops = [cirq.X(self.qubits[b]) for b, bit in enumerate(init_bits) if bit == 1]
                        test_circuit = cirq.Circuit()
                        test_circuit.append(prep_ops)
                        test_circuit += circuit
                        result = sim.simulate(test_circuit, qubit_order=self.qubits)
                        out_state = result.final_state_vector
                        columns.append(out_state)
                    from utils.metrics import unitary_from_basis_columns, process_fidelity
                    U_learned = unitary_from_basis_columns(columns)
                    proc_fid = process_fidelity(self.ideal_unitary, U_learned)
                    reward = proc_fid
                except Exception:
                    # If unitary computation fails, keep shaping reward
                    pass

        # Build info dict with rotation gate information
        circuit_info = self.get_circuit_info()
        info = {
            'fidelity': current_fidelity, 
            'circuit': circuit,
            'rotation_counts': circuit_info['rotation_counts'],
            'rotation_angles': circuit_info['rotation_angles'],
            'total_gates': circuit_info['total_gates'],
            'cnot_count': circuit_info['cnot_count']
        }
        # Expose both Cirq moment-count and actual operation (gate) count for
        # downstream analysis and debugging. Historically the env used
        # len(circuit) (moments) which could differ from total operations.
        info['moment_count'] = len(circuit)
        info['operation_count'] = len(list(circuit.all_operations()))
        if terminated and self.task_mode == 'unitary_preparation' and self.ideal_unitary is not None:
            info['process_fidelity'] = reward
        
        if current_fidelity > self.best_fidelity:
            self.best_fidelity = current_fidelity
            self.champion_circuit = circuit
            info['is_champion'] = True
            info['champion_fidelity'] = current_fidelity

        # Debug print on termination to make it explicit when an episode
        # finishes why it finished (fidelity or complexity) and to show
        # the difference between moments and operations.
        if terminated:
            # Be defensive when printing max_timesteps: sometimes callers
            # may pass None or non-int-like values. Show both the int-cast
            # (when possible) and the raw repr for debugging.
            raw_mt = getattr(self, 'max_timesteps', None)
            try:
                mt_val = int(raw_mt) if raw_mt is not None else None
            except Exception:
                mt_val = None

            if mt_val is None:
                mt_str = f"(raw={raw_mt!r})"
            else:
                mt_str = f"{mt_val} (raw={raw_mt!r})"

            print(
                f"[Env Terminated] fidelity={current_fidelity:.6f}, "
                f"moments={info['moment_count']}, ops={info['operation_count']}, "
                f"max_timesteps={mt_str}"
            )

        # Update the previous final fidelity for the next step's reward shaping
        self.previous_final_fidelity = current_fidelity

        return observation, reward, terminated, truncated, info

    def render(self, mode='human', circuit=None):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        circuit = circuit or self._get_cirq()
        outfile.write(f'\n{circuit}\n')
        
        # Add rotation gate information if present
        if self.include_rotations and self.rotation_params:
            outfile.write(f'\nRotation angles: {self.rotation_params}\n')

        if mode != 'human':
            with closing(outfile):
                # Only StringIO (ansi mode) supports getvalue; degrade gracefully.
                return outfile.getvalue() if isinstance(outfile, StringIO) else ''