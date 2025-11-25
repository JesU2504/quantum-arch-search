import cirq
import numpy as np
from qas_gym.utils import fidelity_pure_target
from qas_gym.envs.qas_env import QuantumArchSearchEnv


def _trace_fidelity(target_state: np.ndarray, circuit: cirq.Circuit, qubits):
    sim = cirq.DensityMatrixSimulator()
    result = sim.simulate(circuit, qubit_order=qubits)
    rho = result.final_density_matrix
    rho = 0.5 * (rho + np.conj(rho).T)
    return float(np.real(np.vdot(target_state, rho @ target_state)))


def build_random_circuit(qubits, depth: int, seed: int = 0):
    rs = np.random.default_rng(seed)
    gates = [cirq.X, cirq.Y, cirq.Z, cirq.H]
    ops = []
    for _ in range(depth):
        g = rs.choice(gates)
        q = rs.choice(qubits)
        ops.append(g(q))
    return cirq.Circuit(ops)


def test_fidelity_helper_matches_trace_form():
    n_qubits = 2
    qubits = list(cirq.LineQubit.range(n_qubits))
    # target Bell state
    circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    target_state = result.final_state_vector

    # build test circuits
    circuits = [cirq.Circuit(), circuit] + [build_random_circuit(qubits, depth=d, seed=d) for d in range(1, 5)]
    for c in circuits:
        f_helper = fidelity_pure_target(c, target_state, qubits) if c.all_operations() else 0.0
        f_trace = _trace_fidelity(target_state, c, qubits) if c.all_operations() else 0.0
        assert abs(f_helper - f_trace) < 1e-10, f"Mismatch: helper={f_helper} trace={f_trace}"


def test_quantum_arch_search_env_uses_helper():
    # Ensure env fidelity equals helper fidelity for a simple circuit
    n_qubits = 2
    qubits = list(cirq.LineQubit.range(n_qubits))
    circuit = cirq.Circuit([cirq.X(qubits[0]), cirq.Y(qubits[1])])
    sim = cirq.Simulator()
    target_state = sim.simulate(cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])).final_state_vector
    env = QuantumArchSearchEnv(target=target_state,
                               fidelity_threshold=0.9,
                               reward_penalty=0.0,
                               max_timesteps=5,
                               qubits=qubits)
    f_env = env.get_fidelity(circuit)
    f_helper = fidelity_pure_target(circuit, target_state, qubits)
    assert abs(f_env - f_helper) < 1e-10
