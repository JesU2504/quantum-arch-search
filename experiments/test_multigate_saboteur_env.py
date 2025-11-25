import numpy as np
import cirq
from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
from qas_gym.utils import get_ghz_state

# Build a simple 3-qubit GHZ circuit
n_qubits = 3
qubits = cirq.LineQubit.range(n_qubits)
circuit = cirq.Circuit()
circuit.append(cirq.H(qubits[0]))
for i in range(n_qubits-1):
    circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))

# Target GHZ state
target = get_ghz_state(n_qubits)

# Create environment (discrete error levels)
env = SaboteurMultiGateEnv(architect_circuit=circuit, target_state=target, discrete=True)

obs, info = env.reset()
print('Initial obs:', obs)

# Take a random action (vector of error indices)
action = env.action_space.sample()
print('Sampled action:', action)
obs2, reward, terminated, truncated, info = env.step(action)
print('Step result:')
print('  obs:', obs2)
print('  reward:', reward)
print('  terminated:', terminated)
print('  truncated:', truncated)
print('  info:', info)

# Check continuous mode
env_cont = SaboteurMultiGateEnv(architect_circuit=circuit, target_state=target, discrete=False)
obs, info = env_cont.reset()
action = env_cont.action_space.sample()
print('Continuous action:', action)
obs2, reward, terminated, truncated, info = env_cont.step(action)
print('Continuous step info:', info)
