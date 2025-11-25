import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='QuantumArchSearch-v0',
    entry_point='qas_gym.envs:QuantumArchSearchEnv',
    max_episode_steps=20,
)

register(id='BasicTwoQubit-v0',
         entry_point='qas_gym.envs:BasicTwoQubitEnv')

register(id='BasicThreeQubit-v0',
         entry_point='qas_gym.envs:BasicThreeQubitEnv')

register(id='BasicNQubit-v0',
         entry_point='qas_gym.envs:BasicNQubitEnv')

register(id='NoisyTwoQubit-v0',
         entry_point='qas_gym.envs:NoisyTwoQubitEnv')

register(id='NoisyThreeQubit-v0',
         entry_point='qas_gym.envs:NoisyThreeQubitEnv')

register(id='NoisyNQubit-v0',
         entry_point='qas_gym.envs:NoisyNQubitEnv')

register(id='Saboteur-v0',
         entry_point='qas_gym.envs:SaboteurEnv')

register(
    id='Architect-v0',
    entry_point='qas_gym.envs:ArchitectEnv',
    max_episode_steps=20,
)
