import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='QuantumArchSearch-v0',
    entry_point='qas_gym.envs:QuantumArchSearchEnv',
    max_episode_steps=20,
)

register(id='Saboteur-v0',
         entry_point='qas_gym.envs:SaboteurEnv')

register(
    id='Architect-v0',
    entry_point='qas_gym.envs:ArchitectEnv',
    max_episode_steps=20,
)
