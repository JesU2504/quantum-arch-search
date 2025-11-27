"""
ArchitectAgent: RL agent wrapper for quantum architecture search.

See ExpPlan.md, Implementation notes section.
This module provides wrappers and utilities for training RL agents
(e.g., PPO from Stable-Baselines3) on quantum architecture search tasks.

The agent learns to design quantum circuits that achieve target states
with high fidelity while minimizing circuit complexity.
"""

from typing import Optional, Dict, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class ArchitectAgent:
    """
    Wrapper for RL agents in quantum architecture search.

    Provides a unified interface for training and evaluating agents
    that design quantum circuits. Supports various RL algorithms
    from Stable-Baselines3.

    Attributes:
        env: The training environment (ArchitectEnv or variants).
        algorithm: RL algorithm name (e.g., 'PPO').
        model: The underlying SB3 model.

    See Also:
        ExpPlan.md - Implementation notes (Parallelization, Logging)
    """

    def __init__(
        self,
        env,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        **kwargs
    ):
        """
        Initialize the ArchitectAgent.

        Args:
            env: The training environment.
            algorithm: RL algorithm to use (default: 'PPO').
            policy: Policy architecture (default: 'MlpPolicy').
            learning_rate: Learning rate (default: 3e-4).
            n_steps: Steps per update (default: 2048).
            **kwargs: Additional algorithm arguments.
        """
        self.env = env
        self.algorithm = algorithm
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.kwargs = kwargs
        self.model = None

        # Initialize the SB3 model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Stable-Baselines3 model."""
        if self.algorithm == "PPO":
            self.model = PPO(
                policy=self.policy,
                env=self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                verbose=0,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train(self, total_timesteps: int, callback=None, log_interval: int = 10):
        """
        Train the agent.

        Args:
            total_timesteps: Total training timesteps.
            callback: Optional callback for logging/checkpoints.
            log_interval: Logging frequency.

        Returns:
            The trained model.
        """
        if self.model is None:
            self._initialize_model()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
        )
        return self.model

    def predict(self, observation, deterministic: bool = True):
        """
        Get action from the agent.

        Args:
            observation: Current environment observation.
            deterministic: Whether to use deterministic policy.

        Returns:
            Tuple of (action, state).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call train() or load() first.")
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        """
        Save the agent model.

        Args:
            path: Path to save the model.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Nothing to save.")
        self.model.save(path)

    def load(self, path: str):
        """
        Load a saved agent model.

        Args:
            path: Path to the saved model.

        Returns:
            The loaded agent.
        """
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        return self

    def evaluate(self, env=None, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the agent on the environment.

        Args:
            env: Evaluation environment (uses training env if None).
            n_episodes: Number of evaluation episodes.

        Returns:
            Dict with evaluation metrics (mean_fidelity, mean_gates, etc.).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call train() or load() first.")

        eval_env = env if env is not None else self.env

        fidelities = []
        gate_counts = []
        cnot_counts = []
        depths = []
        rewards = []

        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

            # Collect final metrics from info
            fidelities.append(info.get("fidelity", 0.0))
            gate_counts.append(info.get("total_gates", 0))
            cnot_counts.append(info.get("cnot_count", 0))
            depths.append(info.get("depth", 0))
            rewards.append(episode_reward)

        return {
            "mean_fidelity": np.mean(fidelities),
            "std_fidelity": np.std(fidelities),
            "mean_gates": np.mean(gate_counts),
            "mean_cnots": np.mean(cnot_counts),
            "mean_depth": np.mean(depths),
            "mean_reward": np.mean(rewards),
            "fidelities": fidelities,
            "gate_counts": gate_counts,
            "cnot_counts": cnot_counts,
        }


class MetricsCallback(BaseCallback):
    """
    Callback for logging training metrics.

    Logs fidelity, gate count, and CNOT count during training
    as specified in ExpPlan.md Implementation notes.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.fidelities = []
        self.gate_counts = []
        self.cnot_counts = []

    def _on_step(self) -> bool:
        """Collect metrics at each step."""
        # Get info from the environment
        infos = self.locals.get("infos", [])
        for info in infos:
            if "fidelity" in info:
                self.fidelities.append(info["fidelity"])
            if "total_gates" in info:
                self.gate_counts.append(info["total_gates"])
            if "cnot_count" in info:
                self.cnot_counts.append(info["cnot_count"])
        return True

    def get_metrics(self) -> Dict[str, list]:
        """Return collected metrics."""
        return {
            "fidelities": self.fidelities,
            "gate_counts": self.gate_counts,
            "cnot_counts": self.cnot_counts,
        }
