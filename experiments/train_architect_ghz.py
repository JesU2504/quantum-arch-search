import os
import sys

# Add repository root to sys.path for standalone execution
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_root = os.path.join(_repo_root, 'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import cirq
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

from experiments import config
from qas_gym.envs import ArchitectEnv  # Import ArchitectEnv
from qas_gym.utils import get_ghz_state, create_ghz_circuit_and_qubits, save_circuit

class ChampionCircuitCallback(BaseCallback):
    """
    A callback to save the best circuit found during training.
    It also logs progress and collects fidelity data for plotting.
    """
    def __init__(self, circuit_filename, verbose=0, print_freq=4096):
        super().__init__(verbose)
        self.fidelities = []
        self.steps = []
        self.best_fidelity = -1.0
        self.print_freq = print_freq # Align with n_steps for cleaner logs
        self.last_printed_step = 0
        self.circuit_filename = circuit_filename

    def _on_step(self) -> bool:
        # Log progress
        if (self.num_timesteps - self.last_printed_step) >= self.print_freq:
            self.last_printed_step = self.num_timesteps
            if self.fidelities:
                # Report the best fidelity seen so far at this print interval
                print(f"Step: {self.num_timesteps}, Best Fidelity So Far: {self.best_fidelity:.4f}")

        # Check for new champion circuit at the end of an episode
        if 'infos' in self.locals and self.locals['infos'] and self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            if 'fidelity' in info:
                fidelity = info['fidelity']
                self.fidelities.append(fidelity)
                self.steps.append(self.num_timesteps)
                if fidelity > self.best_fidelity:
                    self.best_fidelity = fidelity

            # The environment may set 'is_champion' on the step where the improvement
            # occurred, which can be mid-episode. Relying on the terminal-step 'info'
            # means we can miss champions discovered earlier in the episode. Instead
            # query the env's stored champion directly and save whenever it improves
            # beyond what we've already recorded in this callback.
            try:
                champion_circuit = self.training_env.get_attr('champion_circuit')[0]
                champion_fid = self.training_env.get_attr('best_fidelity')[0]
            except Exception:
                champion_circuit = None
                champion_fid = None

            if champion_circuit and (champion_fid is not None) and (champion_fid > self.best_fidelity):
                self.best_fidelity = champion_fid
                print(f"\n--- New Champion Circuit Found at Step {self.num_timesteps} ---")
                print(f"Fidelity: {champion_fid:.6f}")
                print(champion_circuit)
                # Save champion and also record its fidelity and step so
                # the callback's fidelity trace and plotted 'Best Fidelity'
                # include champions that were discovered mid-episode.
                save_circuit(self.circuit_filename, champion_circuit)
                print(f"Saved new champion circuit to {self.circuit_filename}\n")
                # Append the champion fidelity and current timestep to the
                # recorded fidelities/steps so plotting and fidelity files
                # reflect the saved champion.
                try:
                    # Avoid duplicate entries if the same fidelity was already recorded
                    if not self.fidelities or (self.fidelities and abs(self.fidelities[-1] - champion_fid) > 1e-9):
                        self.fidelities.append(champion_fid)
                        self.steps.append(self.num_timesteps)
                except Exception:
                    # Be conservative: don't break the callback if recording fails
                    pass
        return True

def train_baseline_architect(results_dir, n_qubits, architect_steps, n_steps):
    """
    Trains a baseline architect in a noise-free environment to find a circuit
    for the GHZ state.

    Args:
        results_dir (str): The directory to save models, plots, and circuits.
        n_qubits (int): The number of qubits for this training run.
        architect_steps (int): The total number of timesteps to train the agent.
        n_steps (int): The number of steps to run for each environment per update.
    """
    print(f"Training baseline for {n_qubits} qubits.")
    print("--- Phase 1: Training Baseline Architect for GHZ State ---")
    os.makedirs(results_dir, exist_ok=True)

    # Define file paths based on the results_dir
    circuit_filename = os.path.join(results_dir, "circuit_vanilla.json")
    plot_filename = os.path.join(results_dir, "architect_ghz_training_progress.png")
    fidelities_filename = os.path.join(results_dir, "architect_ghz_fidelities.txt")
    steps_filename = os.path.join(results_dir, "architect_ghz_steps.txt")
    
    target_state = get_ghz_state(n_qubits)
    target_circuit, _ = create_ghz_circuit_and_qubits(n_qubits)
    print("\n--- Target Circuit (for reference) ---")
    print(target_circuit)

    # Use ArchitectEnv for a fair comparison with the adversarial setup
    # The environment will handle creation of qubits, observables, and gates internally.
    env = ArchitectEnv(
        target=target_state,
        # Set threshold > 1.0 to ensure episodes only end when max_timesteps is reached.
        # This forces the agent to find the best possible circuit within the complexity limit.
        fidelity_threshold=1.1,
        reward_penalty=0.01,
        max_timesteps=config.MAX_CIRCUIT_TIMESTEPS,
        complexity_penalty_weight=0.01, # Add complexity penalty for baseline too
    )    

    # Use the centralized agent hyperparameters from the config file
    agent_params = config.AGENT_PARAMS.copy()
    agent_params['n_steps'] = n_steps # Override n_steps with the value from the per-qubit config
    
    model = PPO("MlpPolicy", env=env, **agent_params)
    
    print("\n--- Starting Training ---")
    callback = ChampionCircuitCallback(circuit_filename=circuit_filename)
    model.learn(total_timesteps=architect_steps, callback=callback)
    print("\n--- Training Finished ---")

    # --- Final processing ---
    if os.path.exists(circuit_filename):
        print(f"\nFinal vanilla circuit saved in {circuit_filename}")
    else:
        print("\nNo circuit achieving a fidelity > 0 was found and saved.")

    # Save fidelities for analysis
    with open(fidelities_filename, "w") as f:
        for fidelity in callback.fidelities:
            f.write(f"{fidelity}\n")
    print(f"Fidelity data saved to {fidelities_filename}")
    # Save step indices (timesteps) so plots can be aligned by training timestep
    try:
        with open(steps_filename, "w") as f:
            for s in callback.steps:
                f.write(f"{s}\n")
        print(f"Step indices saved to {steps_filename}")
    except Exception:
        print("Warning: could not save architect step indices (file write failed or steps empty)")

    # Plotting
    plt.figure(figsize=(10, 6))
    if callback.steps:
        # Plot raw episode fidelities as a scatter plot for a clear view of performance
        plt.scatter(callback.steps, callback.fidelities, alpha=0.2, s=10, label='Episode Fidelity')

        # Calculate and plot the cumulative best fidelity over time
        cumulative_best = np.maximum.accumulate(callback.fidelities)
        plt.plot(callback.steps, cumulative_best, 'r-', label='Best Fidelity Found', linewidth=2)

    plt.title(f"Baseline Architect Training Progress ({n_qubits}-Qubit GHZ State)")
    plt.xlabel("Training Steps")
    plt.ylabel("Fidelity")
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=-0.05, top=1.05)
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Training plot saved to {plot_filename}")

if __name__ == "__main__":
    params = config.EXPERIMENT_PARAMS[4] # Default to 4 qubits for standalone run
    train_baseline_architect(results_dir="results", n_qubits=4, 
                             architect_steps=params["ARCHITECT_STEPS"], n_steps=params["ARCHITECT_N_STEPS"])