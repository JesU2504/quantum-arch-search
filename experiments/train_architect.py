"""
Baseline Architect Training Script.

This script trains a baseline architect agent to learn a quantum circuit that
prepares a target state or implements a target gate.

Target type and task mode are configured centrally via experiments/config.py:
- TARGET_TYPE: 'toffoli' (default) or 'ghz'
- TASK_MODE: 'state_preparation' (default) or 'unitary_preparation'

See experiments/config.py for detailed documentation on configuration options.
"""

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
from qas_gym.utils import save_circuit
from qas_gym.utils import verify_toffoli_unitary, get_toffoli_unitary, get_ideal_unitary

class ChampionCircuitCallback(BaseCallback):
    """
    A callback to save the best circuit found during training.
    It also logs progress and collects fidelity data for plotting.
    
    Champion circuit logic:
    - Triggers if any circuit achieves fidelity > 0 (saves at least one circuit
      if any rollout is better than zero).
    - Checks ALL parallel environments for champions, not just the first one.
    - If no circuit is saved, logs the reason at the end of training.
    """
    def __init__(self, circuit_filename, verbose=0, print_freq=4096):
        super().__init__(verbose)
        self.fidelities = []
        self.steps = []
        self.best_fidelity = -1.0  # Best fidelity seen (for reporting)
        self.best_saved_fidelity = -1.0  # Best fidelity of circuits actually saved
        self.print_freq = print_freq # Align with n_steps for cleaner logs
        self.last_printed_step = 0
        self.circuit_filename = circuit_filename
        # Optional verification config, set by trainer
        self.verify_unitary: bool = False
        self.n_qubits: int | None = None
        self.unitary_mode: bool = False
        # Track whether any circuit was ever saved
        self.circuit_saved: bool = False
        self.circuits_evaluated: int = 0

    def _on_step(self) -> bool:
        # Log progress
        if (self.num_timesteps - self.last_printed_step) >= self.print_freq:
            self.last_printed_step = self.num_timesteps
            if self.fidelities:
                # Report the best fidelity seen so far at this print interval
                print(f"Step: {self.num_timesteps}, Best Fidelity So Far: {self.best_fidelity:.4f}")

        # Check for new champion circuit at the end of an episode
        # FIXED: Iterate over ALL environments, not just index 0
        if 'infos' in self.locals and self.locals['infos'] and self.locals['dones'] is not None:
            dones = self.locals['dones']
            infos = self.locals['infos']
            
            for i, done in enumerate(dones):
                if not done:
                    continue
                
                info = infos[i]

                # Determine if we are in unitary mode (set once)
                if not self.unitary_mode and self.verify_unitary:
                    self.unitary_mode = True

                # Retrieve circuit and fidelity directly from the info dict of the completed episode
                candidate_circuit = info.get('circuit')
                # If not present in info, try falling back to env attr (less reliable in vec envs)
                if candidate_circuit is None:
                    try:
                        # Attempt to get from specific env index
                        candidate_circuit = self.training_env.env_method('get_circuit', indices=[i])[0]
                    except Exception:
                        continue

                # Get the fidelity reported by the environment
                candidate_fid = info.get('fidelity', 0.0)

                if candidate_circuit:
                    self.circuits_evaluated += 1
                    
                    if self.verify_unitary and self.n_qubits is not None:
                        # Unitary verification mode
                        try:
                            accuracy, process_fid = verify_toffoli_unitary(candidate_circuit, self.n_qubits, silent=True)
                            
                            # Track for reporting
                            self.fidelities.append(process_fid)
                            self.steps.append(self.num_timesteps)
                            if process_fid > self.best_fidelity:
                                self.best_fidelity = process_fid
                            
                            # Save if: first circuit ever OR improved over last saved fidelity
                            should_save = (self.best_saved_fidelity < 0) or (process_fid > self.best_saved_fidelity)
                            if should_save:
                                print(f"\n--- New Champion Circuit Found at Step {self.num_timesteps} (Env {i}) ---")
                                print(f"Process Fidelity: {process_fid:.6f}")
                                print(candidate_circuit)
                                save_circuit(self.circuit_filename, candidate_circuit)
                                print(f"Saved new champion circuit to {self.circuit_filename}\n")
                                self.best_saved_fidelity = process_fid
                                self.circuit_saved = True
                        except Exception as e:
                            print(f"[Verifier] Skipped unitary verification due to error: {e}")
                    else:
                        # State fidelity mode
                        # Track for reporting
                        if candidate_fid is not None:
                            self.fidelities.append(candidate_fid)
                            self.steps.append(self.num_timesteps)
                            if candidate_fid > self.best_fidelity:
                                self.best_fidelity = candidate_fid
                        
                        # Save if: first circuit ever OR improved over last saved fidelity
                        should_save = (self.best_saved_fidelity < 0) or (candidate_fid is not None and candidate_fid > self.best_saved_fidelity)
                        if should_save and candidate_fid is not None:
                            print(f"\n--- New Champion Circuit Found at Step {self.num_timesteps} (Env {i}) ---")
                            print(f"Fidelity: {candidate_fid:.6f}")
                            print(candidate_circuit)
                            save_circuit(self.circuit_filename, candidate_circuit)
                            print(f"Saved new champion circuit to {self.circuit_filename}\n")
                            self.best_saved_fidelity = candidate_fid
                            self.circuit_saved = True
        return True

    def _verify_champion_unitary(self, circuit: cirq.Circuit, n_qubits: int) -> None:
        """Delegate to shared verifier in utils for Toffoli unitary checks."""
        verify_toffoli_unitary(circuit, n_qubits, silent=False)

    def get_final_report(self) -> str:
        """Generate a final report explaining the training outcome.
        
        Returns:
            A string describing whether a circuit was saved and why/why not.
        """
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("CHAMPION CIRCUIT CALLBACK FINAL REPORT")
        lines.append("=" * 60)
        lines.append(f"Circuits evaluated: {self.circuits_evaluated}")
        lines.append(f"Best fidelity achieved: {self.best_fidelity:.6f}")
        lines.append(f"Circuit saved: {'Yes' if self.circuit_saved else 'No'}")
        
        if not self.circuit_saved:
            lines.append("\n--- Why no circuit was saved ---")
            if self.circuits_evaluated == 0:
                lines.append("  - No circuits were evaluated during training.")
                lines.append("  - Possible causes:")
                lines.append("    * No episode completed successfully")
                lines.append("    * Environment did not produce any champion circuits")
            elif self.best_fidelity < 0:
                lines.append("  - No circuit achieved fidelity >= 0.")
                lines.append("  - The RL agent did not find any valid circuits.")
            else:
                lines.append("  - This is unexpected. Please report this as a bug.")
        else:
            lines.append(f"\nCircuit saved to: {self.circuit_filename}")
        
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

def train_baseline_architect(results_dir, n_qubits, architect_steps, n_steps, include_rotations=False, target_type=None, task_mode=None):
    """
    Trains a baseline architect in a noise-free environment to find a circuit
    for the configured target (default: n-controlled Toffoli gate).

    Target type and task mode can be overridden via arguments, otherwise uses
    the central configuration from experiments/config.py.

    Args:
        results_dir (str): The directory to save models, plots, and circuits.
        n_qubits (int): The number of qubits for this training run.
        architect_steps (int): The total number of timesteps to train the agent.
        n_steps (int): The number of steps to run for each environment per update.
        include_rotations (bool): If True, include parameterized rotation gates
            (Rx, Ry, Rz) in the action space. This enables more expressive circuits
            suitable for VQE-style variational circuits. Default is False for
            backward compatibility with Clifford+T gate set.
    """
    # Use central config with optional overrides
    effective_target = target_type if target_type is not None else config.TARGET_TYPE
    effective_mode = task_mode if task_mode is not None else config.TASK_MODE
    experiment_label = config.get_experiment_label(effective_target, effective_mode)
    
    # Log configuration at the start of the run
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"Target Type: {effective_target}")
    print(f"Task Mode: {effective_mode}")
    print(f"Number of Qubits: {n_qubits}")
    print(f"Rotation Gates: {'Enabled' if include_rotations else 'Disabled'}")
    print(f"Total Training Steps: {architect_steps}")
    print(f"Steps per Update: {n_steps}")
    print(f"Max Circuit Timesteps: {config.MAX_CIRCUIT_TIMESTEPS}")
    print(f"Fidelity Threshold: 1.1 (unreachable; training by improvement)")
    print(f"Results Directory: {results_dir}")
    print("=" * 60 + "\n")
    
    rotation_status = "with rotation gates" if include_rotations else "with Clifford+T gates only"
    print(f"--- Phase 1: Training Baseline Architect for {n_qubits}-qubit {effective_target.title()} ({rotation_status}) ---")
    os.makedirs(results_dir, exist_ok=True)

    # Define file paths based on the results_dir (include experiment label)
    circuit_filename = os.path.join(results_dir, "circuit_vanilla.json")
    plot_filename = os.path.join(results_dir, "architect_training_progress.png")
    fidelities_filename = os.path.join(results_dir, "architect_fidelities.txt")
    steps_filename = os.path.join(results_dir, "architect_steps.txt")
    
    # Get target state using central config
    target_state = config.get_target_state(n_qubits, effective_target)
    
    # Show the gate being learned (without input preparation X gates)
    # This makes it clearer what the agent is trying to learn
    if effective_target == 'toffoli':
        gate_circuit, _ = config.get_target_circuit(n_qubits, effective_target, include_input_prep=False)
        print("\n--- Target Gate (what the agent learns to implement) ---")
        print(gate_circuit)
        if effective_mode == 'state_preparation':
            print("\nNote: You are in state_preparation mode.")
            print("      We train to output the state |11...10⟩, which is the result")
            print("      of applying this gate to input |11...1⟩.")
        elif effective_mode == 'unitary_preparation':
            print("\nNote: You are in unitary_preparation mode.")
            print("      We evaluate fidelity across ALL 2^n computational basis inputs.")
    else:
        # For GHZ, show the full preparation circuit
        target_circuit, _ = config.get_target_circuit(n_qubits, effective_target)
        print("\n--- Target Circuit (for reference) ---")
        print(target_circuit)

    # Use ArchitectEnv for a fair comparison with the adversarial setup
    # The environment will handle creation of qubits, observables, and gates internally.
    # Enforce gate set from config.py
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    action_gates = config.get_action_gates(qubits, include_rotations=include_rotations)
    # Ideal unitary for unitary mode (auto from config for any target)
    ideal_U = None
    if effective_mode == 'unitary_preparation':
        try:
            ideal_U = get_ideal_unitary(n_qubits, effective_target)
        except Exception:
            ideal_U = None
    env = ArchitectEnv(
        target=target_state,
        fidelity_threshold=1.1,
        reward_penalty=0.01,
        max_timesteps=config.MAX_CIRCUIT_TIMESTEPS,
        complexity_penalty_weight=0.01,
        include_rotations=include_rotations,
        action_gates=action_gates,
        task_mode=effective_mode,
        ideal_unitary=ideal_U,
    )

    # Use the centralized agent hyperparameters from the config file
    agent_params = config.AGENT_PARAMS.copy()
    agent_params['n_steps'] = n_steps # Override n_steps with the value from the per-qubit config
    
    model = PPO("MlpPolicy", env=env, **agent_params)
    
    print("\n--- Starting Training ---")
    callback = ChampionCircuitCallback(circuit_filename=circuit_filename)
    # Enable unitary verification when in unitary_preparation mode and Toffoli target
    callback.verify_unitary = (effective_mode == 'unitary_preparation' and effective_target == 'toffoli')
    callback.n_qubits = n_qubits
    model.learn(total_timesteps=architect_steps, callback=callback)
    print("\n--- Training Finished ---")

    # --- Final processing ---
    # Print detailed final report from callback
    print(callback.get_final_report())
    
    if os.path.exists(circuit_filename):
        print(f"Final vanilla circuit saved in {circuit_filename}")
    else:
        print("No circuit file was created.")
        print("Check the final report above for details on why no circuit was saved.")

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

    plt.title(f"Baseline Architect Training Progress ({n_qubits}-Qubit Toffoli Gate, {rotation_status})")
    plt.xlabel("Training Steps")
    plt.ylabel("Fidelity")
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=-0.05, top=1.05)
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Training plot saved to {plot_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a baseline architect agent")
    parser.add_argument('--n-qubits', type=int, default=4, help='Number of qubits (default: 4)')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--include-rotations', action='store_true',
                        help='Include parameterized rotation gates (Rx, Ry, Rz) in action space')
    args = parser.parse_args()
    
    params = config.EXPERIMENT_PARAMS[args.n_qubits]
    train_baseline_architect(
        results_dir=args.results_dir,
        n_qubits=args.n_qubits, 
        architect_steps=params["ARCHITECT_STEPS"],
        n_steps=params["ARCHITECT_N_STEPS"],
        include_rotations=args.include_rotations
    )