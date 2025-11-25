import os
import cirq
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from collections import Counter
from qas_gym.envs.saboteur_env import SaboteurEnv
from experiments import config

def analyze_saboteur_policy(model_path, circuit_path, plot_path, title_prefix, n_qubits):
    """
    Loads a trained saboteur agent and analyzes its policy to see if it has
    learned a non-random attack strategy.

    Args:
        model_path (str): Path to the saboteur .zip model file.
        circuit_path (str): Path to the .json circuit file to test against.
        plot_path (str): Path to save the output plot.
        title_prefix (str): A prefix for the plot title (e.g., "Vanilla" or "Robust").
        n_qubits (int): The number of qubits for this analysis.
    """
    print(f"\n--- Analyzing Saboteur Policy for: {title_prefix} ---")
    print(f"Model: {model_path}")
    print(f"Circuit: {circuit_path}")

    # --- Load Agent and Circuit ---
    try:
        saboteur_agent = PPO.load(model_path)
        print(f"Loaded saboteur agent from {model_path}")

        from qas_gym.utils import load_circuit
        circuit = load_circuit(circuit_path)
        print(f"Loaded circuit to analyze from {circuit_path}")
        print("Circuit operations:")
        gate_labels = [str(op) for op in circuit.all_operations()]
        for i, label in enumerate(gate_labels):
            print(f"  Index {i}: {label}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please run the training scripts first.")
        return
    
    if not gate_labels:
        print("Circuit is empty, nothing to analyze.")
        return

    # --- Gather Predictions ---
    print(f"\nGathering {config.N_PREDICTIONS} predictions from the saboteur...")
    # The observation for the saboteur is constant for a fixed circuit.
    # We use a static method on the environment class to generate it.
    observation = SaboteurEnv.create_observation_from_circuit(circuit, n_qubits=n_qubits)

    actions = []
    for _ in range(config.N_PREDICTIONS):
        action, _ = saboteur_agent.predict(observation, deterministic=True)
        # The action is MultiDiscrete, so we take the first part (gate index)
        actions.append(action[0])

    action_counts = Counter(actions)
    
    # --- Visualize the Policy ---
    gate_indices = list(range(len(gate_labels)))
    attack_counts = [action_counts.get(i, 0) for i in gate_indices]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(gate_indices, attack_counts, color='crimson')

    plt.title(f"Saboteur Attack Policy on {title_prefix} Circuit (n={config.N_PREDICTIONS})", fontsize=16)
    plt.xlabel("Gate in Circuit", fontsize=12)
    plt.ylabel("Number of Times Attacked", fontsize=12)
    plt.xticks(ticks=gate_indices, labels=gate_labels, rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nSaboteur policy analysis plot saved to {plot_path}")
    print("--- Analysis Finished ---")


if __name__ == "__main__":
    # Default to the first run directory for standalone analysis
    run_dir = os.path.join(config.RESULTS_DIR, "run_0")
    print(f"--- Running Standalone Analysis on {run_dir} ---")
    
    # Analysis 1: Saboteur trained on the vanilla circuit
    analyze_saboteur_policy(
        model_path=os.path.join(run_dir, "saboteur_trained_on_architect_model.zip"),
        circuit_path=os.path.join(run_dir, "circuit_vanilla.json"),
        plot_path=os.path.join(config.RESULTS_DIR, "saboteur_policy_on_vanilla.png"),
        title_prefix="Vanilla",
        n_qubits=4 # Default to 4 for standalone run
    )

    # Analysis 2: The final saboteur from co-evolution, tested on the final robust circuit
    analyze_saboteur_policy(
        model_path=os.path.join(run_dir, "saboteur_adversarial.zip"),
        circuit_path=os.path.join(run_dir, "circuit_robust.json"),
        plot_path=os.path.join(config.RESULTS_DIR, "saboteur_policy_on_robust.png"),
        title_prefix="Robust",
        n_qubits=4 # Default to 4 for standalone run
    )
