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
import matplotlib.pyplot as plt
import json
from experiments import config
from qas_gym.utils import get_ghz_state, apply_noise, fidelity_pure_target



# --- Multi-gate saboteur attack evaluation ---
def evaluate_multi_gate_attacks(circuit, saboteur_agent, target_state, n_qubits, samples=32):
    """
    Evaluate circuit robustness under multi-gate attacks sampled from saboteur_agent.
    Returns dict with clean fidelity, mean/min/std attacked fidelity, and all samples.
    If saboteur_agent is None, uses zero-vector (no noise) as fallback.
    """
    from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
    qubits = sorted(list(circuit.all_qubits()))
    # --- Dimension check ---
    # Get state vector size from circuit and target_state
    circuit_n_qubits = len(qubits)
    target_dim = target_state.shape[0]
    expected_dim = 2 ** circuit_n_qubits
    if expected_dim != target_dim:
        raise ValueError(f"[ERROR] Circuit qubit count ({circuit_n_qubits}) does not match target_state dimension ({target_dim}). "
                         f"Expected dimension: {expected_dim}.\n"
                         f"Check that the circuit and target_state are for the same number of qubits.\n"
                         f"Circuit: {circuit}\nTarget state shape: {target_state.shape}")
    clean_fid = fidelity_pure_target(circuit, target_state, qubits)
    ops = list(circuit.all_operations())
    attacked_vals = []
    for _ in range(samples):
        sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(circuit, n_qubits=n_qubits)
        if saboteur_agent is not None:
            try:
                sab_action, _ = saboteur_agent.predict(sab_obs, deterministic=False)
            except Exception:
                sab_action = np.zeros(len(ops), dtype=int)
        else:
            sab_action = np.zeros(len(ops), dtype=int)
        # Build noisy circuit
        noisy_ops = []
        all_rates = SaboteurMultiGateEnv.all_error_rates
        max_idx = len(all_rates) - 1
        for i, op in enumerate(ops):
            noisy_ops.append(op)
            idx = int(sab_action[i]) if i < len(sab_action) else 0
            idx = max(0, min(idx, max_idx))
            error_rate = all_rates[idx]
            for q in op.qubits:
                noisy_ops.append(cirq.DepolarizingChannel(error_rate).on(q))
        noisy_circuit = cirq.Circuit(noisy_ops)
        attacked_vals.append(fidelity_pure_target(noisy_circuit, target_state, qubits))
    attacked_arr = np.array(attacked_vals)
    return {
        "clean_fidelity": float(clean_fid),
        "mean_attacked": float(attacked_arr.mean()),
        "min_attacked": float(attacked_arr.min()),
        "std_attacked": float(attacked_arr.std()),
        "samples": attacked_vals
    }

def calculate_fidelity(circuit: cirq.Circuit, target_state: np.ndarray) -> float:
    """Unified fidelity via fidelity_pure_target helper."""
    qubits = sorted(list(circuit.all_qubits())) if circuit.all_qubits() else []
    return fidelity_pure_target(circuit, target_state, qubits) if qubits else 0.0


# --- Move compare_noise_resilience to top-level for import ---
def compare_noise_resilience(base_results_dir, num_runs, n_qubits, samples=32):
    print("--- Aggregating and Comparing Circuit Robustness (Multi-Gate Attacks) ---")
    summary_json = os.path.join(base_results_dir, "robust_eval.json")
    samples_csv = os.path.join(base_results_dir, "attacked_fidelity_samples.csv")
    target_state = get_ghz_state(n_qubits)

    all_metrics_vanilla = []
    all_metrics_robust = []
    all_samples = []

    try:
        from stable_baselines3 import PPO
        from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
        saboteur_model_path = os.path.join(base_results_dir, "../saboteur/saboteur_trained_on_architect_model.zip")
        saboteur_agent = PPO.load(saboteur_model_path) if os.path.exists(saboteur_model_path) else None
    except Exception:
        saboteur_agent = None

    for i in range(num_runs):
        run_dir = os.path.join(base_results_dir, f"run_{i}")
        print(f"\nProcessing Run {i+1}/{num_runs} from {run_dir}")
        vanilla_circuit_file = os.path.join(run_dir, "circuit_vanilla.json")
        robust_circuit_file = os.path.join(run_dir, "circuit_robust.json")

        try:
            from qas_gym.utils import load_circuit
            circuit_vanilla = load_circuit(vanilla_circuit_file)
            circuit_robust = load_circuit(robust_circuit_file)
        except FileNotFoundError as e:
            print(f"  Warning: Could not find circuit files in {run_dir}. Skipping run. Error: {e}")
            continue

        metrics_v = evaluate_multi_gate_attacks(circuit_vanilla, saboteur_agent, target_state, n_qubits, samples=samples)
        metrics_r = evaluate_multi_gate_attacks(circuit_robust, saboteur_agent, target_state, n_qubits, samples=samples)
        all_metrics_vanilla.append(metrics_v)
        all_metrics_robust.append(metrics_r)
        for val in metrics_v["samples"]:
            all_samples.append([i, "vanilla", val])
        for val in metrics_r["samples"]:
            all_samples.append([i, "robust", val])

    with open(summary_json, "w") as f:
        json.dump({
            "vanilla": all_metrics_vanilla,
            "robust": all_metrics_robust
        }, f, indent=2)
    print(f"\nRobustness summary saved to {summary_json}")

    import csv
    with open(samples_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_idx", "circuit_type", "attacked_fidelity"])
        writer.writerows(all_samples)
    print(f"Attacked fidelity samples saved to {samples_csv}")

    print("--- Robustness Comparison Finished ---")

    # --- Plot comparison of vanilla vs robust ---
    try:
        import matplotlib.pyplot as plt
        # Plot mean attacked fidelity for vanilla and robust
        means_v = [m["mean_attacked"] for m in all_metrics_vanilla]
        means_r = [m["mean_attacked"] for m in all_metrics_robust]
        labels = [f"Run {i+1}" for i in range(len(means_v))]
        x = np.arange(len(means_v))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, means_v, width, label="Vanilla", color="tab:blue")
        ax.bar(x + width/2, means_r, width, label="Robust", color="tab:orange")
        ax.set_ylabel("Mean Attacked Fidelity")
        ax.set_title("Robustness Comparison: Vanilla vs Robust Circuits")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(base_results_dir, "robustness_comparison.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[compare_circuits] Saved comparison plot to {out_path}")
    except Exception as e:
        print(f"[compare_circuits] Failed to plot comparison: {e}")


    print("--- Aggregating and Comparing Circuit Robustness (Multi-Gate Attacks) ---")
    summary_json = os.path.join(base_results_dir, "robust_eval.json")
    samples_csv = os.path.join(base_results_dir, "attacked_fidelity_samples.csv")
    target_state = get_ghz_state(n_qubits)

    all_metrics_vanilla = []
    all_metrics_robust = []
    all_samples = []

    # Try to load saboteur agent if available
    try:
        from stable_baselines3 import PPO
        from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
        saboteur_model_path = os.path.join(base_results_dir, "../saboteur/saboteur_trained_on_architect_model.zip")
        saboteur_agent = PPO.load(saboteur_model_path) if os.path.exists(saboteur_model_path) else None
    except Exception:
        saboteur_agent = None

    for i in range(num_runs):
        run_dir = os.path.join(base_results_dir, f"run_{i}")
        print(f"\nProcessing Run {i+1}/{num_runs} from {run_dir}")
        vanilla_circuit_file = os.path.join(run_dir, "circuit_vanilla.json")
        robust_circuit_file = os.path.join(run_dir, "circuit_robust.json")

        try:
            from qas_gym.utils import load_circuit
            circuit_vanilla = load_circuit(vanilla_circuit_file)
            circuit_robust = load_circuit(robust_circuit_file)
        except FileNotFoundError as e:
            print(f"  Warning: Could not find circuit files in {run_dir}. Skipping run. Error: {e}")
            continue

        metrics_v = evaluate_multi_gate_attacks(circuit_vanilla, saboteur_agent, target_state, n_qubits, samples=32)
        metrics_r = evaluate_multi_gate_attacks(circuit_robust, saboteur_agent, target_state, n_qubits, samples=32)
        all_metrics_vanilla.append(metrics_v)
        all_metrics_robust.append(metrics_r)
        # Save all samples for CSV
        for val in metrics_v["samples"]:
            all_samples.append([i, "vanilla", val])
        for val in metrics_r["samples"]:
            all_samples.append([i, "robust", val])

    # Write summary JSON
    with open(summary_json, "w") as f:
        json.dump({
            "vanilla": all_metrics_vanilla,
            "robust": all_metrics_robust
        }, f, indent=2)
    print(f"\nRobustness summary saved to {summary_json}")

    # Write all sample values to CSV
    import csv
    with open(samples_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_idx", "circuit_type", "attacked_fidelity"])
        writer.writerows(all_samples)
    print(f"Attacked fidelity samples saved to {samples_csv}")

    print("--- Robustness Comparison Finished ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare circuit robustness under multi-gate saboteur attacks.")
    parser.add_argument('--base-results-dir', type=str, required=True, help='Base directory containing run subdirectories')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of experimental runs to aggregate (default: 3)')
    parser.add_argument('--n-qubits', type=int, required=True, help='Number of qubits for this analysis')
    parser.add_argument('--samples', type=int, default=32, help='Number of saboteur attack samples per circuit')
    args = parser.parse_args()

    compare_noise_resilience(
        base_results_dir=args.base_results_dir,
        num_runs=args.num_runs,
        n_qubits=args.n_qubits,
        samples=args.samples
    )
