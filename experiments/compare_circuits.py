"""
Circuit comparison and robustness analysis for quantum architecture search.

This module evaluates and compares the robustness of vanilla (baseline) and robust 
(adversarially-trained) circuits under multi-gate saboteur attacks.

Statistical Protocol:
    - Multiple attack samples per circuit for robust statistics
    - Results include mean ± std for attacked fidelities
    - Plots show error bars and sample size annotations
    - Summary files include all hyperparameters and aggregation methods
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
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from experiments import config
from qas_gym.utils import get_ghz_state, apply_noise, fidelity_pure_target

# Import statistical utilities
from utils.stats import (
    aggregate_metrics,
    create_experiment_summary,
    save_experiment_summary,
    get_git_commit_hash,
    format_metric_with_error,
)



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
def compare_noise_resilience(base_results_dir, num_runs, n_qubits, samples=32, logger=None):
    """
    Aggregate and compare circuit robustness under multi-gate attacks.
    
    Statistical Protocol:
        - Multiple attack samples per circuit
        - Results include mean ± std for attacked fidelities  
        - Error bars and sample size annotations on plots
    
    Args:
        base_results_dir: Base directory containing run subdirectories.
        num_runs: Number of experimental runs to aggregate.
        n_qubits: Number of qubits for this analysis.
        samples: Number of saboteur attack samples per circuit.
        logger: Optional logger for output.
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log("--- Aggregating and Comparing Circuit Robustness (Multi-Gate Attacks) ---")
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
        log(f"\nProcessing Run {i+1}/{num_runs} from {run_dir}")
        vanilla_circuit_file = os.path.join(run_dir, "circuit_vanilla.json")
        robust_circuit_file = os.path.join(run_dir, "circuit_robust.json")

        try:
            from qas_gym.utils import load_circuit
            circuit_vanilla = load_circuit(vanilla_circuit_file)
            circuit_robust = load_circuit(robust_circuit_file)
        except FileNotFoundError as e:
            log(f"  Warning: Could not find circuit files in {run_dir}. Skipping run. Error: {e}")
            continue

        metrics_v = evaluate_multi_gate_attacks(circuit_vanilla, saboteur_agent, target_state, n_qubits, samples=samples)
        metrics_r = evaluate_multi_gate_attacks(circuit_robust, saboteur_agent, target_state, n_qubits, samples=samples)
        all_metrics_vanilla.append(metrics_v)
        all_metrics_robust.append(metrics_r)
        for val in metrics_v["samples"]:
            all_samples.append([i, "vanilla", val])
        for val in metrics_r["samples"]:
            all_samples.append([i, "robust", val])

    # Compute aggregated statistics across runs
    if all_metrics_vanilla and all_metrics_robust:
        vanilla_means = [m["mean_attacked"] for m in all_metrics_vanilla]
        vanilla_stds = [m["std_attacked"] for m in all_metrics_vanilla]
        robust_means = [m["mean_attacked"] for m in all_metrics_robust]
        robust_stds = [m["std_attacked"] for m in all_metrics_robust]
        
        # Aggregate across runs
        vanilla_overall = aggregate_metrics(vanilla_means)
        robust_overall = aggregate_metrics(robust_means)
        
        log(f"\nOverall Statistics (n={len(vanilla_means)} runs, {samples} samples each):")
        log(f"  Vanilla: {format_metric_with_error(vanilla_overall['mean'], vanilla_overall['std'], vanilla_overall['n'])}")
        log(f"  Robust:  {format_metric_with_error(robust_overall['mean'], robust_overall['std'], robust_overall['n'])}")
    else:
        vanilla_overall = None
        robust_overall = None

    # Save results to JSON with statistical info
    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_qubits": n_qubits,
            "num_runs": num_runs,
            "samples_per_circuit": samples,
            "statistical_protocol": {
                "aggregation_method": "mean ± std",
                "samples_per_circuit": samples,
            },
        },
        "vanilla": all_metrics_vanilla,
        "robust": all_metrics_robust,
    }
    
    if vanilla_overall and robust_overall:
        results_data["aggregated"] = {
            "vanilla": vanilla_overall,
            "robust": robust_overall,
        }
    
    with open(summary_json, "w") as f:
        json.dump(results_data, f, indent=2)
    log(f"\nRobustness summary saved to {summary_json}")

    # Write all sample values to CSV
    with open(samples_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_idx", "circuit_type", "attacked_fidelity"])
        writer.writerows(all_samples)
    log(f"Attacked fidelity samples saved to {samples_csv}")

    # --- Plot comparison of vanilla vs robust with error bars ---
    try:
        if all_metrics_vanilla and all_metrics_robust:
            means_v = [m["mean_attacked"] for m in all_metrics_vanilla]
            stds_v = [m["std_attacked"] for m in all_metrics_vanilla]
            means_r = [m["mean_attacked"] for m in all_metrics_robust]
            stds_r = [m["std_attacked"] for m in all_metrics_robust]
            
            labels = [f"Run {i+1}" for i in range(len(means_v))]
            x = np.arange(len(means_v))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Bar plot with error bars
            bars_v = ax.bar(x - width/2, means_v, width, yerr=stds_v, 
                           label="Vanilla", color="tab:blue", capsize=5, alpha=0.8)
            bars_r = ax.bar(x + width/2, means_r, width, yerr=stds_r,
                           label="Robust", color="tab:orange", capsize=5, alpha=0.8)
            
            ax.set_ylabel("Mean Attacked Fidelity")
            ax.set_title(f"Robustness Comparison: Vanilla vs Robust Circuits\n(n={samples} attack samples per circuit, error bars: ±1 std)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.05)
            
            plt.tight_layout()
            out_path = os.path.join(base_results_dir, "robustness_comparison.png")
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
            log(f"[compare_circuits] Saved comparison plot to {out_path}")
    except Exception as e:
        log(f"[compare_circuits] Failed to plot comparison: {e}")

    # Create experiment summary file
    if vanilla_overall and robust_overall:
        hyperparameters = {
            "n_qubits": n_qubits,
            "num_runs": num_runs,
            "samples_per_circuit": samples,
        }
        
        aggregated_results = {
            "vanilla_fidelity": vanilla_overall,
            "robust_fidelity": robust_overall,
        }
        
        summary = create_experiment_summary(
            experiment_name="circuit_robustness_comparison",
            n_seeds=num_runs,
            seeds_used=list(range(num_runs)),
            hyperparameters=hyperparameters,
            aggregated_results=aggregated_results,
            commit_hash=get_git_commit_hash(),
            additional_notes=f"Robustness comparison using {samples} saboteur attack samples per circuit."
        )
        save_experiment_summary(summary, base_results_dir, 'experiment_summary.json')

    log("--- Robustness Comparison Finished ---")


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
