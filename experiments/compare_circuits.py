"""
Circuit comparison and robustness analysis for quantum architecture search.

This module evaluates and compares the robustness of vanilla (baseline) and robust 
(adversarially-trained) circuits under multi-gate saboteur attacks.

Target type and task mode are configured centrally via experiments/config.py:
- TARGET_TYPE: 'toffoli' (default) or 'ghz'
- TASK_MODE: 'state_preparation' (default) or 'unitary_preparation'

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
from qas_gym.utils import apply_noise, fidelity_pure_target

# Import statistical utilities
from utils.stats import (
    aggregate_metrics,
    create_experiment_summary,
    save_experiment_summary,
    get_git_commit_hash,
    format_metric_with_error,
)



# --- Multi-gate saboteur attack evaluation ---
def evaluate_multi_gate_attacks(
    circuit,
    saboteur_agent,
    target_state,
    n_qubits,
    samples=32,
    fallback_error_idx=0,
    saboteur_budget: int = 3,
    rng: np.random.Generator | None = None,
    attack_mode: str = "max",  # 'max' (worst-case), 'policy' (agent), 'random_high' (high-level random), 'over_rotation', 'asymmetric_noise'
    epsilon_overrot: float = 0.1,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0,
):
    """
    Evaluate circuit robustness under multi-gate attacks sampled from saboteur_agent.
    Returns dict with clean fidelity, mean/min/std attacked fidelity, and all samples.
    If saboteur_agent is None, uses zero-vector (no noise) as fallback.
    Budgeted top-k attack mirrors training (default budget=3).
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
    rng = rng or np.random.default_rng()
    # Precompute static helpers outside the sampling loop
    ops = list(circuit.all_operations())
    all_rates = SaboteurMultiGateEnv.all_error_rates
    max_idx = len(all_rates) - 1
    valid_gate_count = min(len(ops), config.MAX_CIRCUIT_TIMESTEPS)

    for _ in range(samples):
        # Deterministic noise modes bypass saboteur
        if attack_mode in ("over_rotation", "asymmetric_noise"):
            noisy_ops = []
            for op in ops:
                noisy_ops.append(op)
                if attack_mode == "over_rotation":
                    for q in op.qubits:
                        noisy_ops.append(cirq.rx(epsilon_overrot).on(q))
                else:
                    for q in op.qubits:
                        noisy_ops.append(cirq.asymmetric_depolarize(p_x=p_x, p_y=p_y, p_z=p_z).on(q))
            noisy_circuit = cirq.Circuit(noisy_ops)
            attacked_vals.append(fidelity_pure_target(noisy_circuit, target_state, qubits))
            continue

        sab_action = None
        budget = min(saboteur_budget, valid_gate_count)

        if attack_mode == "max":
            # Worst-case: assign max error to all gates but honor budgeted subset (tie-break randomly)
            sab_action = np.full(valid_gate_count, max_idx, dtype=int)
            budget = min(saboteur_budget, valid_gate_count)
        elif attack_mode == "policy":
            sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(
                circuit, n_qubits=n_qubits, max_circuit_timesteps=config.MAX_CIRCUIT_TIMESTEPS
            )
            if saboteur_agent is not None:
                try:
                    sab_action, _ = saboteur_agent.predict(sab_obs, deterministic=False)
                except Exception:
                    sab_action = None
        else:
            # random_high: random from the top error levels
            high_min = max(0, max_idx - 2)
            sab_action = rng.integers(high_min, max_idx + 1, size=valid_gate_count, dtype=int)
            budget = valid_gate_count


        # Fallback if policy failed
        if sab_action is None:
            high_min = max(0, max_idx - 2)
            sab_action = rng.integers(high_min, max_idx + 1, size=valid_gate_count, dtype=int)
            budget = valid_gate_count

        # Budgeted top-k attack (consistent with training)
        raw_action = np.array(sab_action[:valid_gate_count], dtype=int)
        effective_action = np.zeros_like(raw_action)
        if budget > 0 and len(raw_action) > 0:
            if np.all(raw_action == raw_action[0]):
                # If all scores are equal, choose budgeted gates uniformly at random
                budget_indices = rng.choice(len(raw_action), size=budget, replace=False)
                effective_action[budget_indices] = raw_action[budget_indices]
            else:
                top_k_indices = np.argsort(raw_action)[-budget:]
                effective_action[top_k_indices] = raw_action[top_k_indices]

        noisy_ops = []
        for i, op in enumerate(ops):
            noisy_ops.append(op)
            idx = int(effective_action[i]) if i < len(effective_action) else fallback_error_idx
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
def compare_noise_resilience(
    base_results_dir,
    num_runs,
    n_qubits,
    samples=32,
    saboteur_budget: int = 3,
    seed: int | None = 42,
    logger=None,
    attack_mode: str = "max",
    epsilon_overrot: float = 0.1,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0,
):
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
        attack_mode: 'max' (default, worst-case), 'policy' (agent-driven), or 'random_high'.
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log("--- Aggregating and Comparing Circuit Robustness (Multi-Gate Attacks) ---")
    summary_json = os.path.join(base_results_dir, "robust_eval.json")
    samples_csv = os.path.join(base_results_dir, "attacked_fidelity_samples.csv")
    # Use central config to get target state for circuit robustness evaluation
    target_state = config.get_target_state(n_qubits)

    all_metrics_vanilla = []
    all_metrics_robust = []
    all_metrics_qnas = []
    all_samples = []

    # Attack policy: prefer a trained saboteur; otherwise fall back to the strongest error level.
    try:
        from stable_baselines3 import PPO
        from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
        saboteur_model_path = os.path.join(base_results_dir, "../saboteur/saboteur_trained_on_architect_model.zip")
        saboteur_agent = PPO.load(saboteur_model_path) if os.path.exists(saboteur_model_path) else None
        fallback_error_idx = len(SaboteurMultiGateEnv.all_error_rates) - 1  # worst-case level
        if saboteur_agent is None:
            log(f"[compare_circuits] No saboteur model found at {saboteur_model_path}; using budgeted max-level fallback.")
    except Exception:
        saboteur_agent = None
        fallback_error_idx = 0
        log("[compare_circuits] Failed to load saboteur model; falling back to no-noise attacks.")

    for i in range(num_runs):
        run_dir = os.path.join(base_results_dir, f"run_{i}")
        log(f"\nProcessing Run {i+1}/{num_runs} from {run_dir}")
        vanilla_circuit_file = os.path.join(run_dir, "circuit_vanilla.json")
        robust_circuit_file = os.path.join(run_dir, "circuit_robust.json")
        quantumnas_circuit_file = os.path.join(run_dir, "circuit_quantumnas.json")

        try:
            from qas_gym.utils import load_circuit
            circuit_vanilla = load_circuit(vanilla_circuit_file)
            circuit_robust = load_circuit(robust_circuit_file)
            circuit_qnas = None
            if os.path.exists(quantumnas_circuit_file):
                try:
                    circuit_qnas = load_circuit(quantumnas_circuit_file)
                except Exception as exc:
                    log(f"  Warning: Failed to load QuantumNAS circuit in {run_dir}: {exc}")
        except FileNotFoundError as e:
            log(f"  Warning: Could not find circuit files in {run_dir}. Skipping run. Error: {e}")
            continue

        rng = np.random.default_rng(None if seed is None else seed + i)
        metrics_v = evaluate_multi_gate_attacks(
            circuit_vanilla, saboteur_agent, target_state, n_qubits,
            samples=samples, fallback_error_idx=fallback_error_idx,
            saboteur_budget=saboteur_budget, rng=rng, attack_mode=attack_mode,
            epsilon_overrot=epsilon_overrot, p_x=p_x, p_y=p_y, p_z=p_z
        )
        metrics_v["circuit_path"] = vanilla_circuit_file
        metrics_r = evaluate_multi_gate_attacks(
            circuit_robust, saboteur_agent, target_state, n_qubits,
            samples=samples, fallback_error_idx=fallback_error_idx,
            saboteur_budget=saboteur_budget, rng=rng, attack_mode=attack_mode,
            epsilon_overrot=epsilon_overrot, p_x=p_x, p_y=p_y, p_z=p_z
        )
        metrics_r["circuit_path"] = robust_circuit_file
        all_metrics_vanilla.append(metrics_v)
        all_metrics_robust.append(metrics_r)
        for val in metrics_v["samples"]:
            all_samples.append([i, "vanilla", val])
        for val in metrics_r["samples"]:
            all_samples.append([i, "robust", val])
        if circuit_qnas is not None:
            metrics_q = evaluate_multi_gate_attacks(
                circuit_qnas, saboteur_agent, target_state, n_qubits,
                samples=samples, fallback_error_idx=fallback_error_idx,
                saboteur_budget=saboteur_budget, rng=rng, attack_mode=attack_mode,
                epsilon_overrot=epsilon_overrot, p_x=p_x, p_y=p_y, p_z=p_z
            )
            metrics_q["circuit_path"] = quantumnas_circuit_file
            all_metrics_qnas.append(metrics_q)
            for val in metrics_q["samples"]:
                all_samples.append([i, "quantumnas", val])

    # Compute aggregated statistics across runs
    if all_metrics_vanilla and all_metrics_robust:
        vanilla_means = [m["mean_attacked"] for m in all_metrics_vanilla]
        vanilla_stds = [m["std_attacked"] for m in all_metrics_vanilla]
        robust_means = [m["mean_attacked"] for m in all_metrics_robust]
        robust_stds = [m["std_attacked"] for m in all_metrics_robust]
        qnas_means = [m["mean_attacked"] for m in all_metrics_qnas] if all_metrics_qnas else []
        qnas_stds = [m["std_attacked"] for m in all_metrics_qnas] if all_metrics_qnas else []
        
        # Aggregate across runs
        vanilla_overall = aggregate_metrics(vanilla_means)
        robust_overall = aggregate_metrics(robust_means)
        qnas_overall = aggregate_metrics(qnas_means) if qnas_means else None
        
        log(f"\nOverall Statistics (n={len(vanilla_means)} runs, {samples} samples each):")
        log(f"  Vanilla: {format_metric_with_error(vanilla_overall['mean'], vanilla_overall['std'], vanilla_overall['n'])}")
        log(f"  Robust:  {format_metric_with_error(robust_overall['mean'], robust_overall['std'], robust_overall['n'])}")
        if qnas_overall:
            log(f"  QuantumNAS: {format_metric_with_error(qnas_overall['mean'], qnas_overall['std'], qnas_overall['n'])}")
    else:
        vanilla_overall = None
        robust_overall = None
        qnas_overall = None

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
        "quantumnas": all_metrics_qnas,
    }
    
    if vanilla_overall and robust_overall:
        results_data["aggregated"] = {
            "vanilla": vanilla_overall,
            "robust": robust_overall,
            "quantumnas": qnas_overall,
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
            means_q = [m["mean_attacked"] for m in all_metrics_qnas] if all_metrics_qnas else []
            stds_q = [m["std_attacked"] for m in all_metrics_qnas] if all_metrics_qnas else []

            labels = [f"Run {i+1}" for i in range(len(means_v))]
            # Append aggregated mean/std if available so the plot shows both per-run values and the average.
            if vanilla_overall and robust_overall:
                labels.append("Mean")
                means_v.append(vanilla_overall["mean"])
                stds_v.append(vanilla_overall["std"])
                means_r.append(robust_overall["mean"])
                stds_r.append(robust_overall["std"])
                if qnas_overall:
                    means_q.append(qnas_overall["mean"])
                    stds_q.append(qnas_overall["std"])

            x = np.arange(len(labels))
            width = 0.25

            fig, ax = plt.subplots(figsize=(12, 6))

            bars = []
            bars.append(ax.bar(x - width, means_v, width, yerr=stds_v,
                               label="Vanilla", color="tab:blue", capsize=5, alpha=0.8))
            bars.append(ax.bar(x, means_r, width, yerr=stds_r,
                               label="Robust", color="tab:orange", capsize=5, alpha=0.8))
            if means_q:
                bars.append(ax.bar(x + width, means_q, width, yerr=stds_q,
                                   label="QuantumNAS", color="tab:green", capsize=5, alpha=0.8))

            ax.set_ylabel("Mean Attacked Fidelity")
            ax.set_title(f"Robustness Comparison: Vanilla vs Robust vs QuantumNAS\n(n={samples} attack samples per circuit, error bars: ±1 std)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            max_height = max(means_v + means_r + (means_q if means_q else [0]))
            ax.set_ylim(0, max(1.05, max_height + 0.1))

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
        if qnas_overall:
            aggregated_results["quantumnas_fidelity"] = qnas_overall
        
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
    parser.add_argument('--attack-mode', type=str, default='max',
                        choices=['max', 'policy', 'random_high', 'over_rotation', 'asymmetric_noise'],
                        help="Noise/attack mode: 'max'/'policy'/'random_high' saboteur, or 'over_rotation'/'asymmetric_noise' for deterministic noise.")
    parser.add_argument('--epsilon-overrot', type=float, default=0.1, help='Over-rotation angle (radians) if attack-mode=over_rotation')
    parser.add_argument('--p-x', type=float, default=0.05, help='Asymmetric noise p_x if attack-mode=asymmetric_noise')
    parser.add_argument('--p-y', type=float, default=0.0, help='Asymmetric noise p_y if attack-mode=asymmetric_noise')
    parser.add_argument('--p-z', type=float, default=0.0, help='Asymmetric noise p_z if attack-mode=asymmetric_noise')
    args = parser.parse_args()

    compare_noise_resilience(
        base_results_dir=args.base_results_dir,
        num_runs=args.num_runs,
        n_qubits=args.n_qubits,
        samples=args.samples,
        attack_mode=args.attack_mode,
        epsilon_overrot=args.epsilon_overrot,
        p_x=args.p_x,
        p_y=args.p_y,
        p_z=args.p_z,
    )
