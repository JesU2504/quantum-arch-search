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
from typing import Sequence

from experiments import config
from qas_gym.utils import (
    apply_noise,
    fidelity_pure_target,
    randomized_compile,
    is_twirl_op,
    build_frame_twirled_noisy_circuit,
    apply_inverse_pauli_frame_to_target,
)


MITIGATION_NONE = "none"
MITIGATION_TWIRL = "twirl"
MITIGATION_RC_ZNE = "rc_zne"

MITIGATION_VARIANTS: dict[str, tuple[str, ...]] = {
    MITIGATION_NONE: ("untwirled",),
    MITIGATION_TWIRL: ("untwirled", "twirled"),
    MITIGATION_RC_ZNE: ("untwirled", "mitigated"),
}

RC_ZNE_DEFAULT_SCALES: tuple[float, ...] = (1.0, 1.5, 2.0)


def _richardson_extrapolate(scales: Sequence[float], values: Sequence[float]) -> float:
    """Return zero-noise estimate via first-order Richardson extrapolation."""
    if len(scales) != len(values) or not scales:
        raise ValueError("Scale and value lists must match and be non-empty")
    if len(scales) == 1:
        return float(values[0])
    x = np.asarray(scales, dtype=float)
    y = np.asarray(values, dtype=float)
    # Linear fit and evaluate at scale 0
    coeffs = np.polyfit(x, y, deg=1)
    return float(np.polyval(coeffs, 0.0))


def _polyfit_extrapolate(scales: Sequence[float], values: Sequence[float], order: int) -> float:
    """Generic polynomial fit extrapolation evaluated at zero noise."""
    if len(scales) != len(values) or not scales:
        raise ValueError("Scale and value lists must match and be non-empty")
    if len(scales) == 1:
        return float(values[0])
    order = max(1, min(order, len(scales) - 1))
    x = np.asarray(scales, dtype=float)
    y = np.asarray(values, dtype=float)
    coeffs = np.polyfit(x, y, deg=order)
    return float(np.polyval(coeffs, 0.0))


def _zero_noise_extrapolate(scales: Sequence[float], values: Sequence[float], fit: str = "linear") -> float:
    """
    Flexible zero-noise extrapolation with model selection.

    Args:
        scales: Noise scale factors.
        values: Observed fidelities per scale.
        fit: Extrapolation model ('linear'/'richardson' or 'quadratic').
    """
    mode = (fit or "linear").lower()
    if mode in ("linear", "richardson", "order1"):
        return _richardson_extrapolate(scales, values)
    if mode in ("quadratic", "poly2", "order2", "richardson2"):
        if len(scales) < 3:
            return _richardson_extrapolate(scales, values)
        return _polyfit_extrapolate(scales, values, order=2)
    raise ValueError(f"Unknown rc_zne_fit '{fit}'")


def _scale_noise_kwargs(
    attack_mode: str,
    scale: float,
    *,
    epsilon_overrot: float,
    p_x: float,
    p_y: float,
    p_z: float,
    gamma_amp: float,
    gamma_phase: float,
    p_readout: float,
) -> dict[str, float]:
    """Scale deterministic-noise parameters for zero-noise extrapolation."""

    def _clamp_prob(value: float) -> float:
        return float(np.clip(value, 0.0, 0.999999))

    def _scale_damping(prob: float, factor: float) -> float:
        prob = _clamp_prob(prob)
        if prob <= 0.0:
            return 0.0
        # Preserve physicality via survival probability scaling
        survival = 1.0 - prob
        scaled = 1.0 - survival**factor
        return _clamp_prob(scaled)

    scale = float(max(scale, 0.0))
    scaled = {
        "epsilon_overrot": float(np.clip(epsilon_overrot * scale, -np.pi, np.pi)) if attack_mode == "over_rotation" else epsilon_overrot,
        "p_x": _clamp_prob(p_x * scale) if attack_mode == "asymmetric_noise" else p_x,
        "p_y": _clamp_prob(p_y * scale) if attack_mode == "asymmetric_noise" else p_y,
        "p_z": _clamp_prob(p_z * scale) if attack_mode == "asymmetric_noise" else p_z,
        "gamma_amp": _scale_damping(gamma_amp, scale) if attack_mode == "amplitude_damping" else gamma_amp,
        "gamma_phase": _scale_damping(gamma_phase, scale) if attack_mode == "phase_damping" else gamma_phase,
        "p_readout": _clamp_prob(p_readout * scale) if attack_mode == "readout" else p_readout,
    }
    return scaled

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
    gamma_amp: float = 0.05,
    gamma_phase: float = 0.05,
    p_readout: float = 0.03,
    mitigation_mode: str = MITIGATION_NONE,
    rc_zne_scales: Sequence[float] = RC_ZNE_DEFAULT_SCALES,
    rc_zne_fit: str = "linear",
    rc_zne_reps: int = 1,
):
    """
    Evaluate circuit robustness under multi-gate attacks sampled from saboteur_agent.
    Returns dict with clean fidelity, mean/min/std attacked fidelity, and all samples.
    If saboteur_agent is None, uses zero-vector (no noise) as fallback.
    Budgeted top-k attack mirrors training (default budget=3).

    rc_zne_fit: Extrapolation model for RC-ZNE ('linear' or 'quadratic').
    rc_zne_reps: Number of RC draws to average per noise scale before extrapolation.
    """
    from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
    qubits = sorted(list(circuit.all_qubits()))
    deterministic_modes = {"over_rotation", "asymmetric_noise", "amplitude_damping", "phase_damping", "readout"}
    rc_scales = tuple(float(s) for s in rc_zne_scales if s is not None and float(s) >= 0.0)
    if mitigation_mode == MITIGATION_RC_ZNE and len(rc_scales) < 2:
        rc_scales = RC_ZNE_DEFAULT_SCALES
    rc_zne_reps = max(1, int(rc_zne_reps) if rc_zne_reps is not None else 1)
    # --- Dimension check ---
    circuit_n_qubits = len(qubits)
    target_dim = target_state.shape[0]
    expected_dim = 2 ** circuit_n_qubits
    if expected_dim != target_dim:
        raise ValueError(
            f"[ERROR] Circuit qubit count ({circuit_n_qubits}) does not match target_state dimension ({target_dim}). "
            f"Expected dimension: {expected_dim}.\n"
            f"Check that the circuit and target_state are for the same number of qubits.\n"
            f"Circuit: {circuit}\nTarget state shape: {target_state.shape}"
        )
    clean_fid = fidelity_pure_target(circuit, target_state, qubits)
    ops = list(circuit.all_operations())
    attacked_vals = []
    rc_zne_scale_values: list[list[float]] = []
    rng = rng or np.random.default_rng()
    all_rates = SaboteurMultiGateEnv.all_error_rates
    max_idx = len(all_rates) - 1
    valid_gate_count = min(len(ops), config.MAX_CIRCUIT_TIMESTEPS)

    for _ in range(samples):
        work_circuit = circuit
        if attack_mode in deterministic_modes:
            if mitigation_mode == MITIGATION_RC_ZNE:
                sample_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
                zne_values: list[float] = []
                for scale in rc_scales:
                    scale_reps: list[float] = []
                    for rep in range(rc_zne_reps):
                        rep_seed = sample_seed + rep
                        local_rng = np.random.default_rng(rep_seed)
                        scaled = _scale_noise_kwargs(
                            attack_mode,
                            scale,
                            epsilon_overrot=epsilon_overrot,
                            p_x=p_x,
                            p_y=p_y,
                            p_z=p_z,
                            gamma_amp=gamma_amp,
                            gamma_phase=gamma_phase,
                            p_readout=p_readout,
                        )
                        noisy_circuit, frame = build_frame_twirled_noisy_circuit(
                            work_circuit,
                            local_rng,
                            attack_mode,
                            epsilon_overrot=scaled["epsilon_overrot"],
                            p_x=scaled["p_x"],
                            p_y=scaled["p_y"],
                            p_z=scaled["p_z"],
                            gamma_amp=scaled["gamma_amp"],
                            gamma_phase=scaled["gamma_phase"],
                            p_readout=scaled["p_readout"],
                        )
                        scale_reps.append(
                            fidelity_pure_target(noisy_circuit, target_state, qubits, frame=frame)
                        )
                    zne_values.append(float(np.mean(scale_reps)))
                rc_zne_scale_values.append(zne_values)
                estimate = _zero_noise_extrapolate(rc_scales, zne_values, fit=rc_zne_fit)
                attacked_vals.append(float(np.clip(estimate, 0.0, 1.0)))
                continue

            twirl_for_mode = mitigation_mode == MITIGATION_TWIRL and attack_mode in deterministic_modes
            if twirl_for_mode:
                twirl_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
                local_rng = np.random.default_rng(twirl_seed)
                noisy_circuit, frame = build_frame_twirled_noisy_circuit(
                    work_circuit,
                    local_rng,
                    attack_mode,
                    epsilon_overrot=epsilon_overrot,
                    p_x=p_x,
                    p_y=p_y,
                    p_z=p_z,
                    gamma_amp=gamma_amp,
                    gamma_phase=gamma_phase,
                    p_readout=p_readout,
                )
                attacked_vals.append(fidelity_pure_target(noisy_circuit, target_state, qubits, frame=frame))
            else:
                noisy_ops = []
                for op in work_circuit.all_operations():
                    noisy_ops.append(op)
                    if is_twirl_op(op):
                        continue
                    for q in op.qubits:
                        if attack_mode == "over_rotation":
                            noisy_ops.append(cirq.rx(epsilon_overrot).on(q))
                        elif attack_mode == "asymmetric_noise":
                            noisy_ops.append(cirq.asymmetric_depolarize(p_x=p_x, p_y=p_y, p_z=p_z).on(q))
                        elif attack_mode == "amplitude_damping":
                            noisy_ops.append(cirq.amplitude_damp(gamma_amp).on(q))
                        elif attack_mode == "phase_damping":
                            noisy_ops.append(cirq.phase_damp(gamma_phase).on(q))
                        elif attack_mode == "readout":
                            noisy_ops.append(cirq.bit_flip(p_readout).on(q))
                noisy_circuit = cirq.Circuit(noisy_ops)
                attacked_vals.append(fidelity_pure_target(noisy_circuit, target_state, qubits))
            continue

        sab_action = None
        ops_full = list(work_circuit.all_operations())
        payload_ops = [op for op in ops_full if not is_twirl_op(op)]
        valid_gate_count = min(len(payload_ops), config.MAX_CIRCUIT_TIMESTEPS)
        budget = min(saboteur_budget, valid_gate_count)

        if attack_mode == "max":
            # Worst-case: assign max error to all gates but honor budgeted subset (tie-break randomly)
            sab_action = np.full(valid_gate_count, max_idx, dtype=int)
            budget = min(saboteur_budget, valid_gate_count)
        elif attack_mode == "policy":
            sab_obs = SaboteurMultiGateEnv.create_observation_from_circuit(
                work_circuit, n_qubits=n_qubits, max_circuit_timesteps=config.MAX_CIRCUIT_TIMESTEPS
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
        payload_pos = 0
        for op in ops_full:
            noisy_ops.append(op)
            if is_twirl_op(op):
                continue
            idx = int(effective_action[payload_pos]) if payload_pos < len(effective_action) else fallback_error_idx
            idx = max(0, min(idx, max_idx))
            error_rate = all_rates[idx]
            for q in op.qubits:
                noisy_ops.append(cirq.DepolarizingChannel(error_rate).on(q))
            payload_pos += 1
        noisy_circuit = cirq.Circuit(noisy_ops)
        attacked_vals.append(fidelity_pure_target(noisy_circuit, target_state, qubits))
    attacked_arr = np.array(attacked_vals)
    result = {
        "clean_fidelity": float(clean_fid),
        "mean_attacked": float(attacked_arr.mean()),
        "min_attacked": float(attacked_arr.min()),
        "std_attacked": float(attacked_arr.std()),
        "samples": attacked_vals
    }
    if mitigation_mode == MITIGATION_RC_ZNE:
        result["rc_zne_scales_used"] = list(rc_scales)
        result["rc_zne_fit"] = rc_zne_fit
        result["rc_zne_reps"] = rc_zne_reps
        if rc_zne_scale_values:
            scale_arr = np.asarray(rc_zne_scale_values, dtype=float)
            result["rc_zne_scale_mean"] = scale_arr.mean(axis=0).tolist()
            result["rc_zne_scale_std"] = scale_arr.std(axis=0).tolist()
    return result

def calculate_fidelity(circuit: cirq.Circuit, target_state: np.ndarray) -> float:
    """Unified fidelity via fidelity_pure_target helper."""
    qubits = sorted(list(circuit.all_qubits())) if circuit.all_qubits() else []
    return fidelity_pure_target(circuit, target_state, qubits) if qubits else 0.0


def _count_gates_and_cnots(circuit: cirq.Circuit) -> tuple[int, int]:
    """Return total gate count and CNOT count for a circuit."""
    ops = list(circuit.all_operations())
    total = len(ops)
    cnots = sum(1 for op in ops if isinstance(op.gate, cirq.CNotPowGate))
    return total, cnots


# --- Move compare_noise_resilience to top-level for import ---
def compare_noise_resilience(
    base_results_dir,
    num_runs,
    n_qubits,
    samples=32,
    saboteur_budget: int = 3,
    seed: int | None = 42,
    logger=None,
    attack_mode: str = "random_high",
    attack_modes: list[str] | None = None,
    epsilon_overrot: float = 0.1,
    p_x: float = 0.05,
    p_y: float = 0.0,
    p_z: float = 0.0,
    gamma_amp: float = 0.05,
    gamma_phase: float = 0.05,
    p_readout: float = 0.03,
    quantumnas_circuit_path: str | None = None,
    ignore_saboteur: bool = False,
    mitigation_mode: str = MITIGATION_NONE,
    rc_zne_scales: Sequence[float] = RC_ZNE_DEFAULT_SCALES,
    rc_zne_fit: str = "linear",
    rc_zne_reps: int = 1,
):
    """
    Aggregate and compare circuit robustness under multi-gate attacks.
    
    Statistical Protocol:
        - Multiple attack samples per circuit
        - Results include mean ± std for attacked fidelities  
        - Error bars and sample size annotations on plots
    - Automatic A/B variant generation when mitigation_mode!='none'
    
    Args:
        base_results_dir: Base directory containing run subdirectories.
        num_runs: Number of experimental runs to aggregate.
        n_qubits: Number of qubits for this analysis.
        samples: Number of saboteur attack samples per circuit.
        attack_modes: Optional list of attack modes to sweep (e.g., ['random_high', 'asymmetric_noise', 'over_rotation', 'amplitude_damping']).
        quantumnas_circuit_path: Optional explicit path to a Cirq JSON circuit to include.
        logger: Optional logger for output.
        attack_mode: Kept for backward compatibility; if attack_modes is provided, this is ignored.
        gamma_amp: Amplitude damping probability for 'amplitude_damping' attack mode.
        gamma_phase: Phase damping (dephasing) probability for 'phase_damping' attack mode.
        p_readout: Readout error probability for 'readout' attack mode (applied as bit-flip channel).
        mitigation_mode: Mitigation strategy for deterministic noise ('none', 'twirl', 'rc_zne').
            'twirl' reproduces Pauli-frame twirling; 'rc_zne' performs randomized compiling plus zero-noise extrapolation.
        rc_zne_scales: Noise scaling factors used when mitigation_mode='rc_zne'.
        rc_zne_fit: Extrapolation model for RC-ZNE ('linear' or 'quadratic').
        rc_zne_reps: Number of randomized-compiling draws averaged per scale.
    """
    attack_modes = attack_modes or [attack_mode]
    primary_mode = attack_modes[0]

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if mitigation_mode not in MITIGATION_VARIANTS:
        log(f"[compare_circuits] Unknown mitigation_mode '{mitigation_mode}', defaulting to 'none'.")
        mitigation_mode = MITIGATION_NONE
    variants_for_mode = MITIGATION_VARIANTS[mitigation_mode]
    secondary_variant = next((v for v in variants_for_mode if v != "untwirled"), None)
    if rc_zne_scales is None:
        rc_zne_scales = RC_ZNE_DEFAULT_SCALES
    else:
        rc_zne_scales = tuple(rc_zne_scales)
    rc_zne_reps = max(1, int(rc_zne_reps) if rc_zne_reps is not None else 1)
    
    log("--- Aggregating and Comparing Circuit Robustness (Multi-Gate Attacks) ---")
    summary_json = os.path.join(base_results_dir, "robust_eval.json")
    samples_csv = os.path.join(base_results_dir, "attacked_fidelity_samples.csv")
    # Use central config to get target state for circuit robustness evaluation
    target_state = config.get_target_state(n_qubits)

    per_mode_metrics = {
        mode: {
            "vanilla": [],
            "robust": [],
            "quantumnas": [],
            "samples": [],
        }
        for mode in attack_modes
    }

    # Attack policy: prefer a trained saboteur; otherwise fall back to the strongest error level.
    try:
        from stable_baselines3 import PPO
        from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
        saboteur_model_path = os.path.join(base_results_dir, "../saboteur/saboteur_trained_on_architect_model.zip")
        saboteur_agent = None
        if not ignore_saboteur and os.path.exists(saboteur_model_path):
            saboteur_agent = PPO.load(saboteur_model_path)
        fallback_error_idx = len(SaboteurMultiGateEnv.all_error_rates) - 1  # worst-case level
        if saboteur_agent is None:
            log(f"[compare_circuits] No saboteur model found or ignored (ignore_saboteur={ignore_saboteur}); using budgeted max-level fallback.")
    except Exception:
        saboteur_agent = None
        fallback_error_idx = 0
        log("[compare_circuits] Failed to load saboteur model; falling back to no-noise attacks.")

    # Shared QuantumNAS circuit fallback: either explicit flag or auto-discovery
    shared_qnas_path = None
    if quantumnas_circuit_path:
        candidate = os.path.expanduser(quantumnas_circuit_path)
        if os.path.exists(candidate):
            shared_qnas_path = os.path.abspath(candidate)
        else:
            log(f"  Warning: Provided QuantumNAS circuit not found at {candidate}")
    else:
        default_candidate = os.path.abspath(os.path.join(base_results_dir, "..", "quantumnas", "circuit_quantumnas.json"))
        if os.path.exists(default_candidate):
            shared_qnas_path = default_candidate

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
            qnas_path_used = None
            if os.path.exists(quantumnas_circuit_file):
                qnas_path_used = quantumnas_circuit_file
            elif shared_qnas_path and os.path.exists(shared_qnas_path):
                qnas_path_used = shared_qnas_path

            if qnas_path_used:
                try:
                    circuit_qnas = load_circuit(qnas_path_used)
                except Exception as exc:
                    log(f"  Warning: Failed to load QuantumNAS circuit from {qnas_path_used}: {exc}")
        except FileNotFoundError as e:
            log(f"  Warning: Could not find circuit files in {run_dir}. Skipping run. Error: {e}")
            continue

        for mode in attack_modes:
            rng = np.random.default_rng(None if seed is None else seed + i)
            for variant in variants_for_mode:
                if variant == "untwirled":
                    eval_mitigation = MITIGATION_NONE
                elif variant == "twirled":
                    eval_mitigation = MITIGATION_TWIRL
                elif variant == "mitigated":
                    eval_mitigation = MITIGATION_RC_ZNE
                else:
                    eval_mitigation = MITIGATION_NONE
                metrics_v = evaluate_multi_gate_attacks(
                    circuit_vanilla, saboteur_agent, target_state, n_qubits,
                    samples=samples, fallback_error_idx=fallback_error_idx,
                    saboteur_budget=saboteur_budget, rng=rng, attack_mode=mode,
                    epsilon_overrot=epsilon_overrot, p_x=p_x, p_y=p_y, p_z=p_z,
                    gamma_amp=gamma_amp, gamma_phase=gamma_phase, p_readout=p_readout,
                    mitigation_mode=eval_mitigation,
                    rc_zne_scales=rc_zne_scales,
                    rc_zne_fit=rc_zne_fit,
                    rc_zne_reps=rc_zne_reps,
                )
                metrics_v["variant"] = variant
                metrics_v["seed"] = i
                metrics_v["circuit_path"] = vanilla_circuit_file
                gates_v, cnots_v = _count_gates_and_cnots(circuit_vanilla)
                metrics_v["gate_count"] = gates_v
                metrics_v["cnot_count"] = cnots_v
                per_mode_metrics[mode]["vanilla"].append(metrics_v)
                for val in metrics_v["samples"]:
                    per_mode_metrics[mode]["samples"].append([i, "vanilla", f"{mode}_{variant}", val])

                metrics_r = evaluate_multi_gate_attacks(
                    circuit_robust, saboteur_agent, target_state, n_qubits,
                    samples=samples, fallback_error_idx=fallback_error_idx,
                    saboteur_budget=saboteur_budget, rng=rng, attack_mode=mode,
                    epsilon_overrot=epsilon_overrot, p_x=p_x, p_y=p_y, p_z=p_z,
                    gamma_amp=gamma_amp, gamma_phase=gamma_phase, p_readout=p_readout,
                    mitigation_mode=eval_mitigation,
                    rc_zne_scales=rc_zne_scales,
                    rc_zne_fit=rc_zne_fit,
                    rc_zne_reps=rc_zne_reps,
                )
                metrics_r["variant"] = variant
                metrics_r["seed"] = i
                metrics_r["circuit_path"] = robust_circuit_file
                gates_r, cnots_r = _count_gates_and_cnots(circuit_robust)
                metrics_r["gate_count"] = gates_r
                metrics_r["cnot_count"] = cnots_r
                per_mode_metrics[mode]["robust"].append(metrics_r)
                for val in metrics_r["samples"]:
                    per_mode_metrics[mode]["samples"].append([i, "robust", f"{mode}_{variant}", val])

                if circuit_qnas is not None:
                    metrics_q = evaluate_multi_gate_attacks(
                        circuit_qnas, saboteur_agent, target_state, n_qubits,
                        samples=samples, fallback_error_idx=fallback_error_idx,
                        saboteur_budget=saboteur_budget, rng=rng, attack_mode=mode,
                        epsilon_overrot=epsilon_overrot, p_x=p_x, p_y=p_y, p_z=p_z,
                        gamma_amp=gamma_amp, gamma_phase=gamma_phase, p_readout=p_readout,
                        mitigation_mode=eval_mitigation,
                        rc_zne_scales=rc_zne_scales,
                        rc_zne_fit=rc_zne_fit,
                        rc_zne_reps=rc_zne_reps,
                    )
                    metrics_q["variant"] = variant
                    metrics_q["seed"] = i
                    metrics_q["circuit_path"] = qnas_path_used or quantumnas_circuit_file
                    gates_q, cnots_q = _count_gates_and_cnots(circuit_qnas)
                    metrics_q["gate_count"] = gates_q
                    metrics_q["cnot_count"] = cnots_q
                    per_mode_metrics[mode]["quantumnas"].append(metrics_q)
                    for val in metrics_q["samples"]:
                        per_mode_metrics[mode]["samples"].append([i, "quantumnas", f"{mode}_{variant}", val])

    aggregated_by_mode = {}
    for mode, buckets in per_mode_metrics.items():
        vm = buckets["vanilla"]
        rm = buckets["robust"]
        qm = buckets["quantumnas"]
        if not vm or not rm:
            continue
        # Separate variants
        def _split_variant(metrics_list, variant):
            return [m["mean_attacked"] for m in metrics_list if m.get("variant") == variant]

        vanilla_means = _split_variant(vm, "untwirled")
        vanilla_sec = _split_variant(vm, secondary_variant) if secondary_variant else []
        robust_means = _split_variant(rm, "untwirled")
        robust_sec = _split_variant(rm, secondary_variant) if secondary_variant else []
        qnas_means = _split_variant(qm, "untwirled") if qm else []
        qnas_sec = _split_variant(qm, secondary_variant) if qm and secondary_variant else []

        vanilla_overall = aggregate_metrics(vanilla_means)
        robust_overall = aggregate_metrics(robust_means)
        qnas_overall = aggregate_metrics(qnas_means) if qnas_means else None
        vanilla_sec_overall = aggregate_metrics(vanilla_sec) if vanilla_sec else None
        robust_sec_overall = aggregate_metrics(robust_sec) if robust_sec else None
        qnas_sec_overall = aggregate_metrics(qnas_sec) if qnas_sec else None

        aggregated_by_mode[mode] = {
            "vanilla": vanilla_overall,
            "robust": robust_overall,
            "quantumnas": qnas_overall,
        }
        if secondary_variant:
            aggregated_by_mode[mode][f"vanilla_{secondary_variant}"] = vanilla_sec_overall
            aggregated_by_mode[mode][f"robust_{secondary_variant}"] = robust_sec_overall
            aggregated_by_mode[mode][f"quantumnas_{secondary_variant}"] = qnas_sec_overall
            if secondary_variant == "twirled":
                aggregated_by_mode[mode]["vanilla_twirl"] = vanilla_sec_overall
                aggregated_by_mode[mode]["robust_twirl"] = robust_sec_overall
                aggregated_by_mode[mode]["quantumnas_twirl"] = qnas_sec_overall

        log(f"\nOverall Statistics [{mode}] (n={len(vanilla_means)} runs, {samples} samples each):")
        log(
            f"  Vanilla: {format_metric_with_error(vanilla_overall['mean'], vanilla_overall['std'], int(vanilla_overall['n']))}"
        )
        if vanilla_sec_overall:
            log(
                f"    └ Mitigated ({secondary_variant}): {format_metric_with_error(vanilla_sec_overall['mean'], vanilla_sec_overall['std'], int(vanilla_sec_overall['n']))}"
            )
        log(
            f"  Robust:  {format_metric_with_error(robust_overall['mean'], robust_overall['std'], int(robust_overall['n']))}"
        )
        if robust_sec_overall:
            log(
                f"    └ Mitigated ({secondary_variant}): {format_metric_with_error(robust_sec_overall['mean'], robust_sec_overall['std'], int(robust_sec_overall['n']))}"
            )
        if qnas_overall:
            log(
                f"  HEA baseline: {format_metric_with_error(qnas_overall['mean'], qnas_overall['std'], int(qnas_overall['n']))}"
            )
            if qnas_sec_overall:
                log(
                    f"    └ Mitigated ({secondary_variant}): {format_metric_with_error(qnas_sec_overall['mean'], qnas_sec_overall['std'], int(qnas_sec_overall['n']))}"
                )

    # Backward-compatible primary mode aliases
    primary_metrics = per_mode_metrics.get(primary_mode, {})
    all_metrics_vanilla = primary_metrics.get("vanilla", [])
    all_metrics_robust = primary_metrics.get("robust", [])
    all_metrics_qnas = primary_metrics.get("quantumnas", [])
    vanilla_overall = aggregated_by_mode.get(primary_mode, {}).get("vanilla")
    robust_overall = aggregated_by_mode.get(primary_mode, {}).get("robust")
    qnas_overall = aggregated_by_mode.get(primary_mode, {}).get("quantumnas")

    # Save results to JSON with statistical info
    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_qubits": n_qubits,
            "num_runs": num_runs,
            "samples_per_circuit": samples,
            "primary_attack_mode": primary_mode,
            "attack_modes": attack_modes,
            "mitigation_mode": mitigation_mode,
            "mitigation_secondary_variant": secondary_variant,
            "rc_zne_scales": list(rc_zne_scales) if mitigation_mode == MITIGATION_RC_ZNE else None,
            "rc_zne_fit": rc_zne_fit if mitigation_mode == MITIGATION_RC_ZNE else None,
            "rc_zne_reps": rc_zne_reps if mitigation_mode == MITIGATION_RC_ZNE else None,
            "statistical_protocol": {
                "aggregation_method": "mean ± std",
                "samples_per_circuit": samples,
            },
        },
        "per_mode": {
            mode: {
                "vanilla": buckets["vanilla"],
                "robust": buckets["robust"],
                "quantumnas": buckets["quantumnas"],
                "aggregated": aggregated_by_mode.get(mode, {}),
            }
            for mode, buckets in per_mode_metrics.items()
        },
        # Backward-compatible aliases for the primary mode
        "vanilla": all_metrics_vanilla,
        "robust": all_metrics_robust,
        "quantumnas": all_metrics_qnas,
    }
    
    if vanilla_overall and robust_overall:
        results_data["aggregated"] = aggregated_by_mode.get(primary_mode, {})
    
    with open(summary_json, "w") as f:
        json.dump(results_data, f, indent=2)
    log(f"\nRobustness summary saved to {summary_json}")

    # Write all sample values to CSV (with attack_mode column)
    with open(samples_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_idx", "circuit_type", "attack_mode", "attacked_fidelity"])
        for mode, buckets in per_mode_metrics.items():
            writer.writerows(buckets["samples"])
    log(f"Attacked fidelity samples saved to {samples_csv}")

    # --- Plot comparison of vanilla vs robust with error bars ---
    try:
        if all_metrics_vanilla and all_metrics_robust:
            def _series(metrics_list, overall_base, overall_variant, variant_label):
                untwirled = {m["seed"]: m for m in metrics_list if m.get("variant") == "untwirled" and "seed" in m}
                mitigated = (
                    {m["seed"]: m for m in metrics_list if m.get("variant") == variant_label and "seed" in m}
                    if variant_label else {}
                )
                seeds_sorted = sorted(untwirled.keys())
                labels_local = [f"seed_{s}" for s in seeds_sorted]
                base_means = [untwirled[s]["mean_attacked"] for s in seeds_sorted]
                base_stds = [untwirled[s]["std_attacked"] for s in seeds_sorted]
                mitigated_raw = [mitigated.get(s, {}).get("mean_attacked", 0.0) for s in seeds_sorted]
                mitigated_stds = [mitigated.get(s, {}).get("std_attacked", 0.0) for s in seeds_sorted]
                variant_gain = [max(0.0, m - b) for m, b in zip(mitigated_raw, base_means)] if variant_label else [0.0] * len(base_means)
                labels_local.append("Mean")
                base_means.append(overall_base["mean"] if overall_base else 0.0)
                base_stds.append(overall_base["std"] if overall_base else 0.0)
                if variant_label:
                    agg_variant = overall_variant["mean"] if overall_variant else 0.0
                    variant_gain.append(max(0.0, agg_variant - base_means[-1]))
                    mitigated_stds.append(overall_variant["std"] if overall_variant else 0.0)
                else:
                    variant_gain.append(0.0)
                    mitigated_stds.append(0.0)
                mitigated_stds = [ts if g > 0 else 0.0 for ts, g in zip(mitigated_stds, variant_gain)]
                has_variant = variant_label is not None and (bool(mitigated) or overall_variant is not None)
                return labels_local, base_means, base_stds, variant_gain, mitigated_stds, has_variant

            agg_primary = aggregated_by_mode.get(primary_mode, {})

            variant_key = secondary_variant
            labels_v, means_v, stds_v, variant_v_gain, variant_v_std, variant_v_present = _series(
                all_metrics_vanilla,
                vanilla_overall,
                agg_primary.get(f"vanilla_{variant_key}") if variant_key else None,
                variant_key,
            )
            labels_r, means_r, stds_r, variant_r_gain, variant_r_std, variant_r_present = _series(
                all_metrics_robust,
                robust_overall,
                agg_primary.get(f"robust_{variant_key}") if variant_key else None,
                variant_key,
            )
            labels_q, means_q, stds_q, variant_q_gain, variant_q_std, variant_q_present = ([], [], [], [], [], False)
            if all_metrics_qnas:
                labels_q, means_q, stds_q, variant_q_gain, variant_q_std, variant_q_present = _series(
                    all_metrics_qnas,
                    qnas_overall,
                    agg_primary.get(f"quantumnas_{variant_key}") if variant_key else None,
                    variant_key,
                )

            labels = labels_v  # assume same seeds across methods
            x = np.arange(len(labels))
            width = 0.22  # gap between method groups

            fig, ax = plt.subplots(figsize=(12, 6))
            used_labels = set()
            err_kw = {"capsize": 4, "capthick": 1.2, "elinewidth": 1.0}

            if variant_key == "twirled":
                variant_display_name = "Twirl gain"
            elif variant_key == "mitigated":
                variant_display_name = (
                    "RC-ZNE gain" if mitigation_mode == MITIGATION_RC_ZNE else "Mitigated gain"
                )
            elif variant_key:
                variant_display_name = f"{variant_key.capitalize()} gain"
            else:
                variant_display_name = "Mitigation gain"

            base_colors = {
                "vanilla": "#54A24B",
                "robust": "#F58518",
                "quantumnas": "#4C78A8",
            }
            variant_colors_by_key = {
                "twirled": {
                    "vanilla": "#a1d99b",
                    "robust": "#fdae6b",
                    "quantumnas": "#9ecae1",
                },
                "mitigated": {
                    "vanilla": "#74c476",
                    "robust": "#fdd0a2",
                    "quantumnas": "#6baed6",
                },
            }
            variant_colors = variant_colors_by_key[variant_key] if variant_key in variant_colors_by_key else {}

            def _plot_method(offset, base_means, base_stds, variant_gain, variant_stds, base_color, variant_color, name, has_variant):
                lb = name if name not in used_labels else None
                ax.bar(
                    x + offset,
                    base_means,
                    width=width,
                    yerr=base_stds,
                    error_kw=err_kw,
                    color=base_color,
                    alpha=0.9,
                    label=lb,
                )
                if lb:
                    used_labels.add(lb)
                if has_variant and any(g > 0 for g in variant_gain):
                    lt = f"{name} ({variant_display_name})"
                    lt = lt if lt not in used_labels else None
                    ax.bar(
                        x + offset,
                        variant_gain,
                        width=width,
                        bottom=base_means,
                        yerr=variant_stds,
                        error_kw=err_kw,
                        color=variant_color,
                        alpha=0.75,
                        label=lt,
                    )
                    if lt:
                        used_labels.add(lt)

            offsets = [-width, 0, width]
            _plot_method(
                offsets[0],
                means_v,
                stds_v,
                variant_v_gain,
                variant_v_std,
                base_colors["vanilla"],
                variant_colors.get("vanilla", base_colors["vanilla"]),
                "RL baseline",
                variant_v_present,
            )
            _plot_method(
                offsets[1],
                means_r,
                stds_r,
                variant_r_gain,
                variant_r_std,
                base_colors["robust"],
                variant_colors.get("robust", base_colors["robust"]),
                "Robust",
                variant_r_present,
            )
            if means_q:
                _plot_method(
                    offsets[2],
                    means_q,
                    stds_q,
                    variant_q_gain,
                    variant_q_std,
                    base_colors["quantumnas"],
                    variant_colors.get("quantumnas", base_colors["quantumnas"]),
                    "HEA baseline",
                    variant_q_present,
                )

            ax.set_ylabel("Mean Attacked Fidelity")
            ax.set_title(
                "Robustness Comparison (Primary Attack Mode)\n"
                f"Mode: {primary_mode} | n={samples} samples/circuit (±1 std)"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            def _max_height(base_vals, gain_vals, has_twirl):
                if not base_vals:
                    return 0.0
                if has_twirl:
                    return max(b + g for b, g in zip(base_vals, gain_vals))
                return max(base_vals)

            max_height = max(
                _max_height(means_v, variant_v_gain, variant_v_present),
                _max_height(means_r, variant_r_gain, variant_r_present),
                _max_height(means_q, variant_q_gain, variant_q_present) if means_q else 0.0,
            )
            ax.set_ylim(0, max(1.05, max_height + 0.1))

            # Annotate gate counts (mean across circuits if available)
            def _mean_count(metrics, key):
                return np.mean([m.get(key, 0) for m in metrics]) if metrics else None
            gate_v = _mean_count(all_metrics_vanilla, "gate_count")
            gate_r = _mean_count(all_metrics_robust, "gate_count")
            gate_q = _mean_count(all_metrics_qnas, "gate_count") if all_metrics_qnas else None
            info = []
            if gate_v is not None:
                info.append(f"RL baseline gates≈{gate_v:.1f}")
            if gate_r is not None:
                info.append(f"Robust gates≈{gate_r:.1f}")
            if gate_q is not None:
                info.append(f"HEA gates≈{gate_q:.1f}")
            if info:
                ax.text(0.02, 0.02, "\n".join(info), transform=ax.transAxes,
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85))

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
        if mitigation_mode == MITIGATION_RC_ZNE:
            hyperparameters.update({
                "rc_zne_scales": list(rc_zne_scales),
                "rc_zne_fit": rc_zne_fit,
                "rc_zne_reps": rc_zne_reps,
            })
        
        aggregated_results = {
            "vanilla_fidelity": vanilla_overall,
            "robust_fidelity": robust_overall,
        }
        if qnas_overall:
            aggregated_results["quantumnas_fidelity"] = qnas_overall
        # Gate/CNOT aggregates
        def _agg_counts(metrics):
            if not metrics:
                return None
            gates = [m.get("gate_count", 0) for m in metrics]
            cnots = [m.get("cnot_count", 0) for m in metrics]
            return {
                "gate_count_mean": float(np.mean(gates)),
                "gate_count_std": float(np.std(gates)),
                "cnot_count_mean": float(np.mean(cnots)),
                "cnot_count_std": float(np.std(cnots)),
            }
        aggregated_results["vanilla_counts"] = _agg_counts(all_metrics_vanilla)
        aggregated_results["robust_counts"] = _agg_counts(all_metrics_robust)
        if all_metrics_qnas:
            aggregated_results["quantumnas_counts"] = _agg_counts(all_metrics_qnas)
        
        summary = create_experiment_summary(
            experiment_name="circuit_robustness_comparison",
            n_seeds=num_runs,
            seeds_used=list(range(num_runs)),
            hyperparameters=hyperparameters,
            aggregated_results=aggregated_results,
            commit_hash=get_git_commit_hash(),
            additional_notes=(
                f"Robustness comparison using {samples} attack samples per circuit. "
                f"Primary attack mode: {primary_mode}. Other evaluated modes: {', '.join(attack_modes)}"
            )
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
    parser.add_argument('--attack-mode', type=str, default='random_high',
                        choices=['max', 'policy', 'random_high', 'over_rotation', 'asymmetric_noise', 'amplitude_damping', 'phase_damping', 'readout'],
                        help=("Primary attack mode: saboteur-based ('max','policy','random_high') or deterministic noise "
                              "('over_rotation','asymmetric_noise','amplitude_damping','phase_damping','readout')."))
    parser.add_argument('--attack-modes', nargs='+', default=None,
                        help=("Optional list of attack modes to sweep; first entry is used for plotting/back-compat. "
                              "Examples: random_high max asymmetric_noise over_rotation amplitude_damping phase_damping readout"))
    parser.add_argument('--epsilon-overrot', type=float, default=0.1, help='Over-rotation angle (radians) if attack-mode=over_rotation')
    parser.add_argument('--p-x', type=float, default=0.05, help='Asymmetric noise p_x if attack-mode=asymmetric_noise')
    parser.add_argument('--p-y', type=float, default=0.0, help='Asymmetric noise p_y if attack-mode=asymmetric_noise')
    parser.add_argument('--p-z', type=float, default=0.0, help='Asymmetric noise p_z if attack-mode=asymmetric_noise')
    parser.add_argument('--gamma-amp', type=float, default=0.05, help='Amplitude damping probability if attack-mode=amplitude_damping')
    parser.add_argument('--gamma-phase', type=float, default=0.05, help='Phase damping probability if attack-mode=phase_damping')
    parser.add_argument('--p-readout', type=float, default=0.03, help='Readout bit-flip probability if attack-mode=readout')
    parser.add_argument('--quantumnas-circuit', type=str, default=None,
                        help='Optional path to a Cirq JSON QuantumNAS circuit. If omitted, looks under ../quantumnas/.')
    parser.add_argument('--ignore-saboteur', action='store_true', help='Skip loading saboteur policy and use non-policy attacks.')
    parser.add_argument('--mitigation-mode', choices=[MITIGATION_NONE, MITIGATION_TWIRL, MITIGATION_RC_ZNE], default=MITIGATION_NONE,
                        help="Mitigation strategy applied during evaluation ('none', 'twirl', 'rc_zne').")
    parser.add_argument('--rc-zne-scales', type=float, nargs='+', default=None,
                        help='Noise scale factors for rc_zne mitigation (default: 1.0 1.5 2.0).')
    parser.add_argument('--rc-zne-fit', type=str, default="linear", choices=["linear", "quadratic"],
                        help="Extrapolation model for rc_zne ('linear' or 'quadratic').")
    parser.add_argument('--rc-zne-reps', type=int, default=1,
                        help='Number of RC draws averaged per scale for rc_zne extrapolation (default: 1).')
    args = parser.parse_args()

    compare_noise_resilience(
        base_results_dir=args.base_results_dir,
        num_runs=args.num_runs,
        n_qubits=args.n_qubits,
        samples=args.samples,
        attack_mode=args.attack_mode,
        attack_modes=args.attack_modes,
        epsilon_overrot=args.epsilon_overrot,
        p_x=args.p_x,
        p_y=args.p_y,
        p_z=args.p_z,
        gamma_amp=args.gamma_amp,
        gamma_phase=args.gamma_phase,
        p_readout=args.p_readout,
        quantumnas_circuit_path=args.quantumnas_circuit,
        ignore_saboteur=args.ignore_saboteur,
        mitigation_mode=args.mitigation_mode,
        rc_zne_scales=args.rc_zne_scales,
        rc_zne_fit=args.rc_zne_fit,
        rc_zne_reps=args.rc_zne_reps,
    )
