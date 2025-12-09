"""Robustness evaluation utilities for multiple noise families/budgets."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cirq
import numpy as np

from qas_gym.envs.saboteur_env import SaboteurMultiGateEnv
from qas_gym.utils import fidelity_pure_target


def _apply_noise_family(
    circuit: cirq.Circuit,
    attacked_indices: Iterable[int],
    noise_family: str,
    rate: float,
    noise_kwargs: Optional[Dict] = None,
) -> cirq.Circuit:
    """Return a new circuit with noise applied to the selected gate indices."""
    ops = list(circuit.all_operations())
    attacked_set = set(attacked_indices)
    noisy_ops = []
    for idx, op in enumerate(ops):
        noisy_ops.append(op)
        if idx in attacked_set:
            noisy_ops.extend(
                SaboteurMultiGateEnv._noise_ops_for(
                    rate, op, noise_family, noise_kwargs or {}
                )
            )
    return cirq.Circuit(noisy_ops)


def evaluate_circuit_under_noise(
    circuit: cirq.Circuit,
    target_state,
    noise_family: str,
    rate: float,
    budget: int,
    max_samples: int,
    noise_kwargs: Optional[Dict] = None,
    use_twirl: bool = False,
) -> Dict:
    """Compute clean and attacked fidelities for a circuit under a specific noise model.
    
    Args:
        circuit: The circuit to evaluate.
        target_state: Target quantum state.
        noise_family: Noise family string (e.g., 'depolarizing', 'amplitude_damping').
        rate: Noise rate parameter.
        budget: Number of gates to attack (max_k).
        max_samples: Max number of attack subsets to sample.
        noise_kwargs: Optional dict of noise-specific kwargs.
        use_twirl: If True, apply frame-based twirl sandwich model for deterministic modes.
    
    Returns:
        Dict with clean_fidelity, attacked_mean, attacked_std, etc.
    """
    qubits = sorted(circuit.all_qubits())
    clean_fid = float(fidelity_pure_target(circuit, target_state, qubits))
    ops = list(circuit.all_operations())
    max_k = min(max(budget, 0), len(ops))

    combos: List[Iterable[int]] = []
    for k in range(1, max_k + 1):
        combos.extend(itertools.combinations(range(len(ops)), k))
    if len(combos) > max_samples:
        combos = random.sample(combos, max_samples)

    attacked_vals: List[float] = []
    rng = np.random.default_rng()
    for comb in combos:
        if use_twirl and noise_family in ('over_rotation', 'asymmetric_noise', 'amplitude_damping', 'phase_damping'):
            # Apply frame-based twirl sandwich model for deterministic modes
            from qas_gym.utils import build_frame_twirled_noisy_circuit
            attack_mode_map = {
                'over_rotation': 'over_rotation',
                'asymmetric_noise': 'asymmetric_noise',
                'amplitude_damping': 'amplitude_damping',
                'phase_damping': 'phase_damping',
            }
            attack_mode = attack_mode_map.get(noise_family, 'over_rotation')
            # Map noise_kwargs to build_frame_twirled_noisy_circuit parameters
            twirl_kwargs = {}
            if noise_family == 'over_rotation':
                twirl_kwargs['epsilon_overrot'] = rate
            elif noise_family == 'asymmetric_noise':
                twirl_kwargs.update(noise_kwargs or {})
            elif noise_family == 'amplitude_damping':
                twirl_kwargs['gamma_amp'] = rate
            elif noise_family == 'phase_damping':
                twirl_kwargs['gamma_phase'] = rate
            noisy_circ, _ = build_frame_twirled_noisy_circuit(
                circuit, rng, attack_mode, **twirl_kwargs
            )
        else:
            noisy_circ = _apply_noise_family(
                circuit, comb, noise_family=noise_family, rate=rate, noise_kwargs=noise_kwargs
            )
        attacked_vals.append(fidelity_pure_target(noisy_circ, target_state, qubits))

    if attacked_vals:
        attacked_mean = float(np.mean(attacked_vals))
        attacked_std = float(np.std(attacked_vals, ddof=0))
        attacked_min = float(np.min(attacked_vals))
        attacked_max = float(np.max(attacked_vals))
    else:
        attacked_mean = attacked_std = attacked_min = attacked_max = None

    return {
        "clean_fidelity": clean_fid,
        "attacked_mean": attacked_mean,
        "attacked_std": attacked_std,
        "attacked_min": attacked_min,
        "attacked_max": attacked_max,
        "n_attacks_evaluated": len(attacked_vals),
    }


def sweep_circuit_entries(
    circuit_entries: List[Dict],
    noise_families: List[str],
    budgets: List[int],
    rate: float,
    max_samples: int,
    target_state_fn,
    noise_kwargs: Optional[Dict[str, Dict]] = None,
    use_twirl: bool = False,
):
    """
    Evaluate a collection of circuits across multiple noise families and budgets.

    Args:
        circuit_entries: list of dicts with keys {'group','run','seed','path'}
        noise_families: list of noise family strings
        budgets: list of integer attack budgets
        rate: noise strength applied to each attacked gate
        max_samples: maximum number of attack subsets to sample per circuit/budget
        target_state_fn: callable n_qubits -> target_state vector
        noise_kwargs: optional mapping noise_family -> kwargs dict
        use_twirl: If True, apply frame-based twirl sandwich model for deterministic modes.

    Returns:
        List of result rows (dict) for CSV/JSON consumption.
    """
    rows = []
    for entry in circuit_entries:
        path = Path(entry["path"])
        try:
            circuit = cirq.read_json(json_text=path.read_text())
        except Exception:
            continue
        n_qubits = len(circuit.all_qubits())
        target_state = target_state_fn(n_qubits)

        for family in noise_families:
            family_kwargs = (noise_kwargs or {}).get(family, {})
            for budget in budgets:
                # Generate untwirled variant (always)
                stats_untwirled = evaluate_circuit_under_noise(
                    circuit=circuit,
                    target_state=target_state,
                    noise_family=family,
                    rate=rate,
                    budget=budget,
                    max_samples=max_samples,
                    noise_kwargs=family_kwargs,
                    use_twirl=False,
                )
                rows.append(
                    {
                        "group": entry.get("group"),
                        "run": entry.get("run"),
                        "seed": entry.get("seed"),
                        "noise_family": family,
                        "attack_budget": budget,
                        "noise_rate": rate,
                        "circuit_path": str(path),
                        "variant": "untwirled",
                        **stats_untwirled,
                    }
                )
                
                # Generate twirled variant (for deterministic modes only)
                if family in ('over_rotation', 'asymmetric_noise', 'amplitude_damping', 'phase_damping'):
                    stats_twirled = evaluate_circuit_under_noise(
                        circuit=circuit,
                        target_state=target_state,
                        noise_family=family,
                        rate=rate,
                        budget=budget,
                        max_samples=max_samples,
                        noise_kwargs=family_kwargs,
                        use_twirl=True,
                    )
                    rows.append(
                        {
                            "group": entry.get("group"),
                            "run": entry.get("run"),
                            "seed": entry.get("seed"),
                            "noise_family": family,
                            "attack_budget": budget,
                            "noise_rate": rate,
                            "circuit_path": str(path),
                            "variant": "twirled",
                            **stats_twirled,
                        }
                    )
    return rows
