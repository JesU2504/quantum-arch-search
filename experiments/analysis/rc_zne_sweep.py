#!/usr/bin/env python3
"""
Sweep RC-ZNE hyperparameters (scale sets, fit order, RC repetitions) to hunt for
higher attacked fidelity uplift on a small set of circuits.

Example:
    python experiments/analysis/rc_zne_sweep.py \\
        --baseline-circuit results/run_0/circuit_vanilla.json \\
        --robust-circuit results/run_0/circuit_robust.json \\
        --n-qubits 3 \\
        --attack-mode over_rotation --epsilon-overrot 0.12 \\
        --scale-set 1.0,1.5,2.0 --scale-set 1.0,1.25,1.75,2.25 \\
        --fits linear quadratic --rc-zne-reps 1 3 \\
        --samples 24 --out results/run_0/rc_zne_sweep.csv
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments import config
from experiments.analysis.compare_circuits import (
    MITIGATION_NONE,
    MITIGATION_RC_ZNE,
    RC_ZNE_DEFAULT_SCALES,
    evaluate_multi_gate_attacks,
)
from qas_gym.utils import load_circuit

DEFAULT_SCALE_GRID: list[tuple[float, ...]] = [
    RC_ZNE_DEFAULT_SCALES,
    (1.0, 1.25, 1.75, 2.25),
]


def _parse_scale_set(arg: str) -> tuple[float, ...]:
    vals = tuple(float(p.strip()) for p in arg.split(",") if p.strip())
    if len(vals) < 2:
        raise argparse.ArgumentTypeError("Scale set must have at least two entries, e.g., 1.0,1.5,2.0")
    return vals


def _format_scales(scales: Iterable[float]) -> str:
    return ",".join(f"{s:g}" for s in scales)


def sweep_rc_zne(
    circuits: list[tuple[str, Path]],
    n_qubits: int,
    *,
    samples: int,
    attack_mode: str,
    saboteur_budget: int,
    epsilon_overrot: float,
    p_x: float,
    p_y: float,
    p_z: float,
    gamma_amp: float,
    gamma_phase: float,
    p_readout: float,
    scale_grid: list[tuple[float, ...]],
    fits: list[str],
    reps_grid: list[int],
    seed: int,
) -> pd.DataFrame:
    """Run RC-ZNE sweeps across circuits and return a results DataFrame."""
    target_state = config.get_target_state(n_qubits)
    records: list[dict] = []

    for label, circuit_path in circuits:
        circuit = load_circuit(str(circuit_path))
        # Baseline (no mitigation) for uplift reference
        base_metrics = evaluate_multi_gate_attacks(
            circuit,
            saboteur_agent=None,
            target_state=target_state,
            n_qubits=n_qubits,
            samples=samples,
            saboteur_budget=saboteur_budget,
            rng=np.random.default_rng(seed),
            attack_mode=attack_mode,
            epsilon_overrot=epsilon_overrot,
            p_x=p_x,
            p_y=p_y,
            p_z=p_z,
            gamma_amp=gamma_amp,
            gamma_phase=gamma_phase,
            p_readout=p_readout,
            mitigation_mode=MITIGATION_NONE,
        )
        records.append(
            {
                "circuit": label,
                "mitigation": "none",
                "rc_zne_scales": None,
                "rc_zne_fit": None,
                "rc_zne_reps": None,
                "mean_attacked": base_metrics["mean_attacked"],
                "std_attacked": base_metrics["std_attacked"],
                "clean_fidelity": base_metrics["clean_fidelity"],
                "uplift_vs_baseline": 0.0,
                "rc_zne_scale_mean": None,
                "rc_zne_scale_std": None,
            }
        )

        cfg_idx = 0
        for scales in scale_grid:
            for fit in fits:
                for reps in reps_grid:
                    cfg_seed = seed + cfg_idx
                    cfg_idx += 1
                    metrics = evaluate_multi_gate_attacks(
                        circuit,
                        saboteur_agent=None,
                        target_state=target_state,
                        n_qubits=n_qubits,
                        samples=samples,
                        saboteur_budget=saboteur_budget,
                        rng=np.random.default_rng(cfg_seed),
                        attack_mode=attack_mode,
                        epsilon_overrot=epsilon_overrot,
                        p_x=p_x,
                        p_y=p_y,
                        p_z=p_z,
                        gamma_amp=gamma_amp,
                        gamma_phase=gamma_phase,
                        p_readout=p_readout,
                        mitigation_mode=MITIGATION_RC_ZNE,
                        rc_zne_scales=scales,
                        rc_zne_fit=fit,
                        rc_zne_reps=reps,
                    )
                    records.append(
                        {
                            "circuit": label,
                            "mitigation": "rc_zne",
                            "rc_zne_scales": _format_scales(scales),
                            "rc_zne_fit": fit,
                            "rc_zne_reps": reps,
                            "mean_attacked": metrics["mean_attacked"],
                            "std_attacked": metrics["std_attacked"],
                            "clean_fidelity": metrics["clean_fidelity"],
                            "uplift_vs_baseline": metrics["mean_attacked"] - base_metrics["mean_attacked"],
                            "rc_zne_scale_mean": json.dumps(metrics.get("rc_zne_scale_mean", [])),
                            "rc_zne_scale_std": json.dumps(metrics.get("rc_zne_scale_std", [])),
                        }
                    )
    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(description="Sweep RC-ZNE scale/fit/repetition settings for candidate circuits.")
    parser.add_argument("--baseline-circuit", type=Path, help="Path to baseline circuit JSON.")
    parser.add_argument("--robust-circuit", type=Path, help="Path to robust circuit JSON.")
    parser.add_argument("--quantumnas-circuit", type=Path, help="Path to QuantumNAS circuit JSON.")
    parser.add_argument("--n-qubits", type=int, required=True, help="Number of qubits for the target state.")
    parser.add_argument("--samples", type=int, default=24, help="Saboteur samples per configuration (default: 24).")
    parser.add_argument(
        "--attack-mode",
        type=str,
        default="over_rotation",
        choices=["over_rotation", "asymmetric_noise", "amplitude_damping", "phase_damping", "readout"],
        help="Deterministic noise family to sweep.",
    )
    parser.add_argument("--saboteur-budget", type=int, default=3, help="Budget used for saboteur attacks (default: 3).")
    parser.add_argument("--epsilon-overrot", type=float, default=0.1, help="Over-rotation angle if attack_mode=over_rotation.")
    parser.add_argument("--p-x", type=float, default=0.05, help="Asymmetric noise p_x if attack_mode=asymmetric_noise.")
    parser.add_argument("--p-y", type=float, default=0.0, help="Asymmetric noise p_y if attack_mode=asymmetric_noise.")
    parser.add_argument("--p-z", type=float, default=0.0, help="Asymmetric noise p_z if attack_mode=asymmetric_noise.")
    parser.add_argument("--gamma-amp", type=float, default=0.05, help="Amplitude damping probability if attack_mode=amplitude_damping.")
    parser.add_argument("--gamma-phase", type=float, default=0.05, help="Phase damping probability if attack_mode=phase_damping.")
    parser.add_argument("--p-readout", type=float, default=0.03, help="Readout bit-flip probability if attack_mode=readout.")
    parser.add_argument(
        "--scale-set",
        action="append",
        type=_parse_scale_set,
        default=None,
        help="Noise scale set to evaluate (comma-separated, e.g., 1.0,1.5,2.0). May be repeated.",
    )
    parser.add_argument(
        "--fits",
        nargs="+",
        default=["linear", "quadratic"],
        choices=["linear", "quadratic"],
        help="Extrapolation models to try (default: linear quadratic).",
    )
    parser.add_argument(
        "--rc-zne-reps",
        nargs="+",
        type=int,
        default=[1, 3],
        help="RC draws per scale before extrapolation (default: 1 3).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for reproducibility.")
    parser.add_argument("--out", type=Path, default=Path("results/rc_zne_sweep.csv"), help="Output CSV path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top configs to print per circuit.")
    args = parser.parse_args()

    circuits: list[tuple[str, Path]] = []
    for label, path in (
        ("baseline", args.baseline_circuit),
        ("robust", args.robust_circuit),
        ("quantumnas", args.quantumnas_circuit),
    ):
        if path:
            circuits.append((label, path))
    if not circuits:
        raise SystemExit("Provide at least one circuit via --baseline-circuit/--robust-circuit/--quantumnas-circuit.")

    scale_grid = args.scale_set if args.scale_set is not None else DEFAULT_SCALE_GRID
    reps_grid = [r for r in (args.rc_zne_reps or [1]) if r and r > 0]

    df = sweep_rc_zne(
        circuits=circuits,
        n_qubits=args.n_qubits,
        samples=args.samples,
        attack_mode=args.attack_mode,
        saboteur_budget=args.saboteur_budget,
        epsilon_overrot=args.epsilon_overrot,
        p_x=args.p_x,
        p_y=args.p_y,
        p_z=args.p_z,
        gamma_amp=args.gamma_amp,
        gamma_phase=args.gamma_phase,
        p_readout=args.p_readout,
        scale_grid=scale_grid,
        fits=args.fits,
        reps_grid=reps_grid,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[rc_zne_sweep] Saved sweep results to {args.out} (rows={len(df)})")

    # Print quick leaderboard per circuit
    df_sorted = df[df["mitigation"] == "rc_zne"].sort_values(
        ["circuit", "mean_attacked"], ascending=[True, False]
    )
    for label in df_sorted["circuit"].unique():
        top = df_sorted[df_sorted["circuit"] == label].head(args.top_k)
        print(f"\nTop RC-ZNE configs for {label}:")
        for _, row in top.iterrows():
            print(
                f"  scales={row['rc_zne_scales']} fit={row['rc_zne_fit']} reps={int(row['rc_zne_reps'])} "
                f"mean={row['mean_attacked']:.4f} (uplift {row['uplift_vs_baseline']:+.4f})"
            )


if __name__ == "__main__":
    main()
