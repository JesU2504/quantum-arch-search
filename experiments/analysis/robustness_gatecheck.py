#!/usr/bin/env python3
"""
Gatecheck for robustness claims.

Reads robustness_sweep.csv (from run_experiments.py) and enforces simple pass/fail
criteria on attacked fidelity for each noise_family/attack_budget.

Usage:
  python experiments/analysis/robustness_gatecheck.py \
    --sweep results/run_x/analysis/robustness_sweep.csv \
    --min-fidelity 0.95 \
    --min-gap 0.02

Exit code is nonzero if any required check fails.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing robustness sweep CSV at {path}")
    df = pd.read_csv(path)
    required = {"group", "noise_family", "attack_budget", "attacked_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Robustness sweep missing columns: {missing}")
    return df


def summarize_group(df: pd.DataFrame) -> Dict:
    # Aggregates per (group, noise_family, attack_budget)
    grouped = (
        df.groupby(["group", "noise_family", "attack_budget"])["attacked_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped = grouped.rename(columns={"mean": "attacked_mean_mean", "std": "attacked_mean_std"})
    return grouped


def check_thresholds(
    grouped: pd.DataFrame,
    min_fidelity: float,
    min_gap: float,
    baseline_group: str,
    robust_group: str,
) -> List[str]:
    """
    Checks:
      1) Robust group attacked_mean >= min_fidelity
      2) Robust - baseline >= min_gap (when baseline exists)
    """
    failures = []

    # Build quick lookup for baseline means
    baseline = (
        grouped[grouped["group"] == baseline_group]
        .set_index(["noise_family", "attack_budget"])["attacked_mean_mean"]
        .to_dict()
    )

    for _, row in grouped[grouped["group"] == robust_group].iterrows():
        key = (row["noise_family"], row["attack_budget"])
        val = row["attacked_mean_mean"]
        if pd.isna(val):
            failures.append(f"{robust_group} missing attacked_mean for {key}")
            continue
        if val < min_fidelity:
            failures.append(f"{robust_group} attacked_mean {val:.4f} < min_fidelity {min_fidelity:.4f} for {key}")
        base_val = baseline.get(key)
        if base_val is not None and val - base_val < min_gap:
            failures.append(
                f"Gap too small for {key}: {robust_group} {val:.4f} - {baseline_group} {base_val:.4f} < min_gap {min_gap:.4f}"
            )
    return failures


def main():
    parser = argparse.ArgumentParser(description="Gatecheck robustness results.")
    parser.add_argument("--sweep", type=Path, required=True, help="Path to robustness_sweep.csv")
    parser.add_argument("--min-fidelity", type=float, default=0.95, help="Minimum attacked fidelity for robust group")
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.02,
        help="Minimum improvement over baseline (robust - baseline) when baseline exists",
    )
    parser.add_argument("--baseline-group", type=str, default="architect_baseline", help="Baseline group name")
    parser.add_argument("--robust-group", type=str, default="adversarial", help="Robust group name to evaluate")
    args = parser.parse_args()

    df = load_rows(args.sweep)
    grouped = summarize_group(df)
    failures = check_thresholds(
        grouped,
        min_fidelity=args.min_fidelity,
        min_gap=args.min_gap,
        baseline_group=args.baseline_group,
        robust_group=args.robust_group,
    )

    print("\nRobustness Gatecheck Summary")
    print("---------------------------")
    print(grouped.to_string(index=False))

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)
    else:
        print("\nPASS: all checks satisfied.")


if __name__ == "__main__":
    main()
