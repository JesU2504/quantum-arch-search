#!/usr/bin/env python3
"""
Summarize robustness gains per attack mode from robust_eval.json, including
mitigation (RC-ZNE) variants when present. Optionally compute simple bootstrap
confidence intervals over seeds.

Example:
    python experiments/analysis/robustness_ablation.py \\
        --robust-eval results/ghz_3/compare/robust_eval.json \\
        --out-csv results/ghz_3/compare/robustness_ablation.csv \\
        --bootstrap 2000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _bootstrap_ci(data: np.ndarray, n_samples: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Return (low, high) percentile CI for the mean via bootstrap."""
    if data.size == 0 or n_samples <= 0:
        return (np.nan, np.nan)
    means = []
    rng = np.random.default_rng(1234)
    for _ in range(n_samples):
        resample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(resample))
    low = float(np.percentile(means, 100 * (alpha / 2)))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return low, high


def summarize(
    robust_eval: Dict,
    *,
    bootstrap: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Build a flat summary of gains per attack mode and circuit type.

    Returns a DataFrame with columns:
        attack_mode, circuit, variant, mean, std, delta_vs_base, ci_low, ci_high
    """
    rows: List[Dict] = []
    for mode, data in robust_eval.get("per_mode", {}).items():
        aggregated = data.get("aggregated", {})
        for circuit in ("vanilla", "robust", "quantumnas"):
            base = aggregated.get(circuit)
            mitigated = aggregated.get(f"{circuit}_mitigated") or aggregated.get(f"{circuit}_twirled")
            if not base:
                continue
            # Base row
            rows.append(
                {
                    "attack_mode": mode,
                    "circuit": circuit,
                    "variant": "baseline",
                    "mean": base["mean"],
                    "std": base["std"],
                    "delta_vs_base": 0.0,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
            if mitigated:
                delta = mitigated["mean"] - base["mean"]
                ci_low = np.nan
                ci_high = np.nan
                if bootstrap > 0:
                    # Extract per-seed samples for this variant to bootstrap the mean
                    seeds = {}
                    for entry in data.get(circuit, []):
                        if entry.get("variant") == ("mitigated" if "mitigated" in mitigated else "twirled"):
                            seeds[entry.get("seed")] = entry.get("mean_attacked")
                    seed_vals = np.array(list(seeds.values()))
                    if seed_vals.size > 0:
                        ci_low, ci_high = _bootstrap_ci(seed_vals, n_samples=bootstrap, alpha=alpha)
                rows.append(
                    {
                        "attack_mode": mode,
                        "circuit": circuit,
                        "variant": "mitigated",
                        "mean": mitigated["mean"],
                        "std": mitigated["std"],
                        "delta_vs_base": delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Summarize robustness gains per attack mode from robust_eval.json.")
    p.add_argument("--robust-eval", type=Path, required=True, help="Path to robust_eval.json")
    p.add_argument("--out-csv", type=Path, required=True, help="Output CSV path for summary table")
    p.add_argument("--bootstrap", type=int, default=0, help="Optional number of bootstrap samples for CI over means")
    p.add_argument("--alpha", type=float, default=0.05, help="Bootstrap alpha (default 0.05 -> 95% CI)")
    args = p.parse_args()

    data = json.loads(args.robust_eval.read_text())
    df = summarize(data, bootstrap=args.bootstrap, alpha=args.alpha)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[robustness_ablation] Saved summary to {args.out_csv} (rows={len(df)})")

    # Quick console view of mitigated gains
    mitigated = df[df["variant"] == "mitigated"]
    if not mitigated.empty:
        print("\nMitigated uplifts (mean Δ vs baseline):")
        for (mode, circuit), sub in mitigated.groupby(["attack_mode", "circuit"]):
            delta = sub["delta_vs_base"].iloc[0]
            ci = ""
            if not np.isnan(sub["ci_low"].iloc[0]):
                ci = f" 95% CI [{sub['ci_low'].iloc[0]:.4f}, {sub['ci_high'].iloc[0]:.4f}]"
            print(f"  {mode:<16s} {circuit:<10s}: Δ={delta:+.4f}{ci}")


if __name__ == "__main__":
    main()
