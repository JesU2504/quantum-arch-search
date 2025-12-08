#!/usr/bin/env python3
"""
Plot realistic hardware-style robustness across noise families (single setting per family),
using aggregated results from compare_circuits robust_eval.json.

Input: robust_eval.json produced by compare_circuits.py with attack_modes including
       random_high, amplitude_damping, phase_damping, readout (and optionally others).
Output: Bar plot comparing baseline / robust / HEA baseline across noise families.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.container import BarContainer

PALETTE = {
    "baseline": "#4C78A8",
    "robust": "#F58518",
    "quantumnas": "#54A24B",
}

LABEL_MAP = {
    "baseline": "Baseline",
    "robust": "Robust",
    "quantumnas": "HEA baseline",
}

NOISE_LABELS = {
    "random_high": "Depol (random-high)",
    "asymmetric_noise": "Asymmetric Pauli",
    "over_rotation": "Over-rotation",
    "amplitude_damping": "Amplitude damping",
    "phase_damping": "Phase damping",
    "readout": "Readout",
}


def load_per_run(path: Path) -> pd.DataFrame:
    """Flatten per-run metrics from robust_eval.json."""
    data = json.loads(path.read_text())
    per_mode = data.get("per_mode", {})
    rows = []
    key_map = {"vanilla": "baseline", "robust": "robust", "quantumnas": "quantumnas"}
    for mode, buckets in per_mode.items():
        for raw_name, plot_name in key_map.items():
            entries = buckets.get(raw_name, [])
            for e in entries:
                rows.append(
                    {
                        "noise_family": NOISE_LABELS.get(mode, mode),
                        "attack_mode": mode,
                        "circuit": plot_name,
                        "mean_attacked": e.get("mean_attacked", 0.0),
                    }
                )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path):
    sns.set_theme(style="whitegrid")
    # Order noise families for readability
    order = [
        "Depolirization",
        "Amplitude damping",
        "Phase damping",
        "Readout",
        "Over-rotation",
        "Asymmetric Pauli",
    ]
    df = df[df["noise_family"].isin(order)]
    # Aggregate across runs/seeds: mean/std of mean_attacked
    summary = (
        df.groupby(["noise_family", "circuit"])
        .agg(mean=("mean_attacked", "mean"), std=("mean_attacked", "std"), n=("mean_attacked", "count"))
        .reset_index()
    )
    summary["noise_family"] = pd.Categorical(summary["noise_family"], categories=order, ordered=True)
    hue_order = ["baseline", "robust", "quantumnas"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(order))
    width = 0.25
    for j, circuit in enumerate(hue_order):
        sub = summary[summary["circuit"] == circuit].set_index("noise_family").reindex(order)
        means = sub["mean"].values
        stds = sub["std"].fillna(0).values
        ax.bar(
            x + (j - 1) * width,
            means,
            width,
            yerr=stds,
            label=LABEL_MAP.get(circuit, circuit),
            color=PALETTE.get(circuit, None),
            alpha=0.9,
            capsize=4,
        )
    ax.set_ylabel("Attacked fidelity")
    ax.set_xlabel("Noise family")
    ax.set_ylim(0, 1.05)
    ax.set_title("Realistic noise robustness (single setting per family)")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.legend(title="Circuit", frameon=True, loc="lower right", fontsize=8, title_fontsize=9)

    # Annotate bar values
    for c in ax.containers:
        if isinstance(c, BarContainer):
            ax.bar_label(c, fmt="%.3f", padding=2, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot realistic robustness by noise family from robust_eval.json")
    p.add_argument("--robust-eval", required=True, help="Path to robust_eval.json")
    p.add_argument("--out", required=True, help="Output PNG path")
    args = p.parse_args()

    df = load_per_run(Path(args.robust_eval))
    if df.empty:
        raise SystemExit("No aggregated data found in robust_eval.json")
    plot(df, Path(args.out))


if __name__ == "__main__":
    main()
