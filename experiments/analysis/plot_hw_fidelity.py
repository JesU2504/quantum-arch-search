#!/usr/bin/env python3
"""
Plot hardware-style success probabilities (proxy for fidelity) from qiskit_hw_eval.py outputs.

Input: hardware_eval_results.json produced by experiments/analysis/qiskit_hw_eval.py
Structure:
{
  "backend_name": {
      "baseline": {..., "success_prob": float, ...},
      "robust": {...},
      "quantumnas": {...}
  },
  ...
}
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

PALETTE = {
    "baseline": "#4C78A8",
    "robust": "#F58518",
    "quantumnas": "#54A24B",
}


def load_results(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    records = []
    for backend, circuits in data.items():
        # New format: list of per-run dicts
        if isinstance(circuits, list):
            for stats in circuits:
                records.append(
                    {
                        "backend": backend,
                        "circuit": stats.get("circuit"),
                        "success_prob": stats.get("success_prob", 0.0),
                        "shots": stats.get("shots", None),
                        "depth": stats.get("depth", None),
                    }
                )
        else:
            # Legacy format: dict[label] -> stats
            for label, stats in circuits.items():
                records.append(
                    {
                        "backend": backend,
                        "circuit": label,
                        "success_prob": stats.get("success_prob", 0.0),
                        "shots": stats.get("shots", None),
                        "depth": stats.get("depth", None),
                    }
                )
    return pd.DataFrame(records)


def plot(df: pd.DataFrame, out_path: Path):
    sns.set_theme(style="whitegrid")
    hue_order = ["baseline", "robust", "quantumnas"]
    label_map = {
        "baseline": "Baseline",
        "robust": "Robust",
        "quantumnas": "HEA baseline",
    }

    # Aggregate across seeds/runs if multiple entries per backend/circuit
    summary = (
        df.groupby(["backend", "circuit"])
        .agg(mean=("success_prob", "mean"), std=("success_prob", "std"), n=("success_prob", "count"))
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0)

    order = sorted(summary["backend"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(order))
    width = 0.25

    for j, circuit in enumerate(hue_order):
        sub = summary[summary["circuit"] == circuit].set_index("backend").reindex(order)
        means = sub["mean"].values
        stds = sub["std"].values
        ax.bar(
            x + (j - 1) * width,
            means,
            width,
            yerr=stds,
            label=label_map.get(circuit, circuit),
            color=PALETTE.get(circuit, None),
            alpha=0.9,
            capsize=4,
        )

    ax.set_ylabel("Success probability (proxy fidelity)")
    ax.set_xlabel("Backend")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=10)
    ax.set_ylim(0, 1.05)

    # Annotate bars with values
    from matplotlib.container import BarContainer
    for c in ax.containers:
        if isinstance(c, BarContainer):
            ax.bar_label(c, fmt="%.3f", padding=2, fontsize=8)

    shots = df["shots"].dropna().unique()
    shots_txt = f"shots={int(shots[0])}" if len(shots) == 1 else ""
    ax.set_title(f"Hardware-style evaluation {shots_txt}".strip())
    ax.legend(
        title="Circuit",
        frameon=True,
        loc="lower right",
        fontsize=7,
        title_fontsize=8,
        markerscale=0.7,
        handlelength=1.2,
        handleheight=0.9,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot hardware eval success probabilities.")
    p.add_argument("--results-json", required=True, help="hardware_eval_results.json path")
    p.add_argument("--out", required=True, help="Output PNG path")
    args = p.parse_args()

    df = load_results(Path(args.results_json))
    plot(df, Path(args.out))


if __name__ == "__main__":
    main()
