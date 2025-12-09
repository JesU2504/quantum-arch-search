#!/usr/bin/env python3
"""
Plot robustness sweep results (attacked fidelity vs noise family/budget) for baseline, robust, and HEA baseline circuits.
Consumes results/*/analysis/robustness_sweep.json produced by run_experiments.py.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


PALETTE = {
    "architect_baseline": "#4C78A8",
    "adversarial": "#F58518",
    "quantumnas": "#54A24B",
}


def load_sweep(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path):
    sns.set_theme(style="whitegrid")
    order = ["depolarizing", "amplitude_damping", "coherent_overrotation"]
    group_order = ["architect_baseline", "adversarial", "quantumnas"]
    label_map = {
        "architect_baseline": "RL baseline",
        "adversarial": "Robust",
        "quantumnas": "HEA baseline",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=df,
        x="noise_family",
        y="attacked_mean",
        hue="group",
        order=order,
        hue_order=group_order,
        palette=PALETTE,
        dodge=True,
        ax=ax,
        errorbar="sd",
    )
    ax.set_ylabel("Attacked fidelity")
    ax.set_xlabel("Noise family")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustness sweep: fidelity under budgeted attacks\n(error bars: ±1 sd across seeds/runs)")
    # Annotate gate budgets on x-axis labels
    budgets = sorted(df["attack_budget"].unique())
    ax.text(
        0.02,
        0.02,
        f"Budgets evaluated: {budgets}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )
    handles, labels = ax.get_legend_handles_labels()
    labels = [label_map.get(l, l) for l in labels]
    ax.legend(handles, labels, title="Method", frameon=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot robustness sweep results.")
    p.add_argument("--sweep-json", type=str, required=True, help="Path to robustness_sweep.json")
    p.add_argument("--out", type=str, required=True, help="Output PNG path")
    args = p.parse_args()

    df = load_sweep(Path(args.sweep_json))
    # Aggregate over budgets? Keep both budgets in a facet? For simplicity, average over runs/seeds and plot all rows.
    plot(df, Path(args.out))


if __name__ == "__main__":
    main()
