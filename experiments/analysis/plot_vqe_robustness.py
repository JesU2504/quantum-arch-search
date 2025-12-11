#!/usr/bin/env python3
"""
Plot VQE robustness (energy under noise) from vqe_robust_eval.json produced by compare_vqe_energy.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "vanilla": "#54A24B",  # RL
    "robust": "#F58518",  # Robust
    "quantumnas": "#4C78A8",  # HEA
    "vanilla_mitigated": "#74c476",
    "robust_mitigated": "#fdd0a2",
    "quantumnas_mitigated": "#6baed6",
}

LABEL_MAP = {
    "vanilla": "RL",
    "robust": "Robust",
    "quantumnas": "HEA",
}


def load_eval(path: Path):
    return json.loads(path.read_text())


def plot_vqe(eval_data: dict, out_path: Path):
    per_mode = eval_data.get("per_mode", {})
    modes = list(per_mode.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.22
    x = np.arange(len(modes))
    err_kw = {"capsize": 4, "capthick": 1.2, "elinewidth": 1.0}

    def series(label):
        means = []
        stds = []
        mitigated = []
        mitigated_stds = []
        for m in modes:
            agg = eval_data.get("aggregated", {}).get(m, {})
            base = agg.get(label, {})
            base_mean = base.get("mean", np.nan)
            base_std = base.get("std", 0.0)
            means.append(base_mean)
            stds.append(base_std)
            mit = agg.get(f"{label}_mitigated", {})
            mitigated.append(mit.get("mean", np.nan))
            mitigated_stds.append(mit.get("std", 0.0))
        return np.array(means), np.array(stds), np.array(mitigated), np.array(mitigated_stds)

    labels = ["vanilla", "robust", "quantumnas"]
    offsets = [-width, 0.0, width]
    for off, lbl in zip(offsets, labels):
        base_m, base_s, mit_m, mit_s = series(lbl)
        disp = LABEL_MAP.get(lbl, lbl)
        ax.bar(x + off, base_m, width=width, yerr=base_s, color=PALETTE[lbl], label=f"{disp} (raw)", error_kw=err_kw)
        if not np.all(np.isnan(mit_m)):
            ax.bar(
                x + off,
                mit_m,
                width=width,
                yerr=mit_s,
                color=PALETTE.get(f"{lbl}_mitigated", "#cccccc"),
                alpha=0.7,
                label=f"{disp} (mitigated)" if lbl == "vanilla" else None,
                error_kw=err_kw,
                hatch="//",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15)
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("VQE robustness (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    handles, labels_l = ax.get_legend_handles_labels()
    # dedup labels
    seen = set()
    filtered = []
    for h, l in zip(handles, labels_l):
        if l and l not in seen:
            filtered.append((h, l))
            seen.add(l)
    if filtered:
        ax.legend(*zip(*filtered), fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot VQE robustness from vqe_robust_eval.json")
    p.add_argument("--eval-json", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = load_eval(Path(args.eval_json))
    plot_vqe(data, Path(args.out))


if __name__ == "__main__":
    main()
