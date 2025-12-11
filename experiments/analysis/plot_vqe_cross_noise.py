#!/usr/bin/env python3
"""
Plot VQE cross-noise energy robustness from vqe_cross_noise.json.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "vanilla": "#54A24B",  # RL
    "robust": "#F58518",   # Robust
    "quantumnas": "#4C78A8",  # HEA
}

LABEL_MAP = {
    "vanilla": "RL",
    "robust": "Robust",
    "quantumnas": "HEA",
    "RL": "RL",
    "HEA": "HEA",
    "Robust": "Robust",
}


def load_data(path: Path):
    raw = json.loads(path.read_text())
    # Support both nested {"results": {...}} and flat record list
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    if isinstance(raw, list):
        # Flatten records to nested structure
        results = {}
        for row in raw:
            fam = row.get("noise_family")
            if fam is None:
                continue
            results.setdefault(fam, {"vanilla": [], "robust": [], "quantumnas": []})
            method = row.get("method")
            # Best-effort map method names to series keys
            key_map = {
                "RL": "vanilla",
                "HEA adversarial": "quantumnas",
                "Architect adversarial": "robust",
            }
            key = key_map.get(method, method if method in results[fam] else "vanilla")
            results[fam].setdefault(key, [])
            results[fam][key].append(
                {"rate": float(row.get("rate", 0.0)), "mean": float(row.get("energy", 0.0)), "std": 0.0}
            )
        return results
    raise ValueError("Unsupported vqe_cross_noise format")


def plot_cross_noise(results: dict, out: Path):
    families = list(results.keys())
    fig, axes = plt.subplots(1, len(families), figsize=(4 * len(families), 4), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, fam in zip(axes, families):
        fam_res = results[fam]
        # Accept either vanilla/robust/quantumnas or RL/Robust/HEA keys
        series_keys = []
        for cand in ["vanilla", "robust", "quantumnas", "RL", "Robust", "HEA"]:
            if cand in fam_res:
                series_keys.append(cand)
        if not series_keys:
            continue
        rates = sorted({float(r["rate"]) for key in series_keys for r in fam_res.get(key, [])})

        def agg(label):
            means = []
            stds = []
            for rate in rates:
                vals = [r.get("mean", r.get("energy", np.nan)) for r in fam_res.get(label, []) if float(r.get("rate", 0.0)) == rate]
                svals = [r.get("std", 0.0) for r in fam_res.get(label, []) if float(r.get("rate", 0.0)) == rate]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.mean(svals) if svals else 0.0)
            return np.array(means), np.array(stds)

        x = np.arange(len(rates))
        for idx, label in enumerate(series_keys):
            m, s = agg(label)
            disp = LABEL_MAP.get(label, label)
            ax.errorbar(
                x + idx * 0.08,
                m,
                yerr=s,
                fmt="o-",
                label=disp,
                color=PALETTE.get(label if label in PALETTE else {"RL": "vanilla", "Robust": "robust", "HEA": "quantumnas"}.get(label, "vanilla"), "#888"),
                capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.3f}" for r in rates], rotation=20)
        ax.set_title(fam)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Rate")
    axes[0].set_ylabel("Energy (Ha)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot VQE cross-noise energy robustness.")
    p.add_argument("--cross-noise-json", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    results = load_data(Path(args.cross_noise_json))
    plot_cross_noise(results, Path(args.out))


if __name__ == "__main__":
    main()
