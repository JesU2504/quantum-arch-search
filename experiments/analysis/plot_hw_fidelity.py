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
    "baseline": "#54A24B",  # RL baseline (green)
    "robust": "#F58518",
    "quantumnas": "#4C78A8",  # HEA baseline (blue)
    "baseline_twirl": "#a1d99b",
    "robust_twirl": "#fdae6b",
    "quantumnas_twirl": "#9ecae1",
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
                        "success_prob": stats.get("success_prob"),
                        "energy": stats.get("energy"),
                        "shots": stats.get("shots", None),
                        "depth": stats.get("depth", None),
                        "variant": stats.get("variant", "untwirled"),
                    }
                )
        else:
            # Legacy format: dict[label] -> stats
            for label, stats in circuits.items():
                records.append(
                    {
                        "backend": backend,
                        "circuit": label,
                        "success_prob": stats.get("success_prob"),
                        "energy": stats.get("energy"),
                        "shots": stats.get("shots", None),
                        "depth": stats.get("depth", None),
                        "variant": stats.get("variant", "untwirled"),
                    }
                )
    return pd.DataFrame(records)


def plot(df: pd.DataFrame, out_path: Path):
    sns.set_theme(style="whitegrid")
    hue_order = ["baseline", "robust", "quantumnas"]
    label_map = {
        "baseline": "RL baseline",
        "robust": "Robust",
        "quantumnas": "HEA baseline",
    }

    metric_col = "energy" if df.get("energy") is not None and df["energy"].notna().any() else "success_prob"
    if metric_col not in df.columns:
        metric_col = "success_prob"

    if "variant" not in df.columns:
        df["variant"] = "untwirled"

    summary = (
        df.groupby(["backend", "circuit", "variant"])
        .agg(mean=(metric_col, "mean"), std=(metric_col, "std"), n=(metric_col, "count"))
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0)

    if metric_col == "energy":
        order = sorted(summary["backend"].unique())
        variant_order = ["untwirled", "mitigated", "twirled"]
        summary["plot_label"] = summary.apply(
            lambda r: label_map.get(r["circuit"], r["circuit"]) if r["variant"] == "untwirled" else f"{label_map.get(r['circuit'], r['circuit'])} ({r['variant']})",
            axis=1,
        )
        label_order = []
        for circuit in hue_order:
            for variant in variant_order:
                lab_base = label_map.get(circuit, circuit)
                label = lab_base if variant == "untwirled" else f"{lab_base} ({variant})"
                if label in summary["plot_label"].values and label not in label_order:
                    label_order.append(label)
        if not label_order:
            label_order = sorted(summary["plot_label"].unique())
        fig, ax = plt.subplots(figsize=(max(6.0, 2.0 * len(order)), 4.5))
        x = np.arange(len(order))
        width = 0.8 / max(1, len(label_order))
        for idx, label in enumerate(label_order):
            sub = summary[summary["plot_label"] == label].set_index("backend")
            means = sub.reindex(order)["mean"].astype(float)
            stds = sub.reindex(order)["std"].astype(float).fillna(0.0)
            pos = x - 0.4 + width / 2 + idx * width
            ax.bar(pos, means, width, yerr=stds, label=label)
        ax.set_ylabel("Energy (Ha)")
        ax.set_xlabel("Backend")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=10)
        ax.legend(frameon=True, title="Circuit / Variant", fontsize=7)
        ax.set_title("Hardware-style energy evaluation")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    order = sorted(summary["backend"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(order))
    width = 0.25
    err_kw = {"capsize": 4, "capthick": 1.2, "elinewidth": 1.0}

    used_labels = set()
    for j, circuit in enumerate(hue_order):
        sub = summary[summary["circuit"] == circuit]
        # Pivot to backend x variant
        pivot_mean = sub.pivot(index="backend", columns="variant", values="mean")
        pivot_std = sub.pivot(index="backend", columns="variant", values="std")
        base_series = pivot_mean.reindex(order).get("untwirled", pd.Series(0.0, index=order)).fillna(0.0)
        base_std_series = pivot_std.reindex(order).get("untwirled", pd.Series(0.0, index=order)).fillna(0.0)
        twirl_series_raw = pivot_mean.reindex(order).get("twirled")
        twirl_std_raw = pivot_std.reindex(order).get("twirled")

        twirl_series = twirl_series_raw.fillna(0.0) if twirl_series_raw is not None else None
        twirl_std_series = twirl_std_raw.fillna(0.0) if twirl_std_raw is not None else None

        base_means = np.clip(base_series.to_numpy(dtype=float, na_value=0.0), 0.0, 1.0)
        base_stds = base_std_series.fillna(0.0).to_numpy(dtype=float, na_value=0.0)

        twirl_present = twirl_series is not None and twirl_series.notna().any()
        if twirl_present and twirl_series is not None:
            twr_means = twirl_series.to_numpy(dtype=float, na_value=0.0)
            if twirl_std_series is not None:
                twr_stds = twirl_std_series.to_numpy(dtype=float, na_value=0.0)
            else:
                twr_stds = np.zeros_like(base_means)
        else:
            twr_means = np.zeros_like(base_means)
            twr_stds = np.zeros_like(base_means)

        # Stack only the uplift from twirling so bar height equals twirled fidelity
        base_means = np.clip(base_means, 0.0, 1.0)
        gain_vals = np.maximum(0.0, twr_means - base_means)
        combined = base_means + gain_vals
        over_mask = combined > 1.0
        gain_vals[over_mask] = np.maximum(0.0, 1.0 - base_means[over_mask])

        # Clip error bars to stay within [0,1] and hide gain error when no uplift
        cap_base_stds = []
        cap_gain_stds = []
        for idx in range(len(base_means)):
            headroom_base = max(0.0, 1.0 - base_means[idx])
            cap_base_stds.append(min(base_stds[idx], headroom_base))
            headroom_gain = max(0.0, 1.0 - (base_means[idx] + gain_vals[idx]))
            cap_gain_stds.append(min(twr_stds[idx], headroom_gain) if gain_vals[idx] > 0 else 0.0)
        base_stds = np.array(cap_base_stds)
        gain_stds = np.array(cap_gain_stds)

        label_base = label_map.get(circuit, circuit)
        label_twirled = f"{label_base} (twirl gain)"
        lb = label_base if label_base not in used_labels else None
        lt = label_twirled if (twirl_present and label_twirled not in used_labels) else None
        pos = x + (j - 1) * width
        ax.bar(
            pos,
            base_means,
            width,
            yerr=base_stds,
            label=lb,
            color=PALETTE.get(circuit, None),
            alpha=0.9,
            error_kw=err_kw,
        )
        if twirl_present and np.any(gain_vals > 0):
            ax.bar(
                pos,
                gain_vals,
                width,
                bottom=base_means,
                yerr=gain_stds,
                label=lt,
                color=PALETTE.get(f"{circuit}_twirl", "#c7e9c0"),
                alpha=0.8,
                error_kw=err_kw,
            )
            used_labels.update(filter(None, [lb, lt]))
        else:
            used_labels.update(filter(None, [lb]))

    ax.set_ylabel("Success probability (proxy fidelity)")
    ax.set_xlabel("Backend")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=10)
    ax.set_ylim(0, 1.05)

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
