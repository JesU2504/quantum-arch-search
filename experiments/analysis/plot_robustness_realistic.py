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
    "baseline_twirl": "#9ecae1",
    "robust_twirl": "#fdae6b",
    "quantumnas_twirl": "#a1d99b",
}

LABEL_MAP = {
    "baseline": "RL baseline",
    "robust": "Robust",
    "quantumnas": "HEA baseline",
}

DEFAULT_NOISE_FAMILIES = [
    "random_high",
    "asymmetric_noise",
    "over_rotation",
    "amplitude_damping",
    "phase_damping",
    "readout",
]

NOISE_LABELS = {
    "random_high": "Random depolarization",
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
                        "variant": e.get("variant", "untwirled"),
                    }
                )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path, families: list[str] | None = None):
    processed_families = families or DEFAULT_NOISE_FAMILIES
    families = list(dict.fromkeys(processed_families))
    sns.set_theme(style="whitegrid")
    # Normalize and filter to the requested noise families (default: canonical six)
    normalized = df[df["attack_mode"].isin(families)].copy()

    def _label_noise_family(mode: object) -> str:
        if isinstance(mode, str):
            return NOISE_LABELS.get(mode, mode)
        return str(mode)

    normalized["noise_family"] = normalized["attack_mode"].map(_label_noise_family)
    df = normalized
    # Order noise families for readability and to align with defaults
    order = [NOISE_LABELS.get(mode, mode) for mode in families]
    df = df[df["noise_family"].isin(order)]
    # Aggregate across runs/seeds: mean/std of mean_attacked
    summary = (
        df.groupby(["noise_family", "circuit", "variant"])
        .agg(mean=("mean_attacked", "mean"), std=("mean_attacked", "std"), n=("mean_attacked", "count"))
        .reset_index()
    )
    summary["noise_family"] = pd.Categorical(summary["noise_family"], categories=order, ordered=True)
    hue_order = ["baseline", "robust", "quantumnas"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(order))
    width = 0.22
    err_kw = {"capsize": 4, "capthick": 1.2, "elinewidth": 1.0}
    used_labels = set()
    for j, circuit in enumerate(hue_order):
        sub = summary[summary["circuit"] == circuit].set_index(["noise_family", "variant"]).reindex(order, level=0)
        level_vals = sub.index.get_level_values(1) if not sub.empty else pd.Index([])
        untwirled_present = "untwirled" in level_vals
        twirled_present = "twirled" in level_vals
        if untwirled_present:
            base_means = sub.xs("untwirled", level="variant")["mean"].reindex(order, fill_value=0)
            base_stds = sub.xs("untwirled", level="variant")["std"].reindex(order, fill_value=0)
        else:
            base_means = pd.Series(0.0, index=order)
            base_stds = pd.Series(0.0, index=order)
        if twirled_present:
            twr_means = sub.xs("twirled", level="variant")["mean"].reindex(order, fill_value=0)
            twr_stds = sub.xs("twirled", level="variant")["std"].reindex(order, fill_value=0)
        else:
            twr_means = pd.Series(0.0, index=order)
            twr_stds = pd.Series(0.0, index=order)

        # Stack only the gain from twirling so bar height reflects twirled fidelity
        base_vals = np.clip(base_means.to_numpy(dtype=float, na_value=0.0), 0.0, 1.0)
        gain_vals = np.maximum(0.0, twr_means.to_numpy(dtype=float, na_value=0.0) - base_vals)
        total_vals = base_vals + gain_vals
        over_mask = total_vals > 1.0
        gain_vals[over_mask] = np.maximum(0.0, 1.0 - base_vals[over_mask])
        gain_stds = np.where(gain_vals > 0, twr_stds.to_numpy(dtype=float, na_value=0.0), 0.0)
        base_err = base_stds.fillna(0.0).to_numpy(dtype=float, na_value=0.0)
        base_err_upper = np.minimum(base_err, np.maximum(0.0, 1.0 - base_vals))
        base_err_lower = np.minimum(base_err, base_vals)
        base_yerr = np.vstack([base_err_lower, base_err_upper])
        xpos = x + (j - 1) * width
        lb = LABEL_MAP.get(circuit, circuit) if LABEL_MAP.get(circuit, circuit) not in used_labels else None
        lt = f"{LABEL_MAP.get(circuit, circuit)} (twirl gain)" if f"{LABEL_MAP.get(circuit, circuit)} (twirl gain)" not in used_labels else None

        ax.bar(
            xpos,
            base_vals,
            width,
            yerr=base_yerr,
            label=lb,
            color=PALETTE.get(circuit, None),
            alpha=0.9,
            error_kw=err_kw,
        )
        if twirled_present and np.any(gain_vals > 0):
            gain_err_upper = np.minimum(gain_stds, np.maximum(0.0, 1.0 - (base_vals + gain_vals)))
            gain_err_lower = np.minimum(gain_stds, gain_vals)
            gain_yerr = np.vstack([gain_err_lower, gain_err_upper])
            ax.bar(
                xpos,
                gain_vals,
                width,
                bottom=base_vals,
                yerr=gain_yerr,
                label=lt,
                color=PALETTE.get(f"{circuit}_twirl", "#c7e9c0"),
                alpha=0.8,
                error_kw=err_kw,
            )
            used_labels.update(filter(None, [lb, lt]))
        else:
            used_labels.update(filter(None, [lb]))
        # Skip value annotations to avoid overlap
    ax.set_ylabel("Attacked fidelity")
    ax.set_xlabel("Noise family")
    ax.set_ylim(0, 1.05)
    ax.set_title("Realistic noise robustness (single setting per family)")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.legend(title="Circuit", frameon=True, loc="lower right", fontsize=8, title_fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot realistic robustness by noise family from robust_eval.json")
    p.add_argument("--robust-eval", required=True, help="Path to robust_eval.json")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument(
        "--noise-families",
        nargs="+",
        default=None,
        help=(
            "Noise families (attack modes) to include and order in the plot. "
            "Defaults to: " + ", ".join(DEFAULT_NOISE_FAMILIES)
        ),
    )
    args = p.parse_args()

    df = load_per_run(Path(args.robust_eval))
    if df.empty:
        raise SystemExit("No aggregated data found in robust_eval.json")
    plot(df, Path(args.out), families=args.noise_families)


if __name__ == "__main__":
    main()
