"""
Single-seed co-evolution plot, styled like train_architect:
- Best clean fidelity so far
- Rolling mean (clean)
- Rolling mean (attacked/noisy)
"""

import argparse
import os
import json
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    "best": "#72b7b2",
    "clean_roll": "#f58518",
    "noisy_roll": "#e67e22",
    "ideal": "#7f8c8d",
}


def load_data(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path)
        if data.size == 0:
            return None
        if data.ndim > 1:
            data = data[:, 0]
        return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def rolling_mean(data, window_size):
    if data is None or len(data) == 0:
        return np.array([])
    if window_size <= 1 or len(data) < window_size:
        return data
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="valid")


def find_run_dir(root_dir: str, seed: int | None):
    """Find the latest adversarial_training_* dir for a given seed under root_dir/seed_{seed}."""
    if seed is None:
        return root_dir
    seed_dir = os.path.join(root_dir, f"seed_{seed}")
    if not os.path.isdir(seed_dir):
        return None
    candidates = glob(os.path.join(seed_dir, "adversarial_training_*"))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def plot_coevolution_single(run_dir, save_name, window_frac=0.02):
    print(f"--- Generating Co-Evolution Plot for {run_dir} ---")

    clean_fidelity = load_data(os.path.join(run_dir, "architect_fidelities.txt"))
    clean_steps = load_data(os.path.join(run_dir, "architect_steps.txt"))
    noisy_fidelity = load_data(os.path.join(run_dir, "saboteur_trained_on_architect_fidelities.txt"))
    noisy_steps = load_data(os.path.join(run_dir, "saboteur_trained_on_architect_steps.txt"))

    if clean_fidelity is None or noisy_fidelity is None:
        print("[plot_coevolution] Error: missing fidelity logs.")
        return

    if clean_steps is None:
        clean_steps = np.arange(1, len(clean_fidelity) + 1)
    if noisy_steps is None:
        noisy_steps = np.arange(1, len(noisy_fidelity) + 1)

    clean_window = max(10, int(len(clean_fidelity) * window_frac))
    # Use the same window for attacked to make the curves visually comparable
    noisy_window = clean_window
    clean_roll = rolling_mean(clean_fidelity, clean_window)
    noisy_roll = rolling_mean(noisy_fidelity, noisy_window)

    clean_x = clean_steps[: len(clean_roll)]
    noisy_x = noisy_steps[: len(noisy_roll)]

    best_clean = np.maximum.accumulate(clean_fidelity)
    best_attacked = np.max(noisy_fidelity[-20000:])
    best_attacked = [best_attacked for _ in clean_x]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(clean_steps, best_clean, color=COLORS["best"], linewidth=2.2, label="Best Clean So Far")
    ax.plot(clean_x, clean_roll, color=COLORS["clean_roll"], linewidth=2.2, label=f"Rolling Mean (clean, w={clean_window})")
    ax.plot(clean_x, best_attacked, color="#c0392b", linewidth=2.2, linestyle="--", label="Best Attacked So Far")
    #ax.plot(noisy_x, noisy_roll, color=COLORS["noisy_roll"], linewidth=2.2, label=f"Rolling Mean (attacked, w={noisy_window})")
    ax.axhline(1.0, color=COLORS["ideal"], linestyle="--", linewidth=1)

    ax.scatter(clean_steps[-1], best_clean[-1], color=COLORS["best"], edgecolor="white", zorder=5, s=40)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(-0.05, 1.05)
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    order = [
        "Best Clean So Far",
        "Best Attacked So Far",
        f"Rolling Mean (clean, w={clean_window})",
        f"Rolling Mean (attacked, w={noisy_window})",
    ]
    label_to_handle = {lbl: h for h, lbl in zip(handles, labels)}
    ordered_handles = [label_to_handle[lbl] for lbl in order if lbl in label_to_handle]
    ordered_labels = [lbl for lbl in order if lbl in label_to_handle]
    ax.legend(ordered_handles, ordered_labels, loc="lower right", frameon=True)

    ax.text(
        0.02, 0.95,
        f"Final best clean: {best_clean[-1]:.3f}\n"
        f"Final roll clean: {clean_roll[-1]:.3f}\n"
        f"Final roll attacked: {noisy_roll[-1]:.3f}\n"
        f"Best attacked: {best_attacked[-1]:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"[plot_coevolution] Saved plot to {save_name}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot co-evolution metrics from adversarial training (single seed).")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to adversarial_training_* directory (if provided, seed/root ignored)")
    parser.add_argument("--root-dir", type=str, default=None,
                        help="Path to adversarial/ parent containing seed_k/ directories")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed index to plot (default: 0)")
    parser.add_argument("--out", type=str, default="coevolution_corrected.png",
                        help="Output filename (PNG)")
    parser.add_argument("--window-frac", type=float, default=0.02,
                        help="Rolling window as fraction of series length (default: 0.02)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        if args.root_dir is None:
            raise SystemExit("Provide either --run-dir or --root-dir with --seed")
        run_dir = find_run_dir(args.root_dir, args.seed)
        if run_dir is None:
            raise SystemExit(f"Could not find adversarial_training_* for seed {args.seed} under {args.root_dir}")

    plot_coevolution_single(run_dir, args.out, window_frac=args.window_frac)
