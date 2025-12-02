"""
Multi-seed co-evolution plotting.

Given a ROOT directory that contains
  seed_k/adversarial/adversarial_training_YYYY.../
for k = 0..N-1, this script:

- Computes per-generation stats for each seed:
    * best clean fidelity
    * mean attacked fidelity
    * robustness gap (clean - attacked)
    * mean saboteur error rate
    * mean gate count (complexity)
- Aggregates across seeds (mean ± std)
- Plots a 3-panel dashboard with error bands.

Usage:
  python experiments/plot_coevolution_multiseed.py \
      --root-dir results/coevolution_multiseed \
      --out coevolution_multiseed.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Shared palette for consistency across plots
COLORS = {
    "arch_clean": "#2ecc71",
    "sab_noisy": "#e67e22",
    "gap": "#c0392b",
    "error": "#8e44ad",
    "complexity": "#3498db",
}


def find_adversarial_run_dirs(root_dir):
    """
    Discover adversarial_training_* directories under each seed_* subdir.
    Supports both layouts:
      seed_k/adversarial/adversarial_training_*
      seed_k/adversarial_training_*
    Returns a list of full paths.
    """
    run_dirs = []
    for seed_name in sorted(os.listdir(root_dir)):
        seed_path = os.path.join(root_dir, seed_name)
        if not os.path.isdir(seed_path):
            continue

        # Pattern A: seed_k/adversarial/adversarial_training_*
        adv_root = os.path.join(seed_path, "adversarial")
        candidates = []
        if os.path.isdir(adv_root):
            candidates = [
                os.path.join(adv_root, d)
                for d in os.listdir(adv_root)
                if d.startswith("adversarial_training")
            ]
        # Pattern B: seed_k/adversarial_training_*
        if not candidates:
            candidates = [
                os.path.join(seed_path, d)
                for d in os.listdir(seed_path)
                if d.startswith("adversarial_training")
            ]
        if not candidates:
            continue

        candidates.sort()
        run_dirs.append(candidates[0])

    return run_dirs


def load_txt_or_none(path):
    if not os.path.exists(path):
        return None
    arr = np.loadtxt(path)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr


def load_metadata_from_run_dir(run_dir):
    """
    Use the same assumption as single-run plot:
    metadata.json is two levels up from adversarial_training_*/ .
    """
    base_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))
    meta_path = os.path.join(base_run_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)

    # Fallback: walk up ancestors to find the first metadata.json (e.g., at run root)
    current = os.path.abspath(base_run_dir)
    root = os.path.abspath(os.sep)
    while current != root:
        cand = os.path.join(current, "metadata.json")
        if os.path.exists(cand):
            print(f"[WARN] metadata.json not found at {meta_path}; using {cand}")
            with open(cand, "r") as f:
                return json.load(f)
        current = os.path.dirname(current)

    print(f"[WARN] metadata.json not found for run {run_dir}")
    return None


def assign_generations(steps, steps_per_gen, n_gens):
    gens = (steps - 1) // steps_per_gen
    return np.clip(gens, 0, n_gens - 1).astype(int)


def compute_per_generation_stats(run_dir):
    """
    For a single adversarial_training_* directory, compute per-generation:
      best_clean, mean_clean, mean_noisy, mean_error, mean_complex, gap
    """
    arch_fid = load_txt_or_none(os.path.join(run_dir, "architect_fidelities.txt"))
    arch_steps = load_txt_or_none(os.path.join(run_dir, "architect_steps.txt"))
    sab_fid = load_txt_or_none(os.path.join(run_dir, "saboteur_trained_on_architect_fidelities.txt"))
    sab_steps = load_txt_or_none(os.path.join(run_dir, "saboteur_trained_on_architect_steps.txt"))
    arch_complex = load_txt_or_none(os.path.join(run_dir, "architect_complexity.txt"))
    sab_err = load_txt_or_none(os.path.join(run_dir, "saboteur_error_rates.txt"))

    if arch_fid is None or sab_fid is None:
        raise RuntimeError(f"Missing fidelity logs in {run_dir}")

    meta = load_metadata_from_run_dir(run_dir)
    if meta is None:
        raise RuntimeError(f"Missing metadata for {run_dir}")

    n_gens = int(meta["adversarial_gens"])
    arch_steps_per_gen = int(meta["adversarial_arch_steps"])
    sab_total_steps = int(meta["saboteur_steps"])
    sab_steps_per_gen = sab_total_steps // max(1, n_gens)

    if arch_steps is None:
        arch_steps = np.arange(1, len(arch_fid) + 1)
    if sab_steps is None:
        sab_steps = np.arange(1, len(sab_fid) + 1)

    arch_gens = assign_generations(arch_steps, arch_steps_per_gen, n_gens)
    sab_gens = assign_generations(sab_steps, sab_steps_per_gen, n_gens)

    gen_indices = np.arange(n_gens)

    best_clean = np.zeros(n_gens)
    mean_clean = np.zeros(n_gens)
    mean_noisy = np.zeros(n_gens)
    mean_error = np.zeros(n_gens)
    mean_complex = np.zeros(n_gens)

    for g in gen_indices:
        m_arch = (arch_gens == g)
        m_sab = (sab_gens == g)

        if np.any(m_arch):
            best_clean[g] = np.max(arch_fid[m_arch])
            mean_clean[g] = np.mean(arch_fid[m_arch])
            if arch_complex is not None:
                mean_complex[g] = np.mean(arch_complex[m_arch])
            else:
                mean_complex[g] = np.nan
        else:
            best_clean[g] = np.nan
            mean_clean[g] = np.nan
            mean_complex[g] = np.nan

        if np.any(m_sab):
            mean_noisy[g] = np.mean(sab_fid[m_sab])
            if sab_err is not None:
                mean_error[g] = np.mean(sab_err[m_sab])
            else:
                mean_error[g] = np.nan
        else:
            mean_noisy[g] = np.nan
            mean_error[g] = np.nan

    gap = best_clean - mean_noisy

    return {
        "n_gens": n_gens,
        "best_clean": best_clean,
        "mean_clean": mean_clean,
        "mean_noisy": mean_noisy,
        "mean_error": mean_error,
        "mean_complex": mean_complex,
        "gap": gap,
    }


def plot_multiseed(root_dir, out_path):
    run_dirs = find_adversarial_run_dirs(root_dir)
    if not run_dirs:
        raise RuntimeError(f"No adversarial_training_* dirs found under {root_dir}")

    print("[plot_coevolution_multiseed] Found runs:")
    for d in run_dirs:
        print("  -", d)

    all_stats = []
    valid_runs = []
    for d in run_dirs:
        try:
            all_stats.append(compute_per_generation_stats(d))
            valid_runs.append(d)
        except Exception as e:
            print(f"[WARN] Skipping {d}: {e}")

    if not all_stats:
        raise RuntimeError("No valid runs with fidelity logs found.")

    # Check n_gens consistency
    n_gens = all_stats[0]["n_gens"]
    for st in all_stats[1:]:
        if st["n_gens"] != n_gens:
            raise RuntimeError("Inconsistent number of generations across seeds.")

    gen_indices = np.arange(n_gens)
    gens_plot = gen_indices + 1

    def stack_and_agg(key, clip01=False):
        arr = np.stack([st[key] for st in all_stats], axis=0)  # [n_seeds, n_gens]
        if clip01:
            arr = np.clip(arr, 0.0, 1.0)
        return arr, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    best_clean_all, best_clean_mean, best_clean_std = stack_and_agg("best_clean", clip01=True)
    mean_noisy_all, mean_noisy_mean, mean_noisy_std = stack_and_agg("mean_noisy", clip01=True)
    gap_all, gap_mean, gap_std = stack_and_agg("gap")
    mean_error_all, mean_error_mean, mean_error_std = stack_and_agg("mean_error")
    mean_complex_all, mean_complex_mean, mean_complex_std = stack_and_agg("mean_complex")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]}
    )

    # ---------- PANEL 1: Fidelity & Robustness Gap ---------- #
    # Light per-seed overlays for context
    for i in range(best_clean_all.shape[0]):
        ax1.plot(gens_plot, best_clean_all[i], color=COLORS["arch_clean"], alpha=0.15, linewidth=1)
        ax1.plot(gens_plot, mean_noisy_all[i], color=COLORS["sab_noisy"], alpha=0.15, linewidth=1)

    ax1.plot(
        gens_plot, best_clean_mean, marker="o", linestyle="-", color=COLORS["arch_clean"],
        label="Best Clean Fidelity (mean ± std)"
    )
    ax1.fill_between(
        gens_plot,
        best_clean_mean - best_clean_std,
        best_clean_mean + best_clean_std,
        color=COLORS["arch_clean"],
        alpha=0.15,
    )

    ax1.plot(
        gens_plot, mean_noisy_mean, marker="s", linestyle="-", color=COLORS["sab_noisy"],
        label="Fidelity Under Attack (mean ± std)"
    )
    ax1.fill_between(
        gens_plot,
        mean_noisy_mean - mean_noisy_std,
        mean_noisy_mean + mean_noisy_std,
        color=COLORS["sab_noisy"],
        alpha=0.15,
    )

    ax1.set_ylabel("Fidelity", fontweight="bold")
    ax1.set_ylim(0.0, 1.05)
    ax1.margins(x=0.01)

    # Legend (fidelity traces only)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    title = f"Co-Evolution Across Seeds (N={len(valid_runs)} runs, {n_gens} generations)"
    ax1.set_title(
        title,
        fontsize=14,
        fontweight="bold",
    )

    # Summary box with finals
    ax1.text(
        0.98, 0.98,
        f"Seeds: {len(valid_runs)}\n"
        f"Final clean: {best_clean_mean[-1]:.3f} ± {best_clean_std[-1]:.3f}\n"
        f"Final attacked: {mean_noisy_mean[-1]:.3f} ± {mean_noisy_std[-1]:.3f}",
        transform=ax1.transAxes,
        fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.9)
    )

    # ---------- PANEL 2: Saboteur Intensity (Error Rate) ---------- #
    ax2.plot(
        gens_plot, mean_error_mean, marker="o", linestyle="-",
        color=COLORS["error"], label="Mean Error Rate (over seeds)"
    )
    ax2.fill_between(
        gens_plot,
        mean_error_mean - mean_error_std,
        mean_error_mean + mean_error_std,
        color=COLORS["error"],
        alpha=0.2,
    )
    ax2.set_ylabel("Error Rate", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.margins(x=0.01)

    # ---------- PANEL 3: Architect Complexity (Gate Count) ---------- #
    ax3.plot(
        gens_plot, mean_complex_mean, marker="^", linestyle="-",
        color=COLORS["complexity"], label="Mean Gate Count (over seeds)"
    )
    ax3.fill_between(
        gens_plot,
        mean_complex_mean - mean_complex_std,
        mean_complex_mean + mean_complex_std,
        color=COLORS["complexity"],
        alpha=0.2,
    )
    ax3.set_ylabel("Gate Count", fontweight="bold")
    ax3.set_xlabel("Generation", fontweight="bold")
    ax3.legend(loc="upper left")
    ax3.margins(x=0.01)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[plot_coevolution_multiseed] Saved multi-seed figure to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True,
                        help="Root directory containing seed_k/adversarial/adversarial_training_*/")
    parser.add_argument("--out", type=str, default="coevolution_multiseed.png",
                        help="Output filename (PNG)")
    args = parser.parse_args()

    plot_multiseed(args.root_dir, args.out)
