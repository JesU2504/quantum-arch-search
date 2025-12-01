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


def find_adversarial_run_dirs(root_dir):
    """
    Discover all adversarial_training_* directories under each seed_* subdir.
    Returns a list of full paths.
    """
    run_dirs = []
    for seed_name in sorted(os.listdir(root_dir)):
        seed_path = os.path.join(root_dir, seed_name)
        if not os.path.isdir(seed_path):
            continue

        adv_root = os.path.join(seed_path, "adversarial")
        if not os.path.isdir(adv_root):
            continue

        # Find the first adversarial_training_* subdir (there should be one)
        candidates = [
            d for d in os.listdir(adv_root)
            if d.startswith("adversarial_training")
        ]
        if not candidates:
            continue

        candidates.sort()
        run_dir = os.path.join(adv_root, candidates[0])
        run_dirs.append(run_dir)

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
    if not os.path.exists(meta_path):
        print(f"[WARN] metadata.json not found at {meta_path}")
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


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

    all_stats = [compute_per_generation_stats(d) for d in run_dirs]

    # Check n_gens consistency
    n_gens = all_stats[0]["n_gens"]
    for st in all_stats[1:]:
        if st["n_gens"] != n_gens:
            raise RuntimeError("Inconsistent number of generations across seeds.")

    gen_indices = np.arange(n_gens)
    gens_plot = gen_indices + 1

    def stack_and_agg(key):
        arr = np.stack([st[key] for st in all_stats], axis=0)  # [n_seeds, n_gens]
        return arr, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    best_clean_all, best_clean_mean, best_clean_std = stack_and_agg("best_clean")
    mean_noisy_all, mean_noisy_mean, mean_noisy_std = stack_and_agg("mean_noisy")
    gap_all, gap_mean, gap_std = stack_and_agg("gap")
    mean_error_all, mean_error_mean, mean_error_std = stack_and_agg("mean_error")
    mean_complex_all, mean_complex_mean, mean_complex_std = stack_and_agg("mean_complex")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]}
    )

    # ---------- PANEL 1: Fidelity & Robustness Gap ---------- #
    ax1.plot(
        gens_plot, best_clean_mean, marker="o", linestyle="-", color="#2ecc71",
        label="Best Clean Fidelity (mean over seeds)"
    )
    ax1.fill_between(
        gens_plot,
        best_clean_mean - best_clean_std,
        best_clean_mean + best_clean_std,
        color="#2ecc71",
        alpha=0.2,
    )

    ax1.plot(
        gens_plot, mean_noisy_mean, marker="s", linestyle="-", color="#e67e22",
        label="Mean Fidelity Under Attack (mean over seeds)"
    )
    ax1.fill_between(
        gens_plot,
        mean_noisy_mean - mean_noisy_std,
        mean_noisy_mean + mean_noisy_std,
        color="#e67e22",
        alpha=0.2,
    )

    ax1.set_ylabel("Fidelity", fontweight="bold")
    ax1.set_ylim(0.0, 1.05)

    # Robustness gap on a twin axis
    ax1b = ax1.twinx()
    ax1b.plot(
        gens_plot, gap_mean, marker="d", linestyle="--", color="#c0392b",
        label="Robustness Gap (clean - attacked, mean)"
    )
    ax1b.fill_between(
        gens_plot,
        gap_mean - gap_std,
        gap_mean + gap_std,
        color="#c0392b",
        alpha=0.15,
    )
    ax1b.set_ylabel("Robustness Gap", color="#c0392b", fontweight="bold")
    ax1b.tick_params(axis='y', labelcolor="#c0392b")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title(
        f"Co-Evolution Across Seeds (N={len(all_stats)} runs, {n_gens} generations)",
        fontsize=14,
        fontweight="bold",
    )

    # ---------- PANEL 2: Saboteur Intensity (Error Rate) ---------- #
    ax2.plot(
        gens_plot, mean_error_mean, marker="o", linestyle="-",
        color="#8e44ad", label="Mean Error Rate (over seeds)"
    )
    ax2.fill_between(
        gens_plot,
        mean_error_mean - mean_error_std,
        mean_error_mean + mean_error_std,
        color="#8e44ad",
        alpha=0.2,
    )
    ax2.set_ylabel("Error Rate", fontweight="bold")
    ax2.legend(loc="upper left")

    # ---------- PANEL 3: Architect Complexity (Gate Count) ---------- #
    ax3.plot(
        gens_plot, mean_complex_mean, marker="^", linestyle="-",
        color="#3498db", label="Mean Gate Count (over seeds)"
    )
    ax3.fill_between(
        gens_plot,
        mean_complex_mean - mean_complex_std,
        mean_complex_mean + mean_complex_std,
        color="#3498db",
        alpha=0.2,
    )
    ax3.set_ylabel("Gate Count", fontweight="bold")
    ax3.set_xlabel("Generation", fontweight="bold")
    ax3.legend(loc="upper left")

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
