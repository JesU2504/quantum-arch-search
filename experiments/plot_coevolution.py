"""
Co-evolution plotting utilities for adversarial quantum architecture search.

This version emphasizes GENERATION-LEVEL dynamics:

- Architect: clean fidelity, circuit complexity per generation.
- Saboteur: fidelity under attack (damage) and error rate per generation.

It uses:
- architect_fidelities.txt
- architect_steps.txt
- saboteur_trained_on_architect_fidelities.txt
- saboteur_trained_on_architect_steps.txt
- saboteur_error_rates.txt
- architect_complexity.txt

and (optionally) metadata.json two levels up from run_dir.
"""

import argparse
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


# ----------------- Basics ----------------- #

def load_data(path):
    """Load numeric data from a text file, or return None."""
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


def compute_rolling_stats(data, window_size):
    """Compute rolling mean and standard deviation for smoothing."""
    if data is None:
        return np.array([]), np.array([])
    n = len(data)
    if n < window_size:
        return data, np.zeros_like(data)
    window = np.ones(window_size) / window_size
    means = np.convolve(data, window, mode="valid")
    stds = np.zeros(len(means))
    for i in range(len(means)):
        window_data = data[i : i + window_size]
        stds[i] = np.std(window_data)
    return means, stds


def load_metadata_from_run_dir(run_dir):
    """
    Try to load metadata.json from results/run_XXXX/... structure.

    Assumes run_dir is something like:
        results/run_XXXX/adversarial/adversarial_training_YYYY
    """
    base_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))
    meta_path = os.path.join(base_run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"[plot_coevolution] metadata.json not found at {meta_path} (using defaults)")
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta
    except Exception as e:
        print(f"[plot_coevolution] Error loading metadata.json: {e}")
        return None


def assign_generations(steps, steps_per_gen, n_gens):
    """
    Given an array of step indices and steps_per_gen,
    assign each step to a generation index in [0, n_gens-1].
    """
    if steps is None:
        return None
    gens = (steps - 1) // steps_per_gen
    gens = np.clip(gens, 0, n_gens - 1).astype(int)
    return gens


# ----------------- Main plotting ----------------- #

def plot_coevolution_dashboard(run_dir, save_name, window=100):
    """
    Generate a 3-panel dashboard:

    PANEL 1 (top):  Architect vs Saboteur per generation
        - Architect best clean fidelity
        - Saboteur mean fidelity under attack
        - "Robustness gap" = difference between them

    PANEL 2 (middle): Saboteur INTENSITY per generation
        - Mean error rate

    PANEL 3 (bottom): Architect STRATEGY per generation
        - Mean circuit gate count (complexity)

    Additionally, lightly show smoothed curves over time (optional)
    for more detailed intra-generation behaviour.
    """
    print(f"--- Generating Co-Evolution Dashboard for {run_dir} ---")

    # 1. LOAD DATA
    arch_fid_path = os.path.join(run_dir, "architect_fidelities.txt")
    arch_steps_path = os.path.join(run_dir, "architect_steps.txt")
    sab_fid_path = os.path.join(run_dir, "saboteur_trained_on_architect_fidelities.txt")
    sab_steps_path = os.path.join(run_dir, "saboteur_trained_on_architect_steps.txt")
    arch_complex_path = os.path.join(run_dir, "architect_complexity.txt")
    sab_error_path = os.path.join(run_dir, "saboteur_error_rates.txt")

    clean_fidelity = load_data(arch_fid_path)
    clean_steps = load_data(arch_steps_path)
    noisy_fidelity = load_data(sab_fid_path)
    noisy_steps = load_data(sab_steps_path)
    complexity = load_data(arch_complex_path)
    error_rate = load_data(sab_error_path)

    if clean_fidelity is None or noisy_fidelity is None:
        print("[plot_coevolution] Error: Critical fidelity data files missing.")
        return

    # 2. METADATA for generation structure
    meta = load_metadata_from_run_dir(run_dir)
    if meta is not None:
        n_gens = int(meta.get("adversarial_gens", 1))
        arch_steps_per_gen = int(meta.get("adversarial_arch_steps", len(clean_steps) // max(1, n_gens)))
        sab_total_steps = int(meta.get("saboteur_steps", len(noisy_steps)))
        sab_steps_per_gen = sab_total_steps // max(1, n_gens)
    else:
        # Fallback: single "generation"
        n_gens = 1
        arch_steps_per_gen = len(clean_steps) if clean_steps is not None else len(clean_fidelity)
        sab_steps_per_gen = len(noisy_steps)

    print(f"[plot_coevolution] n_gens={n_gens}, arch_steps_per_gen={arch_steps_per_gen}, sab_steps_per_gen={sab_steps_per_gen}")

    # 3. ASSIGN GENERATIONS
    if clean_steps is None:
        # If no steps are logged, just treat as evenly spaced
        clean_steps = np.arange(1, len(clean_fidelity) + 1)
    if noisy_steps is None:
        noisy_steps = np.arange(1, len(noisy_fidelity) + 1)

    arch_gens = assign_generations(clean_steps, arch_steps_per_gen, n_gens)
    sab_gens = assign_generations(noisy_steps, sab_steps_per_gen, n_gens)

    # 4. AGGREGATE PER GENERATION
    gen_indices = np.arange(n_gens)

    # Architect per-gen stats
    arch_best_clean = np.zeros(n_gens)
    arch_mean_clean = np.zeros(n_gens)
    arch_mean_complex = np.zeros(n_gens)

    for g in gen_indices:
        mask = (arch_gens == g)
        if not np.any(mask):
            arch_best_clean[g] = np.nan
            arch_mean_clean[g] = np.nan
            arch_mean_complex[g] = np.nan
            continue
        arch_best_clean[g] = np.max(clean_fidelity[mask])
        arch_mean_clean[g] = np.mean(clean_fidelity[mask])
        if complexity is not None:
            # complexity has same length & ordering as clean_fidelity / arch_steps
            arch_mean_complex[g] = np.mean(complexity[mask])
        else:
            arch_mean_complex[g] = np.nan

    # Saboteur per-gen stats
    sab_mean_noisy = np.zeros(n_gens)
    sab_mean_error = np.zeros(n_gens)

    for g in gen_indices:
        mask = (sab_gens == g)
        if not np.any(mask):
            sab_mean_noisy[g] = np.nan
            sab_mean_error[g] = np.nan
            continue
        sab_mean_noisy[g] = np.mean(noisy_fidelity[mask])
        if error_rate is not None:
            sab_mean_error[g] = np.mean(error_rate[mask])
        else:
            sab_mean_error[g] = np.nan

    # 5. OPTIONAL: Smooth curves over entire run (for context)
    clean_smooth, _ = compute_rolling_stats(clean_fidelity, window)
    noisy_smooth, _ = compute_rolling_stats(noisy_fidelity, window)
    x_clean_smooth = np.linspace(0, n_gens, len(clean_smooth))
    x_noisy_smooth = np.linspace(0, n_gens, len(noisy_smooth))

    # 6. PLOTTING
    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]}
    )

    # ---------- PANEL 1: Robustness per generation ---------- #
    # Per-generation points
    gens_plot = gen_indices + 1  # 1-based for readability

    # Architect best clean fidelity
    line_arch, = ax1.plot(
        gens_plot, arch_best_clean, marker="o", linestyle="-", color=COLORS["arch_clean"],
        label="Architect Best Clean Fidelity"
    )
    # Saboteur mean attacked fidelity
    line_sab, = ax1.plot(
        gens_plot, sab_mean_noisy, marker="s", linestyle="-", color=COLORS["sab_noisy"],
        label="Saboteur Mean Fidelity (Under Attack)"
    )

    # Robustness gap = F_clean - F_attacked (per generation)
    gap = arch_best_clean - sab_mean_noisy

    # Optional: light bar to visualize gap as a band
    ax1.bar(
        gens_plot, gap, bottom=sab_mean_noisy,
        width=0.3, alpha=0.15, color="#e74c3c"
    )

    # Light smoothed curves across training (for intra-gen dynamics)
    if len(clean_smooth) > 0:
        ax1.plot(
            x_clean_smooth * (n_gens / max(1, x_clean_smooth.max())),
            clean_smooth, color="#2ecc71", alpha=0.3, linewidth=1, linestyle="--"
        )
    if len(noisy_smooth) > 0:
        ax1.plot(
            x_noisy_smooth * (n_gens / max(1, x_noisy_smooth.max())),
            noisy_smooth, color="#e67e22", alpha=0.3, linewidth=1, linestyle="--"
        )

    ax1.set_ylabel("Fidelity", fontweight="bold")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(gens_plot)

    # --- NEW: twin y-axis for the robustness gap curve --- #
    ax1b = ax1.twinx()
    line_gap, = ax1b.plot(
        gens_plot, gap, marker="d", linestyle="--", color=COLORS["gap"],
        label="Robustness Gap (clean - attacked)"
    )
    ax1b.set_ylabel("Robustness Gap", color=COLORS["gap"], fontweight="bold")
    ax1b.tick_params(axis='y', labelcolor=COLORS["gap"])

    # Combine legends from both y-axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # --- NEW: small text annotation summarizing gap evolution --- #
    if n_gens >= 1 and not np.all(np.isnan(gap)):
        g0 = 0
        g_last = n_gens - 1
        gap0 = gap[g0]
        gap_last = gap[g_last]
        ax1.text(
            0.02, 0.05,
            f"Gap Gen1: {gap0:.3f}\nGap Gen{n_gens}: {gap_last:.3f}",
            transform=ax1.transAxes,
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )

    ax1.set_title(
        f"Co-Evolution per Generation (n_gens={n_gens})",
        fontsize=14, fontweight="bold"
    )


    # ---------- PANEL 2: Saboteur intensity per generation ---------- #
    if not np.all(np.isnan(sab_mean_error)):
        ax2.plot(
            gens_plot, sab_mean_error, marker="d", linestyle="-",
            color=COLORS["error"], label="Mean Error Rate"
        )
        ax2.set_ylabel("Avg Error Rate", fontweight="bold")
        ax2.legend(loc="upper left")
    else:
        ax2.text(0.5, 0.5, "No Error Rate Data", ha="center")

    # ---------- PANEL 3: Architect strategy (complexity) ---------- #
    if not np.all(np.isnan(arch_mean_complex)):
        ax3.plot(
            gens_plot, arch_mean_complex, marker="^", linestyle="-",
            color=COLORS["complexity"], label="Mean Gate Count"
        )
        ax3.set_ylabel("Gate Count", fontweight="bold")
        ax3.legend(loc="upper left")
    else:
        ax3.text(0.5, 0.5, "No Complexity Data", ha="center")

    ax3.set_xlabel("Generation", fontweight="bold", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(run_dir, save_name)
    plt.savefig(out_path, dpi=300)
    print(f"[plot_coevolution] Dashboard saved to: {out_path}")
    plt.close(fig)


# ----------------- CLI ----------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", required=True,
        help="Directory containing adversarial_training_* logs"
    )
    parser.add_argument(
        "--out", default="coevolution_generations.png",
        help="Output filename (saved inside run-dir)"
    )
    parser.add_argument(
        "--window", type=int, default=50,
        help="Smoothing window size for intra-generation curves"
    )
    args = parser.parse_args()

    plot_coevolution_dashboard(args.run_dir, args.out, args.window)
