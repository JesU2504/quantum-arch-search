"""
Co-evolution plotting utilities for adversarial quantum architecture search.

This module provides functions for visualizing the co-evolutionary training process
between the Architect and Saboteur agents.

Statistical Protocol:
    - Plots show smoothed curves with raw data overlay (faint individual points)
    - Sample size (n) is annotated on plots
    - Moving average window is configurable for smoothing
    - Shaded regions indicate variability (where applicable)

See README.md for full statistical reporting guidelines.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    """Load numeric data from a text file."""
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    try:
        data = np.loadtxt(path)
        if data.size == 0: return None
        return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def smooth(data, window_size):
    """Apply moving average smoothing to data."""
    if data is None or len(data) < window_size:
        return data
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')


def compute_rolling_stats(data, window_size):
    """
    Compute rolling mean and standard deviation for error bar display.
    
    Args:
        data: Input array of values.
        window_size: Size of the rolling window.
    
    Returns:
        Tuple of (mean_array, std_array) with length = len(data) - window_size + 1.
    """
    if data is None or len(data) < window_size:
        return data, np.zeros_like(data) if data is not None else None
    
    n = len(data)
    means = np.zeros(n - window_size + 1)
    stds = np.zeros(n - window_size + 1)
    
    for i in range(len(means)):
        window_data = data[i:i + window_size]
        means[i] = np.mean(window_data)
        stds[i] = np.std(window_data)
    
    return means, stds


def plot_coevolution_corrected(run_dir, save_name, window=100, show_error_bands=True):
    """
    Generate corrected co-evolution plot with statistical annotations.
    
    Shows the "true robustness gap" between clean circuit performance and
    fidelity under saboteur attack. Includes:
    - Sample size annotations (n=...)
    - Smoothed curves with optional error bands
    - Individual raw data points as faint overlay
    
    Args:
        run_dir: Directory containing training data files.
        save_name: Output filename for the plot.
        window: Window size for moving average smoothing.
        show_error_bands: Whether to show ±1 std error bands.
    """
    print(f"--- Generating Corrected Co-Evolution Plot for {run_dir} ---")
    
    arch_fid_path = os.path.join(run_dir, "architect_ghz_fidelities.txt")
    sab_fid_path = os.path.join(run_dir, "saboteur_trained_on_architect_ghz_fidelities.txt")

    raw_clean_fidelity = load_data(arch_fid_path)
    noisy_fidelity = load_data(sab_fid_path)

    if raw_clean_fidelity is None or noisy_fidelity is None:
        print("Error: Missing data files.")
        return

    # Get sample sizes for annotation
    n_arch = len(raw_clean_fidelity)
    n_sab = len(noisy_fidelity)

    # Plot "Best Circuit Found So Far"
    best_clean_fidelity = np.maximum.accumulate(raw_clean_fidelity)

    # Align Time Axes
    x_arch = np.linspace(0, 100, len(best_clean_fidelity))
    x_sab = np.linspace(0, 100, len(noisy_fidelity))

    # Smooth the Saboteur with rolling statistics
    noisy_smooth, noisy_std = compute_rolling_stats(noisy_fidelity, window)
    x_sab_smooth = np.linspace(0, 100, len(noisy_smooth))
    
    # Calculate Saboteur Reward
    saboteur_reward = 1.0 - noisy_fidelity
    reward_smooth, reward_std = compute_rolling_stats(saboteur_reward, window)

    # --- Calculate Stats for Annotation ---
    # Get average of last 10% of training
    last_10_percent_idx = int(len(noisy_smooth) * 0.9)
    final_avg_robustness = np.mean(noisy_smooth[last_10_percent_idx:])
    final_std_robustness = np.mean(noisy_std[last_10_percent_idx:])
    final_avg_damage = np.mean(reward_smooth[last_10_percent_idx:])
    final_std_damage = np.mean(reward_std[last_10_percent_idx:])

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # TOP PANEL: True Robustness Gap
    # Plot faint individual data points
    ax1.scatter(x_arch, raw_clean_fidelity, alpha=0.1, s=2, color="#2ecc71", label="_nolegend_")
    ax1.scatter(x_sab, noisy_fidelity, alpha=0.1, s=2, color="#e74c3c", label="_nolegend_")
    
    ax1.plot(x_arch, best_clean_fidelity, color="#2ecc71", linewidth=3, label="Clean Circuit (Best So Far)")
    ax1.plot(x_sab_smooth, noisy_smooth, color="#e74c3c", linewidth=2, alpha=0.9, label="Fidelity Under Attack (mean)")
    
    # Add error band if requested
    if show_error_bands and len(noisy_std) > 0:
        ax1.fill_between(x_sab_smooth, noisy_smooth - noisy_std, noisy_smooth + noisy_std, 
                        color="#e74c3c", alpha=0.2, label="±1 std")

    # Fill the robustness gap
    clean_interp = np.interp(x_sab_smooth, x_arch, best_clean_fidelity)
    ax1.fill_between(x_sab_smooth, clean_interp, noisy_smooth, color="#e74c3c", alpha=0.1, label="Susceptibility (Damage)")

    # ANNOTATION 1: Final Score with error
    ax1.text(98, final_avg_robustness + 0.02, 
             f"Final Robustness: {final_avg_robustness:.4f}±{final_std_robustness:.4f}\n(n={n_sab})", 
             fontsize=11, fontweight='bold', color='#c0392b', ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#c0392b', boxstyle='round,pad=0.5'))

    ax1.set_ylabel("Fidelity", fontsize=12, fontweight='bold')
    ax1.set_ylim(0.5, 1.05) 
    ax1.set_title(f"The Co-Evolutionary War: True Robustness (n_arch={n_arch}, n_sab={n_sab}, window={window})", 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.7)

    # BOTTOM PANEL: Saboteur Learning Curve
    # Plot faint individual data points
    x_sab_reward = np.linspace(0, 100, len(saboteur_reward))
    ax2.scatter(x_sab_reward, saboteur_reward, alpha=0.1, s=2, color="#8e44ad", label="_nolegend_")
    
    x_reward_smooth = np.linspace(0, 100, len(reward_smooth))
    ax2.plot(x_reward_smooth, reward_smooth, color="#8e44ad", linewidth=2.5, label="Saboteur Reward (mean)")
    
    # Add error band if requested
    if show_error_bands and len(reward_std) > 0:
        ax2.fill_between(x_reward_smooth, reward_smooth - reward_std, reward_smooth + reward_std,
                        color="#8e44ad", alpha=0.2, label="±1 std")
    else:
        ax2.fill_between(x_reward_smooth, reward_smooth, 0, color="#8e44ad", alpha=0.1)

    # ANNOTATION 2: Damage Score with error
    ax2.text(98, final_avg_damage + 0.01, 
             f"Avg Damage: {final_avg_damage:.4f}±{final_std_damage:.4f}", 
             fontsize=11, fontweight='bold', color='#8e44ad', ha='right', va='bottom')

    ax2.set_ylabel("Saboteur Reward\n(1 - Fidelity)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Experiment Progress (%)", fontsize=12, fontweight='bold')
    
    max_reward = np.max(reward_smooth) if len(reward_smooth) > 0 else 0.5
    ax2.set_ylim(0.0, max(0.1, max_reward * 1.3))
    
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.7)

    out_path = os.path.join(run_dir, save_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Corrected plot saved to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate co-evolution plots with statistical annotations"
    )
    parser.add_argument("--run-dir", required=True, help="Directory containing training data")
    parser.add_argument("--out", default="coevolution_corrected.png", help="Output filename")
    parser.add_argument("--window", type=int, default=50, help="Smoothing window size")
    parser.add_argument("--no-error-bands", action="store_true", 
                        help="Disable error bands on plots")
    args = parser.parse_args()

    plot_coevolution_corrected(args.run_dir, args.out, args.window, 
                               show_error_bands=not args.no_error_bands)
