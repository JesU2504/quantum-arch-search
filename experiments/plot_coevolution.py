"""
Co-evolution plotting utilities for adversarial quantum architecture search.
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
        # Ensure 1D array
        if data.ndim > 1:
            data = data[:, 0]
        return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def compute_rolling_stats(data, window_size):
    """
    Compute rolling mean and standard deviation.
    Returns (mean_array, std_array).
    """
    if data is None:
        return np.array([]), np.array([])
    
    n = len(data)
    if n < window_size:
        # Fallback for short runs: no smoothing, return raw data and zero std
        return data, np.zeros_like(data)
    
    # Efficient calculation using convolution for mean
    window = np.ones(window_size) / window_size
    means = np.convolve(data, window, mode='valid')
    
    # Calculate std deviation
    stds = np.zeros(len(means))
    for i in range(len(means)):
        window_data = data[i:i + window_size]
        stds[i] = np.std(window_data)
    
    return means, stds

def plot_coevolution_corrected(run_dir, save_name, window=100, show_error_bands=True):
    """
    Generate co-evolution plot with corrected filenames and proper variable definitions.
    """
    print(f"--- Generating Corrected Co-Evolution Plot for {run_dir} ---")
    
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    arch_fid_path = os.path.join(run_dir, "architect_fidelities.txt")
    sab_fid_path = os.path.join(run_dir, "saboteur_trained_on_architect_fidelities.txt")

    raw_clean_fidelity = load_data(arch_fid_path)
    noisy_fidelity = load_data(sab_fid_path)

    if raw_clean_fidelity is None or noisy_fidelity is None:
        print("Error: Missing data files. Check if 'architect_fidelities.txt' exists.")
        return

    # ---------------------------------------------------------
    # 2. PREPARE VARIABLES (Step-by-Step Definition)
    # ---------------------------------------------------------
    # A. Sample sizes
    n_arch = len(raw_clean_fidelity)
    n_sab = len(noisy_fidelity)

    # B. Architect Metrics
    # "Best Circuit Found So Far" (Cumulative Max)
    best_clean_fidelity = np.maximum.accumulate(raw_clean_fidelity)
    # X-axis for Architect (0% to 100%)
    x_arch = np.linspace(0, 100, len(best_clean_fidelity))

    # C. Saboteur Fidelity Metrics (Top Panel)
    # Smooth the noisy fidelity
    noisy_smooth, noisy_std = compute_rolling_stats(noisy_fidelity, window)
    # X-axis for Saboteur Fidelity
    x_sab = np.linspace(0, 100, len(noisy_fidelity))
    x_sab_smooth = np.linspace(0, 100, len(noisy_smooth))

    # D. Saboteur Reward Metrics (Bottom Panel)
    # Explicitly define saboteur_reward first
    saboteur_reward = 1.0 - noisy_fidelity
    
    # Now compute smoothing on that defined variable
    reward_smooth, reward_std = compute_rolling_stats(saboteur_reward, window)
    
    # Define X-axes for the Reward plot
    x_sab_reward_raw = np.linspace(0, 100, len(saboteur_reward))
    x_reward_smooth = np.linspace(0, 100, len(reward_smooth))

    # ---------------------------------------------------------
    # 3. PLOTTING
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- TOP PANEL: True Robustness Gap ---
    # 1. Raw Data (Faint)
    ax1.scatter(x_arch, raw_clean_fidelity, alpha=0.1, s=2, color="#2ecc71", label="_nolegend_")
    ax1.scatter(x_sab, noisy_fidelity, alpha=0.1, s=2, color="#e74c3c", label="_nolegend_")
    
    # 2. Smooth Curves
    ax1.plot(x_arch, best_clean_fidelity, color="#2ecc71", linewidth=3, label="Clean Circuit (Best So Far)")
    ax1.plot(x_sab_smooth, noisy_smooth, color="#e74c3c", linewidth=2, alpha=0.9, label="Fidelity Under Attack (mean)")
    
    # 3. Error Bands
    if show_error_bands and len(noisy_std) > 0:
        ax1.fill_between(x_sab_smooth, noisy_smooth - noisy_std, noisy_smooth + noisy_std, 
                        color="#e74c3c", alpha=0.2, label="±1 std")

    # 4. Robustness Gap Fill
    # Interpolate clean fidelity to match saboteur x-axis for filling area
    clean_interp = np.interp(x_sab_smooth, x_arch, best_clean_fidelity)
    ax1.fill_between(x_sab_smooth, clean_interp, noisy_smooth, color="#e74c3c", alpha=0.1, label="Susceptibility Gap")

    ax1.set_ylabel("Fidelity", fontsize=12, fontweight='bold')
    ax1.set_ylim(0.0, 1.05) 
    ax1.set_title(f"Co-Evolutionary Robustness (n_arch={n_arch}, n_sab={n_sab})", fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.7)

    # --- BOTTOM PANEL: Saboteur Learning Curve ---
    # 1. Raw Data (Faint)
    # Uses 'x_sab_reward_raw' and 'saboteur_reward' defined in Section 2D
    ax2.scatter(x_sab_reward_raw, saboteur_reward, alpha=0.1, s=2, color="#8e44ad", label="_nolegend_")
    
    # 2. Smooth Curve
    # Uses 'x_reward_smooth' and 'reward_smooth' defined in Section 2D
    ax2.plot(x_reward_smooth, reward_smooth, color="#8e44ad", linewidth=2.5, label="Saboteur Reward (mean)")
    
    # 3. Error Bands
    # Uses 'reward_std' defined in Section 2D
    if show_error_bands and len(reward_std) > 0:
        ax2.fill_between(x_reward_smooth, reward_smooth - reward_std, reward_smooth + reward_std,
                        color="#8e44ad", alpha=0.2, label="±1 std")

    ax2.set_ylabel("Saboteur Reward\n(1 - Fidelity)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Experiment Progress (%)", fontsize=12, fontweight='bold')
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.7)

    out_path = os.path.join(run_dir, save_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Corrected plot saved to: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate co-evolution plots")
    parser.add_argument("--run-dir", required=True, help="Directory containing training data")
    parser.add_argument("--out", default="coevolution_corrected.png", help="Output filename")
    parser.add_argument("--window", type=int, default=50, help="Smoothing window size")
    parser.add_argument("--no-error-bands", action="store_true", help="Disable error bands")
    args = parser.parse_args()

    plot_coevolution_corrected(args.run_dir, args.out, args.window, show_error_bands=not args.no_error_bands)