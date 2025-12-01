"""
Co-evolution plotting utilities for adversarial quantum architecture search.
Generates a multi-panel dashboard for Strategy, Robustness, and Intensity.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    """Load numeric data from a text file."""
    if not os.path.exists(path):
        # Return None so we can handle missing data gracefully
        return None
    try:
        data = np.loadtxt(path)
        if data.size == 0: return None
        if data.ndim > 1: data = data[:, 0]
        return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def compute_rolling_stats(data, window_size):
    """Compute rolling mean and standard deviation."""
    if data is None:
        return np.array([]), np.array([])
    
    n = len(data)
    if n < window_size:
        return data, np.zeros_like(data)
    
    window = np.ones(window_size) / window_size
    means = np.convolve(data, window, mode='valid')
    
    stds = np.zeros(len(means))
    for i in range(len(means)):
        window_data = data[i:i + window_size]
        stds[i] = np.std(window_data)
    
    return means, stds

def plot_coevolution_dashboard(run_dir, save_name, window=100):
    """
    Generate a 3-panel dashboard:
    1. Robustness: Clean vs Noisy Fidelity
    2. Intensity: Saboteur Error Rate
    3. Strategy: Circuit Complexity (Gate Count)
    """
    print(f"--- Generating Co-Evolution Dashboard for {run_dir} ---")
    
    # 1. LOAD DATA
    arch_fid_path = os.path.join(run_dir, "architect_fidelities.txt")
    sab_fid_path = os.path.join(run_dir, "saboteur_trained_on_architect_fidelities.txt")
    arch_complex_path = os.path.join(run_dir, "architect_complexity.txt")
    sab_error_path = os.path.join(run_dir, "saboteur_error_rates.txt")

    clean_fidelity = load_data(arch_fid_path)
    noisy_fidelity = load_data(sab_fid_path)
    complexity = load_data(arch_complex_path)
    error_rate = load_data(sab_error_path)

    if clean_fidelity is None or noisy_fidelity is None:
        print("Error: Critical fidelity data files missing.")
        return

    # 2. PREPARE X-AXES (Normalized 0-100%)
    x_arch = np.linspace(0, 100, len(clean_fidelity))
    x_sab = np.linspace(0, 100, len(noisy_fidelity))
    
    # Best-so-far for clean fidelity
    best_clean = np.maximum.accumulate(clean_fidelity)

    # Smoothing
    noisy_smooth, noisy_std = compute_rolling_stats(noisy_fidelity, window)
    x_sab_smooth = np.linspace(0, 100, len(noisy_smooth))

    # 3. PLOTTING
    sns.set_theme(style="whitegrid")
    # Create 3 subplots sharing X axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 1, 1]})

    # --- PANEL 1: ROBUSTNESS GAP ---
    ax1.plot(x_arch, best_clean, color="#2ecc71", linewidth=3, label="Clean Fidelity (Best)")
    ax1.plot(x_sab_smooth, noisy_smooth, color="#e74c3c", linewidth=2, label="Fidelity Under Attack (Mean)")
    
    # Fill the gap
    clean_interp = np.interp(x_sab_smooth, x_arch, best_clean)
    ax1.fill_between(x_sab_smooth, clean_interp, noisy_smooth, color="#e74c3c", alpha=0.1, label="Robustness Gap")
    
    ax1.set_ylabel("Fidelity", fontweight='bold')
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(loc="lower right")
    ax1.set_title(f"Co-Evolution Dashboard (n_arch={len(clean_fidelity)})", fontsize=14, fontweight='bold')

    # --- PANEL 2: SABOTEUR INTENSITY (Error Rate) ---
    if error_rate is not None:
        err_smooth, err_std = compute_rolling_stats(error_rate, window)
        x_err_smooth = np.linspace(0, 100, len(err_smooth))
        
        ax2.plot(x_err_smooth, err_smooth, color="#8e44ad", linewidth=2, label="Mean Error Rate")
        ax2.fill_between(x_err_smooth, err_smooth - err_std, err_smooth + err_std, color="#8e44ad", alpha=0.2)
        ax2.set_ylabel("Avg Error Rate", fontweight='bold')
        ax2.legend(loc="upper right")
    else:
        ax2.text(0.5, 0.5, "No Error Rate Data Available", ha='center')

    # --- PANEL 3: ARCHITECT STRATEGY (Complexity) ---
    if complexity is not None:
        comp_smooth, comp_std = compute_rolling_stats(complexity, window)
        x_comp_smooth = np.linspace(0, 100, len(comp_smooth))
        
        ax3.plot(x_comp_smooth, comp_smooth, color="#3498db", linewidth=2, label="Circuit Gates (Count)")
        ax3.fill_between(x_comp_smooth, comp_smooth - comp_std, comp_smooth + comp_std, color="#3498db", alpha=0.2)
        ax3.set_ylabel("Gate Count", fontweight='bold')
        ax3.legend(loc="upper right")
    else:
        ax3.text(0.5, 0.5, "No Complexity Data Available", ha='center')

    ax3.set_xlabel("Experiment Progress (%)", fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    out_path = os.path.join(run_dir, save_name)
    plt.savefig(out_path, dpi=300)
    print(f"Dashboard saved to: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Directory containing training data")
    parser.add_argument("--out", default="coevolution_dashboard.png", help="Output filename")
    parser.add_argument("--window", type=int, default=50, help="Smoothing window size")
    args = parser.parse_args()

    plot_coevolution_dashboard(args.run_dir, args.out, args.window)