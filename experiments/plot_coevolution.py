import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
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
    if data is None or len(data) < window_size:
        return data
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def plot_coevolution_corrected(run_dir, save_name, window=100):
    print(f"--- Generating Corrected Co-Evolution Plot for {run_dir} ---")
    
    arch_fid_path = os.path.join(run_dir, "architect_ghz_fidelities.txt")
    sab_fid_path = os.path.join(run_dir, "saboteur_trained_on_architect_ghz_fidelities.txt")

    raw_clean_fidelity = load_data(arch_fid_path)
    noisy_fidelity = load_data(sab_fid_path)

    if raw_clean_fidelity is None or noisy_fidelity is None:
        print("Error: Missing data files.")
        return

    # --- THE FIX ---
    # Plot "Best Circuit Found So Far"
    best_clean_fidelity = np.maximum.accumulate(raw_clean_fidelity)

    # Align Time Axes
    x_arch = np.linspace(0, 100, len(best_clean_fidelity))
    x_sab = np.linspace(0, 100, len(noisy_fidelity))

    # Smooth the Saboteur
    noisy_smooth = smooth(noisy_fidelity, window)
    x_sab_smooth = np.linspace(0, 100, len(noisy_smooth))
    
    # Calculate Saboteur Reward
    saboteur_reward = 1.0 - noisy_fidelity
    reward_smooth = smooth(saboteur_reward, window)

    # --- Calculate Stats for Annotation ---
    # Get average of last 10% of training
    last_10_percent_idx = int(len(noisy_smooth) * 0.9)
    final_avg_robustness = np.mean(noisy_smooth[last_10_percent_idx:])
    final_avg_damage = np.mean(reward_smooth[last_10_percent_idx:])

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # TOP PANEL: True Robustness Gap
    ax1.plot(x_arch, best_clean_fidelity, color="#2ecc71", linewidth=3, label="Clean Circuit (Best So Far)")
    ax1.plot(x_sab_smooth, noisy_smooth, color="#e74c3c", linewidth=2, alpha=0.9, label="Fidelity Under Attack")

    # Fill the gap
    clean_interp = np.interp(x_sab_smooth, x_arch, best_clean_fidelity)
    ax1.fill_between(x_sab_smooth, clean_interp, noisy_smooth, color="#e74c3c", alpha=0.15, label="Susceptibility (Damage)")

    # ANNOTATION 1: Final Score
    ax1.text(98, final_avg_robustness + 0.02, f"Final Robustness: {final_avg_robustness:.4f}", 
             fontsize=12, fontweight='bold', color='#c0392b', ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#c0392b', boxstyle='round,pad=0.5'))

    ax1.set_ylabel("Fidelity", fontsize=12, fontweight='bold')
    ax1.set_ylim(0.5, 1.05) 
    ax1.set_title("The Co-Evolutionary War: True Robustness", fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.7) # Lighter grid

    # BOTTOM PANEL: Saboteur Learning Curve
    ax2.plot(x_sab_smooth, reward_smooth, color="#8e44ad", linewidth=2.5, label="Saboteur Reward (Damage Done)")
    ax2.fill_between(x_sab_smooth, reward_smooth, 0, color="#8e44ad", alpha=0.1)

    # ANNOTATION 2: Damage Score
    ax2.text(98, final_avg_damage + 0.01, f"Avg Damage: {final_avg_damage:.4f}", 
             fontsize=12, fontweight='bold', color='#8e44ad', ha='right', va='bottom')

    ax2.set_ylabel("Saboteur Reward\n(1 - Fidelity)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Experiment Progress (%)", fontsize=12, fontweight='bold')
    
    max_reward = np.max(reward_smooth) if len(reward_smooth) > 0 else 0.5
    ax2.set_ylim(0.0, max(0.1, max_reward * 1.3)) # Give it some headroom
    
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.7)

    out_path = os.path.join(run_dir, save_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Corrected plot saved to: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default="coevolution_corrected.png")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    plot_coevolution_corrected(args.run_dir, args.out, args.window)

