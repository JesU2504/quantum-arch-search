import argparse
import os
import numpy as np
from typing import Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt

# --- Data Loading Helpers ---

def load_fidelities(path: str) -> Optional[np.ndarray]:
    """Load fidelity values from a text file, handling errors gracefully."""
    if not path or not os.path.exists(path):
        return None

    vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                # Strip whitespace and check if line is not empty
                clean_line = line.strip()
                if clean_line:
                    vals.append(float(clean_line))
            except ValueError:
                continue

    return np.array(vals, dtype=float) if vals else None


def load_steps(path: str) -> Optional[np.ndarray]:
    """Load step values from a text file."""
    if not path or not os.path.exists(path):
        return None

    vals: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                clean_line = line.strip()
                if clean_line:
                    vals.append(int(clean_line))
            except ValueError:
                continue

    return np.array(vals, dtype=int) if vals else None


def find_file(root: str, candidates: Iterable[str]) -> Optional[str]:
    """Find the first existing file from a list of relative paths."""
    for rel in candidates:
        full_path = os.path.join(root, rel)
        if os.path.exists(full_path):
            return full_path
    return None


def load_curriculum_changes(run_dir: str) -> List[Tuple[int, int]]:
    """Load curriculum changes (level, step) from the run directory."""
    candidates = [
        "saboteur_curriculum_changes.txt",
        "saboteur/saboteur_curriculum_changes.txt",
    ]
    changes = []
    for candidate in candidates:
        path = os.path.join(run_dir, candidate)
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            level, step = map(int, parts[:2])
                            changes.append((level, step))
                    except ValueError:
                        continue
            break
    return changes


# --- Analysis Helpers ---

def smooth_and_align(x: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters NaNs, applies moving average, and aligns the X-axis.
    Falls back to raw data if the cleaned data is shorter than the window.
    """
    # 1. Filter NaNs
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # 2. Check sufficiency
    if len(y_clean) < window or window <= 1:
        return x_clean, y_clean

    # 3. Convolve (Moving Average)
    # mode='valid' returns an array of length N - K + 1
    y_smooth = np.convolve(y_clean, np.ones(window) / window, mode='valid')

    # 4. Align X-axis
    # We slice the X array to match the central part of the convolution
    start_idx = (window - 1) // 2
    end_idx = start_idx + len(y_smooth)
    x_smooth = x_clean[start_idx:end_idx]

    return x_smooth, y_smooth


# --- Main Plotting Logic ---

def plot_coevolution(
    run_dir: str,
    save_name: str = "coevolutionary_process.png",
    window: int = 50,
    stride: int = 20,
    smooth: bool = True,
) -> None:
    """Plot the co-evolutionary process of Architect and Saboteur."""

    # 1. Locate files
    arch_file = find_file(run_dir, ["architect_ghz_fidelities.txt", "baseline/architect_ghz_fidelities.txt"])
    sab_file = find_file(run_dir, ["saboteur_trained_on_architect_ghz_fidelities.txt", "saboteur/saboteur_trained_on_architect_ghz_fidelities.txt"])
    
    arch_steps_file = find_file(run_dir, ["architect_ghz_steps.txt", "baseline/architect_ghz_steps.txt"])
    sab_steps_file = find_file(run_dir, ["saboteur_trained_on_architect_ghz_steps.txt", "saboteur/saboteur_trained_on_architect_ghz_steps.txt"])

    # 2. Load data
    arch_fidelities = load_fidelities(arch_file)
    sab_fidelities = load_fidelities(sab_file)
    arch_steps = load_steps(arch_steps_file)
    sab_steps = load_steps(sab_steps_file)

    if arch_fidelities is None and sab_fidelities is None:
        print(f"No fidelity traces found in {run_dir}. Exiting.")
        return

    # 3. Construct X-axes
    # Architect
    if arch_fidelities is not None:
        if arch_steps is not None and arch_steps.size == arch_fidelities.size:
            arch_x = arch_steps.astype(float)
        else:
            arch_x = np.arange(1, arch_fidelities.size + 1, dtype=float)

    # Saboteur
    x_label = "Episode Index"
    if sab_fidelities is not None:
        if sab_steps is not None and sab_steps.size == sab_fidelities.size:
            sab_x = sab_steps.astype(float)
            x_label = "Total Iteration"
        else:
            sab_x = np.arange(1, sab_fidelities.size + 1, dtype=float)


    # 4. Initialize Plot
    fig, (ax_arch, ax_sab) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    fig.patch.set_facecolor('#f7f7fa')
    ax_arch.set_facecolor('#f7f7fa')
    ax_sab.set_facecolor('#f7f7fa')


    # 5. Plot Architect (Top)
    if arch_fidelities is not None and len(arch_x) > 0:
        # Smooth
        if smooth:
            curr_x, curr_y = smooth_and_align(arch_x, arch_fidelities, window)
        else:
            curr_x, curr_y = arch_x, arch_fidelities
        
        # Stride (downsample for performance/visuals)
        if stride > 1:
            curr_x = curr_x[::stride]
            curr_y = curr_y[::stride]

        if len(curr_y) > 0:
            ax_arch.plot(curr_x, curr_y, color="tab:blue", label="Clean Fidelity (Architect)", linewidth=2)
            # Add markers if data is sparse to make it visible
            if len(curr_y) < 100:
                ax_arch.scatter(curr_x, curr_y, color="tab:blue", s=20, edgecolor='k')
        else:
            ax_arch.text(0.5, 0.5, 'Data insufficient', transform=ax_arch.transAxes, ha='center', color='gray')
    else:
        ax_arch.text(0.5, 0.5, 'No Architect data', transform=ax_arch.transAxes, ha='center', color='gray')

    ax_arch.set_ylabel("Clean Fidelity", fontsize=14)
    ax_arch.set_ylim(-0.05, 1.05)
    ax_arch.grid(True, linestyle="--", alpha=0.5)
    ax_arch.legend(loc='lower right')


    # 6. Plot Saboteur (Bottom)
    if sab_fidelities is not None and len(sab_x) > 0:
        # Smooth
        if smooth:
            curr_x, curr_y = smooth_and_align(sab_x, sab_fidelities, window)
        else:
            curr_x, curr_y = sab_x, sab_fidelities

        # Stride
        if stride > 1:
            curr_x = curr_x[::stride]
            curr_y = curr_y[::stride]

        if len(curr_y) > 0:
            ax_sab.plot(curr_x, curr_y, color="tab:orange", linewidth=2, label="Fidelity Under Attack")
            
            # Effectiveness (1 - Fidelity)
            effectiveness = 1.0 - curr_y
            ax_sab.plot(curr_x, effectiveness, color="tab:red", linewidth=2, linestyle="--", label="Saboteur Effectiveness")
        else:
            ax_sab.text(0.5, 0.5, 'Data insufficient', transform=ax_sab.transAxes, ha='center', color='gray')
    else:
        ax_sab.text(0.5, 0.5, 'No Saboteur data', transform=ax_sab.transAxes, ha='center', color='gray')

    # Curriculum Annotations
    curriculum_changes = load_curriculum_changes(run_dir)
    if curriculum_changes:
        y_text = ax_sab.get_ylim()[1] - 0.05
        # Only plot vertical lines if we have data to attach them to
        if sab_fidelities is not None:
            for iteration, level in curriculum_changes:
                ax_sab.axvline(iteration, color="tab:green", linestyle=":", linewidth=1.5, alpha=0.8)
                ax_sab.text(iteration, y_text, f" Level {level}", rotation=90, color="tab:green", va="top", fontsize=10)

    ax_sab.set_ylabel("Saboteur Metrics", fontsize=14)
    ax_sab.set_ylim(-0.05, 1.05)
    ax_sab.grid(True, linestyle="--", alpha=0.5)
    ax_sab.legend(loc='center right')
    ax_sab.set_xlabel(x_label, fontsize=14)

    plt.suptitle("Coevolutionary Process: Architect vs Saboteur", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    # 7. Save
    output_path = os.path.join(run_dir, save_name)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot coevolutionary fidelity traces")
    parser.add_argument("--run-dir", required=True, help="Directory containing fidelity trace files")
    parser.add_argument("--out", "-o", default="coevolutionary_process.png", help="Output filename")
    parser.add_argument("--window", "-w", type=int, default=500, help="Moving-average window size")
    parser.add_argument("--stride", "-s", type=int, default=20, help="Downsampling stride")
    
    args = parser.parse_args()
    plot_coevolution(args.run_dir, save_name=args.out, window=args.window, stride=args.stride)