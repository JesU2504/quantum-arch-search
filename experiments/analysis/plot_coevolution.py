"""
Single-seed co-evolution plot, styled like train_architect:
- Best clean fidelity so far
- EMA (clean)
- EMA (attacked/noisy, aligned to clean steps)
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cirq
from pathlib import Path

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


def exponential_moving_average(data, alpha):
    """EMA smoother (mirrors train_architect plotting)."""
    if data is None or len(data) == 0:
        return np.array([])
    ema = np.empty_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def find_run_dir(root_dir: str, seed: int | None):
    """
    Resolve the directory that contains fidelity logs for a single seed.

    Supported layouts:
      - <root>/adversarial/seed_k/                        (files live here)
      - <root>/seed_k/                                    (files live here)
      - <root>/seed_k/adversarial_training_*/             (older layout)
      - <root>/adversarial/seed_k/adversarial_training_*  (older layout)
      - Passing the seed directory itself as --root-dir.
    """
    required = (
        "architect_fidelities.txt",
        "saboteur_trained_on_architect_fidelities.txt",
    )

    def has_logs(path: str) -> bool:
        return all(os.path.exists(os.path.join(path, f)) for f in required)

    def latest_adv_training(base: str) -> str | None:
        if not os.path.isdir(base):
            return None
        candidates = [
            os.path.join(base, d)
            for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and d.startswith("adversarial_training")
        ]
        candidates.sort()
        return candidates[-1] if candidates else None

    root_dir = os.path.abspath(root_dir)

    # If the user already pointed to the directory with logs, return it directly.
    if has_logs(root_dir):
        return root_dir

    seed_dirs = []
    if seed is not None:
        seed_names = [f"seed_{seed}", f"seed{seed}"]
        # Include the root itself if it already looks like a seed directory.
        if os.path.basename(root_dir) in seed_names and os.path.isdir(root_dir):
            seed_dirs.append(root_dir)

        for base in (root_dir, os.path.join(root_dir, "adversarial")):
            for seed_name in seed_names:
                candidate = os.path.join(base, seed_name)
                if os.path.isdir(candidate):
                    seed_dirs.append(candidate)
    else:
        # No explicit seed provided: scan for the first seed-like directory.
        for base in (root_dir, os.path.join(root_dir, "adversarial")):
            if not os.path.isdir(base):
                continue
            for entry in sorted(os.listdir(base)):
                if entry.startswith("seed") and os.path.isdir(os.path.join(base, entry)):
                    seed_dirs.append(os.path.join(base, entry))

    for seed_dir in seed_dirs:
        if has_logs(seed_dir):
            return seed_dir
        adv = latest_adv_training(seed_dir)
        if adv and has_logs(adv):
            return adv
        adv_nested = latest_adv_training(os.path.join(seed_dir, "adversarial"))
        if adv_nested and has_logs(adv_nested):
            return adv_nested

    return None


def find_robust_circuit_path(run_dir: str) -> str | None:
    """Heuristic search for the robust circuit used in comparisons."""
    run_dir = os.path.abspath(run_dir)
    parents = [
        run_dir,
        os.path.dirname(run_dir),
        os.path.dirname(os.path.dirname(run_dir)),
    ]

    candidates: list[str] = []
    for base in parents:
        if not base:
            continue
        for name in ("circuit_robust.json", "circuit_robust_final.json", "robust_circuit.json"):
            candidates.append(os.path.join(base, name))
        candidates.append(os.path.join(base, "compare", "run_0", "circuit_robust.json"))

    seen = set()
    deduped = []
    for c in candidates:
        if c and c not in seen:
            deduped.append(c)
            seen.add(c)

    for path in deduped:
        if os.path.exists(path):
            return path
    return None


def load_gate_count(circuit_path: str) -> int | None:
    try:
        circuit_text = Path(circuit_path).read_text()
        circuit = cirq.read_json(json_text=circuit_text)
        return len(list(circuit.all_operations()))
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"[plot_coevolution] Warning: failed to read circuit {circuit_path}: {exc}")
        return None


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

    # EMA smoothing (alpha mirrors train_architect; default fixed alpha=0.01 unless overridden)
    def get_alpha():
        return window_frac if window_frac and window_frac > 0 else 0.01

    alpha_clean = get_alpha()
    alpha_noisy = get_alpha()

    clean_ema = exponential_moving_average(clean_fidelity, alpha_clean)
    noisy_ema = exponential_moving_average(noisy_fidelity, alpha_noisy)

    clean_x = clean_steps
    noisy_x = noisy_steps
    # Align noisy EMA to clean steps to show both curves over same iterations
    if len(noisy_ema) > 0 and len(noisy_x) > 0:
        noisy_ema_aligned = np.interp(
            clean_x, noisy_x, noisy_ema,
            left=noisy_ema[0], right=noisy_ema[-1]
        )
    else:
        noisy_ema_aligned = np.array([])

    best_clean = np.maximum.accumulate(clean_fidelity)
    best_attacked = np.max(noisy_fidelity[-1000:])
    best_attacked_series = np.full(len(clean_x), best_attacked, dtype=float) if len(clean_x) else np.array([])

    # Thin curves to reduce visual clutter (similar density to train_architect EMA plots)
    def thin_curve(x_arr, y_arr, max_points=5000):
        if len(x_arr) <= max_points:
            return x_arr, y_arr
        idx = np.linspace(0, len(x_arr) - 1, max_points).astype(int)
        return x_arr[idx], y_arr[idx]

    clean_steps_thin, best_clean_thin = thin_curve(clean_steps, best_clean)
    clean_x_thin, clean_ema_thin = thin_curve(clean_x, clean_ema) if len(clean_ema) else (clean_x, clean_ema)
    _, best_attacked_thin = thin_curve(clean_x, best_attacked_series) if len(best_attacked_series) else (clean_x, best_attacked_series)
    _, noisy_ema_thin = thin_curve(clean_x, noisy_ema_aligned) if len(noisy_ema_aligned) else (clean_x, noisy_ema_aligned)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))


    ax.plot(clean_steps_thin, best_clean_thin, color=COLORS["best"], linewidth=2.0, label="Best Clean So Far")
    if len(clean_ema_thin) > 0:
        ax.plot(clean_x_thin, clean_ema_thin, color="#e74c3c", linewidth=2.0, linestyle="--", label=f"EMA (clean, alpha={alpha_clean:.3f})")
    if len(best_attacked_thin) > 0:
        ax.plot(clean_x_thin, best_attacked_thin, color="#c0392b", linewidth=2.0, linestyle="--", label="Best Attacked So Far")
    # if len(noisy_ema_thin) > 0:
    #     ax.plot(clean_x_thin, noisy_ema_thin, color=COLORS["noisy_roll"], linewidth=2.0, label=f"EMA (attacked, alpha={alpha_noisy:.3f})")
    ax.axhline(1.0, color=COLORS["ideal"], linestyle="--", linewidth=1)

    ax.scatter(clean_steps[-1], best_clean[-1], color=COLORS["best"], edgecolor="white", zorder=5, s=40)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(-0.05, 1.05)
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.25)

    robust_circuit_path = find_robust_circuit_path(run_dir)
    robust_gate_count = None
    if robust_circuit_path:
        robust_gate_count = load_gate_count(robust_circuit_path)
        if robust_gate_count is None:
            print(f"[plot_coevolution] Could not infer gate count from {robust_circuit_path}")
        else:
            print(f"[plot_coevolution] Robust circuit {robust_circuit_path} has {robust_gate_count} gates")
    else:
        print("[plot_coevolution] Robust circuit not found for gate count annotation")

    handles, labels = ax.get_legend_handles_labels()
    order = [
        "Best Clean So Far",
        "Best Attacked So Far",
        f"EMA (clean, alpha={alpha_clean:.3f})",
        f"EMA (attacked, alpha={alpha_noisy:.3f})",
    ]
    label_to_handle = {lbl: h for h, lbl in zip(handles, labels)}
    ordered_handles = [label_to_handle[lbl] for lbl in order if lbl in label_to_handle]
    ordered_labels = [lbl for lbl in order if lbl in label_to_handle]
    ax.legend(ordered_handles, ordered_labels, loc="lower right", frameon=True)

    ax.text(
        0.02, 0.95,
        f"Final best clean: {best_clean[-1]:.3f}\n"
        f"Final EMA clean: {clean_ema[-1] if len(clean_ema) else float('nan'):.3f}\n"
        f"Final EMA attacked: {noisy_ema_aligned[-1] if len(noisy_ema_aligned) else float('nan'):.3f}\n"
        f"Best attacked: {best_attacked:.3f}\n"
        f"Robust circuit gates: {robust_gate_count if robust_gate_count is not None else 'N/A'}",
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
                        help="Path to run root, adversarial/ folder, or a specific seed directory")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed index to plot (default: 0)")
    parser.add_argument("--out", type=str, default="coevolution_corrected.png",
                        help="Output path (PNG). If a directory is provided, the plot is saved as coevolution_corrected.png inside it.")
    parser.add_argument("--window-frac", type=float, default=0.01,
                        help="EMA smoothing alpha (default: 0.01).")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        if args.root_dir is None:
            raise SystemExit("Provide either --run-dir or --root-dir with --seed")
        run_dir = find_run_dir(args.root_dir, args.seed)
        if run_dir is None:
            raise SystemExit(f"Could not locate fidelity logs for seed {args.seed} under {args.root_dir}")

    out_path = args.out
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, "coevolution_corrected.png")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_coevolution_single(run_dir, out_path, window_frac=args.window_frac)
