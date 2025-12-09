"""
Default configuration for VQE pipelines.

Quick full-run command (all baselines + robustness/cross-noise/hw-like eval):
    python run_vqe.py --molecule H2 --n-seeds 3 --run-adv-architect --run-hw-eval

Separated from experiments/config.py to avoid cross-talk with the state-prep/unitary
experiments. Provides a single place to tune defaults for RL, HEA, and adversarial
architect VQE baselines plus robustness evaluation settings.
"""

# Supported molecules for convenience
MOLECULES = ["H2", "HeH+", "LiH", "BeH2"]

# --- RL Architect (PPO on ArchitectEnv, task_mode="vqe") ---
RL = {
    "max_gates": 12,
    "total_timesteps": 10_000,
    "complexity_penalty": 0.01,
    "lr": 3e-4,
}

# --- HEA adversarial baseline (TorchQuantum) ---
HEA = {
    "steps": 200,
    "layers": 3,
    "noise_levels": [0.0, 0.02, 0.05],
    "lr": 0.1,
    "noise_samples_per_step": 0,
}

# --- Architect adversarial VQE (ArchitectEnv + saboteur) ---
ADV_ARCH = {
    "n_generations": 5,
    "architect_steps_per_gen": 12000,
    "saboteur_steps_per_gen": 4096,
    "max_gates": 15,
    "complexity_penalty": 0.01,
    "alpha_start": 0.6,
    "alpha_end": 0.0,
    "saboteur_budget": 3,
    "saboteur_noise_family": "depolarizing",
    "saboteur_error_rates": [0.0, 0.005, 0.01, 0.02, 0.05],
}

# --- Robustness evaluation ---
# Noise families and a single representative "realistic" rate per family.
ROBUSTNESS = {
    "families": ["depolarizing", "amplitude_damping", "coherent_overrotation", "readout"],
    "rates": {
        "depolarizing": 0.02,
        "amplitude_damping": 0.02,
        "coherent_overrotation": 0.02,
        "readout": 0.03,
    },
    "shots": 128,  # if a sampling-based simulator is used in the future
}

# --- Cross-noise sweep (parity with state-prep) ---
CROSS_NOISE = {
    "families": ["depolarizing", "amplitude_damping", "coherent_overrotation", "readout"],
    "rates": [0.0, 0.01, 0.02, 0.05],
}

# Hardware eval placeholders (kept for parity; not wired to qiskit_hw_eval)
HW = {
    "backends": ["fake_quito", "fake_belem", "fake_athens", "fake_lagos", "fake_oslo"],
    "shots": 4096,
    "opt_level": 3,
}
