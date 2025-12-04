"""
Multi-seed adversarial co-evolution training.

This script simply calls train_adversarial() several times with different seeds,
storing each run under results_dir/seed_{k}/...

It does NOT change your existing single-run train_adversarial.py.
"""

import os
import sys
import time
import numpy as np

# Add repo root to path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from experiments.adversarial.train_adversarial import train_adversarial  # reuse your existing function
from experiments import config


def set_global_seed(seed: int):
    """Best-effort global seeding for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/coevolution_multiseed")
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--n-generations", type=int, default=None,
                        help="Override generations. Defaults to config.EXPERIMENT_PARAMS for the given qubit count.")
    parser.add_argument("--architect-steps", type=int, default=None,
                        help="Override architect steps per generation. Defaults to config.EXPERIMENT_PARAMS.")
    parser.add_argument("--saboteur-steps", type=int, default=None,
                        help="Override saboteur steps per generation. Defaults to config.EXPERIMENT_PARAMS.")
    parser.add_argument("--max-circuit-gates", type=int, default=config.MAX_CIRCUIT_TIMESTEPS)
    parser.add_argument("--fidelity-threshold", type=float, default=0.99)
    parser.add_argument("--lambda-penalty", type=float, default=0.5)
    # Gate set controlled via experiments/config.py (INCLUDE_ROTATIONS/ROTATION_TYPES)
    parser.add_argument("--task-mode", type=str, default=None)
    parser.add_argument("--n-seeds", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    start_time = time.time()

    # Resolve per-qubit defaults when user did not override via CLI
    qubit_params = config.get_params_for_qubits(args.n_qubits)
    n_generations = args.n_generations if args.n_generations is not None else qubit_params["N_GENERATIONS"]
    architect_steps = args.architect_steps if args.architect_steps is not None else qubit_params["ARCHITECT_STEPS_PER_GENERATION"]
    saboteur_steps = args.saboteur_steps if args.saboteur_steps is not None else qubit_params["SABOTEUR_STEPS_PER_GENERATION"]

    # Gate set is controlled centrally in experiments/config.py
    include_rotations = config.INCLUDE_ROTATIONS

    print("\n" + "=" * 70)
    print("MULTI-SEED ADVERSARIAL TRAINING")
    print("=" * 70)
    print(f"Results root : {args.results_dir}")
    print(f"Qubits       : {args.n_qubits}")
    print(f"Generations  : {n_generations}")
    print(f"Seeds        : {args.n_seeds}")
    print("=" * 70 + "\n")

    run_dirs = []

    for seed in range(args.n_seeds):
        print(f"\n***** SEED {seed} / {args.n_seeds - 1} *****")

        set_global_seed(seed)

        # Each seed gets its own subdir
        seed_dir = os.path.join(args.results_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        architect_agent, saboteur_agent, log_dir = train_adversarial(
            results_dir=seed_dir,
            n_qubits=args.n_qubits,
            n_generations=n_generations,
            architect_steps_per_generation=architect_steps,
            saboteur_steps_per_generation=saboteur_steps,
            max_circuit_gates=args.max_circuit_gates,
            fidelity_threshold=args.fidelity_threshold,
            lambda_penalty=args.lambda_penalty,
            include_rotations=include_rotations,
            task_mode=args.task_mode,
        )

        # Optional: save models per seed
        architect_agent.save(os.path.join(log_dir, "architect_adversarial.zip"))
        saboteur_agent.save(os.path.join(log_dir, "saboteur_adversarial.zip"))

        run_dirs.append(log_dir)

    total_minutes = (time.time() - start_time) / 60.0
    print(f"\n=== Multi-seed adversarial training complete ({total_minutes:.2f} min) ===")
    print("Per-seed log dirs:")
    for d in run_dirs:
        print("  -", d)


if __name__ == "__main__":
    main()
