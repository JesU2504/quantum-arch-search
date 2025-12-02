Experiments Overview
====================

Source of truth
---------------
- Hyperparameters live in `experiments/config.py`:
  - Gate set: `INCLUDE_ROTATIONS` and `ROTATION_TYPES` (action_gates are built via `get_action_gates`).
  - Per-qubit training budgets: `EXPERIMENT_PARAMS[...]` (architect/saboteur steps and PPO `n_steps`).
  - PPO base params: `AGENT_PARAMS`.
- All training scripts now read gates/steps from the config so runs stay aligned.

Baseline (Architect only)
-------------------------
- `train_architect.py` trains the Architect on the target (`config.TARGET_TYPE`, `config.TASK_MODE`), using the per-qubit budgets and gate set from the config. Produces the “vanilla” circuit and progress plots.
- `lambda_sweep.py` sweeps the Architect’s complexity penalty λ with the same gate set and per-qubit step budgets for consistency; reports fidelity/CNOT vs λ with error bars.

Saboteur only
-------------
- `train_saboteur_only.py` trains the Saboteur against a fixed baseline circuit (typically from `train_architect.py`). Uses the shared PPO params and saboteur budgets from `config.EXPERIMENT_PARAMS`.

Adversarial co-evolution
------------------------
- `train_adversarial.py` and `train_adversarial_multiseed.py` co-train Architect and Saboteur using the same gate set and per-qubit budgets from the config.
- `run_experiments.py` orchestrates the story: baseline Architect → saboteur-only (optional) → adversarial co-evolution (multi-seed) and plotting.
- Plotting helpers: `plot_coevolution.py` and `plot_coevolution_multiseed.py`.

Analysis / utilities
--------------------
- `cross_noise_robustness.py`, `parameter_recovery.py`, `compare_circuits.py`, `verify_saboteur.py`, and the VQE examples (`vqe_architecture_search_example.py`, `vqe_h4_benchmark.py`) consume saved circuits/results from the above steps.
- Plots and summaries land under `results/` with per-run timestamps; the top-level config keeps gate choices and budgets consistent across experiments for paper-ready comparisons.
