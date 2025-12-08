# Paper Runbook: Commands and What They Do

Use this as a checklist to generate all artifacts for the paper. Run from the repo root inside your virtualenv.

## 0) Environment setup (one-time)
- `python3 -m venv .venv` — Create an isolated Python environment.
- `source .venv/bin/activate` — Activate the virtualenv.
- `pip install -r requirements.txt` — Install pinned dependencies.
- `pip freeze > results/env.txt` — Snapshot exact package versions for reproducibility.

## 1) Core adversarial vs. baselines (state/unitary prep)
- **State prep (GHZ target by default; set TARGET_TYPE=\"ghz\" in experiments/config.py)**  
  1. `python run_experiments.py --n-qubits 3 --n-seeds 5 --saboteur-noise-family depolarizing --robustness-noise-families depolarizing amplitude_damping coherent_overrotation --robustness-budgets 1,3 --attack-samples 2000`  
    Multi-seed adversarial PPO on 3 qubits (state prep), logging robustness sweeps across multiple noise families and budgets. Produces circuits, plots, and `analysis/robustness_sweep.{csv,json}`.
  2. `python run_experiments.py --n-qubits 3 --n-seeds 5 --robustness-noise-families depolarizing amplitude_damping coherent_overrotation --robustness-budgets 1,3`  
    Non-adversarial PPO baseline with matching steps/seeds to compare clean and attacked fidelities.
  3. `python run_experiments.py --n-qubits 3 --skip-saboteur --skip-compare --skip-cross-noise --skip-parameter-recovery --skip-hw-eval`  
    Static-penalty λ sweep only (ExpPlan Part 1) to benchmark against tuned penalties.
  4. `python run_experiments.py --n-qubits 3 --run-quantumnas --robustness-noise-families depolarizing amplitude_damping coherent_overrotation --robustness-budgets 1,3`  
    QuantumNAS scaffold baseline (or imports if provided) for head-to-head robustness comparisons.

- **Unitary prep (Toffoli family by default; set TARGET_TYPE=\"toffoli\" in experiments/config.py and pass `--task-mode unitary_preparation`)**  
  - `python run_experiments.py --n-qubits 3 --task-mode unitary_preparation --n-seeds 5 --saboteur-noise-family depolarizing --robustness-noise-families depolarizing amplitude_damping coherent_overrotation --robustness-budgets 1,3 --attack-samples 2000`  
    Multi-seed adversarial PPO for unitary compilation (Toffoli target), with robustness sweeps.
  - `python run_experiments.py --n-qubits 3 --task-mode unitary_preparation --n-seeds 5 --skip-adversarial --robustness-noise-families depolarizing amplitude_damping coherent_overrotation --robustness-budgets 1,3`  
    Non-adversarial PPO baseline for unitary prep.
  - `python run_experiments.py --n-qubits 3 --task-mode unitary_preparation --skip-adversarial --skip-saboteur --skip-compare --skip-cross-noise --skip-parameter-recovery --skip-hw-eval`  
    Static-penalty λ sweep for unitary prep.
  - `python run_experiments.py --n-qubits 3 --task-mode unitary_preparation --run-quantumnas --robustness-noise-families depolarizing amplitude_damping coherent_overrotation --robustness-budgets 1,3`  
    QuantumNAS scaffold baseline for unitary prep (if applicable).
d
## 2) VQE evidence
- `python experiments/vqe/train_architect_vqe_rl.py --molecule H2 --total-timesteps 10000 --max-gates 12 --out-dir results/vqe_architect_vqe_h2_seed0`  
  RL architecture search for VQE on H2; logs energies, gate counts, and best circuits (repeat with different out-dirs for multi-seed).
- `python experiments/adversarial/train_adversarial_vqe.py --molecule H2 --steps 200 --noise-levels 0.0 0.02 0.05 --out-dir results/adversarial_vqe_h2_seed0`  
  Adversarial VQE baseline (worst-case over depolarizing rates) using TorchQuantum HEA ansatz. Repeat with different `--seed`/`--out-dir` for multi-seed.
- `python experiments/vqe/vqe_h4_benchmark.py --max-iterations 200 --n-seeds 3`  
  H4 benchmark comparing ansätze (UCCSD/HEA/adversarial); produces energies and CNOT counts.

## 3) Robustness gatecheck (per run directory)
- `python experiments/analysis/robustness_gatecheck.py --sweep results/run_<ts>/analysis/robustness_sweep.csv --min-fidelity 0.95 --min-gap 0.02`  
  Enforces pass/fail on attacked fidelity and improvement over baseline. Fails (nonzero exit) if thresholds aren’t met.

## 4) Plots and summaries
- `python experiments/analysis/plot_coevolution_multiseed.py --root-dir results/run_<ts>/adversarial --out results/run_<ts>/adversarial/coevolution_multiseed.png`  
  Renders multi-seed co-evolution plot.
- `python experiments/analysis/plot_coevolution.py --run-dir results/run_<ts>/adversarial/seed_0/adversarial_training_0 --out results/run_<ts>/adversarial/seed_0/coevolution_seed_0.png`  
  Renders a single-seed co-evolution plot (adjust seed/run dir as needed).
- Robustness/Pareto data live in `results/run_<ts>/analysis/` (e.g., `robustness_sweep.csv`, `fidelity_raw_data.csv`); feed these into your plotting notebook for figures.

## 5) Verification and smoke tests
- `pytest tests/test_toffoli_verification.py -q` — Full-basis Toffoli/gate verification regression.
- `pytest tests/test_saboteur_efficacy.py -q` — Confirms saboteur actually drops fidelity and respects budget.
- `pytest tests/test_architect_circuit_saving.py tests/test_vqe_env_integration.py -q` — PPO/VQE smoke tests and circuit saving sanity.

## 6) Artifact collation (for paper bundle)
- Collect from `results/run_<ts>/`:  
  - `analysis/robustness_sweep.{csv,json}`, `analysis/fidelity_raw_data.csv`  
  - `adversarial/*/coevolution_*.png`, `compare/*/robustness_comparison.png` (if generated)  
  - Champion circuits: `compare/run_*/circuit_vanilla.json`, `circuit_robust.json`, `circuit_quantumnas.json` (if present)  
  - `metadata.json` and `env.txt` for reproducibility.

## 7) Optional longer/full run
- `python run_experiments.py --n-qubits 3 --n-seeds 5 --saboteur-noise-family depolarizing --robustness-noise-families depolarizing amplitude_damping coherent_overrotation --robustness-budgets 1,3,5 --attack-samples 5000`  
  Longer training for publication-quality curves (slower). Optionally bump `--baseline-steps`, `--saboteur-steps`, or `--max-circuit-gates` if you need deeper/longer runs.
