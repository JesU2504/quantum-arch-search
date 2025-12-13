# Paper Runbook: Commands and What They Do

Use this as a checklist to generate all artifacts for the paper. Run from the repo root inside your virtualenv.

## 0) Environment setup (one-time)
- `python3 -m venv .venv` — Create an isolated Python environment.
- `source .venv/bin/activate` — Activate the virtualenv.
- `pip install -r requirements.txt` — Install pinned dependencies.
- `pip freeze > results/env.txt` — Snapshot exact package versions for reproducibility.

## 1) Core adversarial vs. baselines (state/unitary prep)
- **State prep (GHZ target by default; set TARGET_TYPE=\"ghz\" in experiments/config.py)**  
  1. "python run_experiments.py \
  --n-qubits 5 \
  --n-seeds 5 \
  --run-quantumnas \
  --robustness-noise-families amplitude_damping coherent_overrotation depolarizing readout \
  --robustness-budgets 1,3 \
  --attack-samples 100 \
  --compare-samples 64 \
  --run-hw-eval \
  --hw-backends fake_yorktown fake_oslo fake_lagos fake_vigo \
  --hw-shots 4096 \
  --hw-opt-level 1 \
  --skip-lambda-sweep \
  --mitigation-mode rc_zne --rc-zne-scales 0.75 1.0 1.5 2.0 3.0 --rc-zne-fit linear --rc-zne-reps 3 \
  --per-step-penalty -0.05 \
  --stop-success-bonus 0.4 \
  --stop-failure-penalty -0.3 \
  --saboteur-attack-candidate-fraction 0.6"



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
2. " python run_vqe.py \
  --molecule H2 \
  --n-seeds 2 \
  --rl-total-timesteps 50000 \
  --run-adv-architect \
  --adv-n-generations 10 \
  --adv-architect-steps-per-gen 4000 \
  --adv-saboteur-steps-per-gen 2000 \
  --adv-saboteur-budget 3 \
  --adv-max-gates 20 \
  --adv-saboteur-noise-family depolarizing \
  --skip-robustness \
  --skip-cross-noise
"

  > Re-run plots/analysis without retraining by pointing at the prior results directory:  
  > `python run_vqe.py --base-dir results/vqe_run_<timestamp> --analysis-only --reuse-existing`


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
