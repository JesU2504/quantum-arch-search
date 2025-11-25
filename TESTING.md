# Testing & Experiment Checklist

Purpose: a concise, copyable checklist to remember the common experiment and validation steps for this repo.

## Quick checklist (pilot)
1. Run a quick preset to validate the pipeline and artifact collection. This should be fast (smoke test).
   - Example (detached):

```bash
# from repo root
python3 run_experiments.py --preset quick --n-qubits 3 --base-dir results/run_quick_pilot --seed 0 &> results/run_quick_pilot/daemon.log &
```

2. Watch `results/run_quick_pilot/daemon.log` for milestones: "Starting pipeline", "Running baseline architect", "Saving champion", "compare finished", "Pipeline finished".
3. Confirm artifacts were produced in `results/run_quick_pilot` (look for `metadata.json`, `experiment.log`, `compare/aggregated_noise_resilience_data.csv`, saved circuits like `circuit_vanilla.json`).

## Full single-seed run (validation)
1. Run the `full` preset (longer training) to generate data for a single-seed comparison.

```bash
python3 run_experiments.py --preset full --n-qubits 3 --base-dir results/run_full_seed0 --seed 0 &> results/run_full_seed0/daemon.log &
```

2. Wait for the run to complete (or monitor `daemon.log`).
3. After completion, check `results/run_full_seed0/compare/` for CSVs and PNGs. Important files:
   - `aggregated_noise_resilience_data.csv` — per-noise average/STD
   - `aggregated_noise_resilience_worst.csv` — worst-case (min over gates) CSV
   - `noise_resilience_comparison.png` and `noise_resilience_comparison_worst.png`

## Multi-seed aggregation (recommended for statistics)
1. Run N independent `full` runs with different seeds (recommended N=5).
2. Use different `--base-dir` per run (e.g., `results/run_full_seed1`, `results/run_full_seed2`, ...).
3. After all runs complete, use `experiments/compare_circuits.py` or the provided aggregator (the orchestrator may already aggregate into `results/<run>/compare/aggregated_*.csv`) to compute mean, std, and worst-case across seeds.

## Transfer and per-gate heatmaps
1. Use `experiments/compare_circuits.py` to generate per-gate heatmaps and AUC metrics.
2. Save CSVs for reproducibility and statistical testing.

## Typical commands and flags
- Presets: `quick`, `full`, `long` (the `long` preset is for extended runs).
- Useful flags:
  - `--preset` : quick|full|long
  - `--n-qubits` : number of qubits (3, 4, ...)
  - `--base-dir` : output directory for this run
  - `--seed` : RNG seed

Example quick run (foreground):

```bash
python3 run_experiments.py --preset quick --n-qubits 3 --base-dir results/run_quick_pilot --seed 0
```

Example long run (detached):

```bash
nohup python3 run_experiments.py --preset long --n-qubits 4 --base-dir results/run_long_seed0 --seed 0 > results/run_long_seed0/daemon.log 2>&1 &
```

## Inspecting artifacts
- `metadata.json` — records the CLI flags and preset used.
- `daemon.log` or `experiment.log` — pipeline-level logs and milestone messages.
- `compare/` — CSVs and PNGs from comparisons.
- `results/.../run_*/` — saved models, circuits (`circuit_vanilla.json`, `circuit_robust.json`), and fidelity traces.

Quick checks after a run finishes:
```bash
# Check for success/final milestone
grep -E "Pipeline finished|compare finished|Saving champion" results/<run>/daemon.log || tail -n 200 results/<run>/daemon.log
# List compare folder
ls -l results/<run>/compare
# Show the aggregated CSV head
head -n 20 results/<run>/compare/aggregated_noise_resilience_data.csv
```

## Quality gates
- Unit tests: run `pytest -q` (there is at least `tests/test_champion_callback.py`).
- Smoke-run: run the `quick` preset and confirm artifacts appear.
- For changes to envs or training code: run a short quick run before large multi-seed runs.

## Troubleshooting tips
- Missing baseline for saboteur step: ensure the baseline `circuit_vanilla.json` exists in the expected path; the orchestrator passes `baseline_circuit_path` to saboteur training now.
- SB3 shape errors: make sure `--n-qubits` matches the saved artifacts and the saboteur/architect networks expect the same observation/action shapes.
- If `daemon.log` shows library warnings (e.g., `libtinfo`), they are usually non-blocking — check for later error lines.

## Minimal verification after run finishes
1. `pytest -q` → all tests pass.
2. Quick run completed with `Pipeline finished` in `daemon.log`.
3. `compare/aggregated_noise_resilience_data.csv` and `_worst.csv` exist and contain numeric rows.

## Next steps you may want to add to this file
- Exact statistical test commands for comparing paired seeds (e.g., paired t-test or bootstrap on worst-case fidelities).
- A short script to aggregate multiple `compare/aggregated_noise_resilience_worst.csv` files into one table with seeds.

---

If you'd like, I can also:
- add a small helper script that launches and monitors N detached runs, or
- add an aggregation script that combines per-seed worst-case CSVs into a single DataFrame and computes paired statistics.

