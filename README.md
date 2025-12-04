# Quantum Architecture Search via Adversarial Co‑Evolution

Co‑evolutionary quantum circuit search with two agents: an Architect (constructs circuits) and a Saboteur (injects noise). The adversarial game acts as a parameter‑free, dynamic regularizer that discourages bloat and promotes robustness.

This repository contains a turnkey pipeline to train baseline and adversarial agents, generate co‑evolution plots, and compare robustness of “vanilla” vs “robust” circuits.


## Research goal (for the talk)

We demonstrate that Adversarial Co‑Evolution is a parameter‑free regularizer that outperforms static penalty methods on stability, robustness, and Pareto efficiency. See `ExpPlan.md` for the detailed experimental plan and rationale.


## Default Target: n-Controlled Toffoli Gates

**New default**: All experiments now use n-controlled Toffoli (multi-controlled NOT) gates as the compilation target:
- **2 qubits**: CNOT gate
- **3 qubits**: Toffoli (CCNOT) gate
- **4 qubits**: CCCNOT gate  
- **n qubits**: (n-1)-controlled NOT gate

The Toffoli gate target provides a more challenging benchmark than GHZ state preparation, as it requires precise multi-qubit controlled operations. GHZ state preparation remains available as a legacy option via `get_ghz_state()`.


## Install

Tested with Python 3.12 on Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (pinned): Cirq, Gymnasium, Stable‑Baselines3, NumPy, Matplotlib.


## Quick demo (3-qubit Toffoli, full pipeline)

Runs baseline architect training, saboteur‑only training, adversarial co‑evolution for a couple of generations, plotting, and a robustness comparison.

```bash
python run_experiments.py --preset quick
```

Outputs (example): `results/run_YYYYMMDD-HHMMSS/`
- `baseline/`
	- `circuit_vanilla.json` — champion vanilla (noiseless) circuit
	- `architect_fidelities.txt`, `architect_steps.txt`, `architect_training_progress.png`
- `saboteur/`
	- `saboteur_trained_on_architect_model.zip`
	- `saboteur_trained_on_architect_fidelities.txt`, `saboteur_trained_on_architect_steps.txt`, `...training_progress.png`
- `adversarial/adversarial_training_*/`
	- `circuit_robust.json` — robust circuit after co‑evolution
	- `coevolution_corrected.png` — corrected co‑evolution plot
- `compare/run_0/`
	- `circuit_vanilla.json`, `circuit_robust.json`
- `compare/`
	- `robust_eval.json`, `attacked_fidelity_samples.csv`, `robustness_comparison.png`

### Experimental baseline: QuantumNAS (scaffold)
- Optional flag: `--run-quantumnas` in `run_experiments.py` runs a QuantumNAS baseline scaffold and saves `quantumnas/circuit_quantumnas.json` for robustness analysis.
- Status: if no external circuit is provided, `--run-quantumnas` now runs a lightweight TorchQuantum-based baseline (GHZ state prep or Toffoli unitary prep) and writes `quantumnas/circuit_quantumnas.json`. You can override epochs/depth/lr via `--quantumnas-simple-*` flags.
- Compare/cross-noise analyses will include the QuantumNAS circuit automatically when present.
- Parameter recovery also includes QuantumNAS automatically if `quantumnas/circuit_quantumnas.json` exists in the run.
- Official QuantumNAS paper harness: `--run-quantumnas-paper` launches `experiments/train_quantumnas_paper.py` to call the upstream TorchQuantum/QuantumNAS training loop and export `quantumnas/circuit_quantumnas.json`. Benchmarks supported out of the box: classification on `mnist4`, `fashionmnist4`, `cifar10_4`, `gtsrb4`, or `vowel` (choose with `--quantumnas-paper-dataset`) and VQE on `H2`, `LiH`, `BeH2`, or `HeH+` (choose with `--quantumnas-paper-molecule`).
- If you train a circuit externally with TorchQuantum/QuantumNAS and export OpenQASM, convert it for analysis with:
  ```bash
  python experiments/import_quantumnas_qasm.py --qasm path/to/circuit.qasm --out results/run_x/quantumnas/circuit_quantumnas.json
  ```
- To evaluate a QASM circuit directly (clean + attacked fidelity) and save metrics/circuit JSON:
  ```bash
  python experiments/eval_qasm_circuit.py --qasm path/to/circuit.qasm --target-type ghz --task-mode state_preparation
  ```
- If you have a TorchQuantum `op_history` (e.g., from QuantumNAS/VQE), convert it to QASM and Cirq JSON:
  ```bash
  python experiments/export_tq_op_history.py --op-history path/to/op_history.pt
  ```
- You can also pass `--quantumnas-qasm / --quantumnas-op-history` to `run_experiments.py` to import external QuantumNAS circuits into the pipeline (saved as `quantumnas/circuit_quantumnas.json`).


## Reproducing key artifacts

- Co‑evolution plot: produced automatically by the quick pipeline and saved under `adversarial/adversarial_training_*/coevolution_corrected.png`.
- Robustness comparison plot: produced under `compare/robustness_comparison.png` comparing vanilla vs robust circuits under multi‑gate depolarizing attacks.

To scale up, use:

```bash
# Longer runs
python run_experiments.py --preset full --n-qubits 3

# Custom steps (override parts of a preset)
python run_experiments.py --preset quick --baseline-steps 5000 --saboteur-steps 5000 --n-qubits 3
```


## How it works (files you’ll touch)

- Entry point: `run_experiments.py` — orchestrates baseline → saboteur‑only → adversarial → compare.
- Experiment scripts (in `experiments/`):
	- `train_architect_ghz.py`: baseline Architect for GHZ preparation
	- `train_saboteur_only.py`: Saboteur learns to attack the baseline circuit
	- `train_adversarial.py`: co‑evolution between Architect and Saboteur
	- `plot_coevolution.py`: renders corrected co‑evolution plots
	- `compare_circuits.py`: aggregates vanilla vs robust robustness metrics/plots
	- `vqe_architecture_search_example.py`: VQE architecture search demo
	- `vqe_h4_benchmark.py`: H4 benchmark comparing ansatzes (Part 4)
- Environments (in `src/qas_gym/envs/`):
	- `architect_env.py`: reward‑shaped ArchitectEnv and AdversarialArchitectEnv
	- `saboteur_env.py`: SaboteurMultiGateEnv with attack budget and vectorized noise actions
	- `qas_env.py`: shared circuit/env utilities
	- `vqe_architect_env.py`: VQE environment for molecular ground state optimization


## Mapping to the Experimental Plan

The repository ships the core pipeline and figures for Parts 2 and 7, and a strong baseline for Part 3. For completeness, the full plan is in `ExpPlan.md`.

- Part 1 (λ‑sweep brittleness, static penalty): not included as code here. Our method avoids λ entirely; reproducing tuned static‑penalty sweeps would require a simple reward variant (R = Fidelity − λ·Cost) and a small sweep harness.
- Part 2 (Robustness to shift): included — the Saboteur varies attacks during training; robustness comparison is generated by `compare_circuits.py` (multi‑gate depolarizing). Extending to asymmetric/coherent noise sweeps is straightforward in that script.
- Part 3 (Pareto frontier): partially included — we output fidelity and circuits. To plot CNOT‑count vs fidelity, compute CNOT count from `circuit_*.json` and scatter. If desired, we can add explicit CNOT logging in env info.
- Part 4 (VQE on H4): **fully implemented** — see VQE Architecture Search section below.
- Part 5 (Overhead analysis): not included as an automated benchmark; you can wrap wall‑clock timing around each phase in `run_experiments.py`.
- Part 6 (QEC resource plot): not included in this repository. See `ExpPlan.md` for the argument framing; adding a simple bar‑chart script to visualize physical‑qubit overhead is straightforward.
- Part 7 (Verification): included conceptually — saboteur‑only run demonstrates fidelity degradation and learning. A dedicated `verify_saboteur` script can be added or adapted if you want a one‑shot check on a perfect GHZ circuit.

## Fast 3‑qubit sanity run (aligned with ExpPlan Stage 7.4)

- Defaults in `experiments/config.py` for 3 qubits are tuned for a quick pass: 12 generations, 8000 architect steps/gen (96k total), saboteur 2048 steps/gen.
- Empirically reaches ~1.0 fidelity around 75k architect steps; use this as a smoke test before longer jobs.
- You can replot without retraining: `python experiments/train_architect.py --n-qubits 3 --replot-from results/<run>/baseline`


## VQE Architecture Search (Part 4)

This repository includes full support for VQE-based quantum chemistry experiments using the `VQEArchitectEnv` environment.

### VQEArchitectEnv Overview

The `VQEArchitectEnv` is a Gymnasium environment where an RL agent designs quantum circuits to minimize molecular Hamiltonian energy:

- **Molecules supported**: H2 (2 qubits) and H4 (4 qubits at 1.5 Å stretched geometry)
- **Action space**: Discrete actions for parameterized rotation gates (Rx, Ry, Rz on each qubit), CNOT gates (all ordered pairs), and a DONE action to terminate
- **Classical optimization**: At episode end, rotation angles are optimized using scipy's L-BFGS-B with multiple restarts
- **Reward**: Based on optimized energy relative to FCI ground state, with optional CNOT penalty

### Quick VQE Demo

```bash
# Quick demo on H2 (2 qubits)
python experiments/vqe_architecture_search_example.py --molecule H2 --episodes 10

# H4 benchmark (4 qubits, Part 4 of ExpPlan.md)
python experiments/vqe_architecture_search_example.py --molecule H4 --episodes 50

# Using greedy agent instead of random
python experiments/vqe_architecture_search_example.py --molecule H2 --agent greedy --episodes 5
```

### VQE H4 Benchmark

For the full H4 benchmark comparing UCCSD, Hardware Efficient, and Adversarial ansatzes:

```bash
python experiments/vqe_h4_benchmark.py --max-iterations 200 --n-seeds 3
```

Win condition (from ExpPlan.md): Achieve chemical accuracy (1.6 mHa) with fewer CNOTs than UCCSD.

### Using VQEArchitectEnv Programmatically

```python
from src.qas_gym.envs import VQEArchitectEnv

# Create environment for H4 at 1.5 Å bond distance
env = VQEArchitectEnv(
    molecule="H4",
    bond_distance=1.5,
    max_timesteps=15,
    log_dir="results/my_vqe_run"  # Optional: enable logging
)

# Reset and run episode
obs, info = env.reset(seed=42)
print(f"Initial (HF) energy: {info['initial_energy']:.4f} Ha")

# Agent adds gates (example: Ry on each qubit + CNOTs)
for action in [4, 5, 6, 7, 12, 13]:  # Ry gates then CNOTs
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

# Terminate episode
obs, reward, terminated, truncated, info = env.step(24)  # DONE action for H4

print(f"Optimized energy: {info['optimized_energy']:.4f} Ha")
print(f"Energy error: {info['energy_error']*1000:.2f} mHa")
print(f"Chemical accuracy achieved: {info['chemical_accuracy_achieved']}")

# Get best circuit found
circuit, energy = env.get_best_circuit()
```

### Action Space Encoding

For a molecule with `n` qubits:
- Actions 0 to n-1: Rx gates on qubits 0, 1, ..., n-1
- Actions n to 2n-1: Ry gates on each qubit
- Actions 2n to 3n-1: Rz gates on each qubit
- Actions 3n to 3n + n(n-1) - 1: CNOT gates (all ordered pairs)
- Last action: DONE (terminate episode)

### Logging and Reproducibility

When `log_dir` is specified, the environment logs:
- Circuit architecture (gate sequence)
- Initial and optimized rotation angles
- CNOT and total gate counts
- Optimization details (iterations, success)
- Initial and final energies
- Whether chemical accuracy was achieved

All logs are saved in JSON format for full reproducibility.


## Tips and troubleshooting

- Seeds: pass `--seed <int>` to `run_experiments.py` for reproducibility.
- QuBits: use `--n-qubits 3` (default). Higher qubit counts raise simulation cost.
- Results hygiene: archive or prune older runs in `results/` to keep the bundle small.
- Gym vs Gymnasium: this project uses Gymnasium; if you see a warning about Gym, it’s safe to ignore.


## Citation

If you use this code, please cite the corresponding talk/paper. For questions or issues, open an issue or contact the authors.



## Changelog

- **Toffoli Gate Verification Harness**: Added full basis-sweep fidelity evaluation for Toffoli (n-controlled NOT) gate synthesis verification. This ensures candidate circuits are evaluated on all 2^n computational basis inputs, not just |000...0>. Key functions: `full_basis_fidelity()`, `full_basis_fidelity_toffoli()`, `toffoli_truth_table()`. Added comprehensive regression tests.
- **Parameterized Rotation Gates Support**: Enhanced `QuantumArchSearchEnv`, `ArchitectEnv`, and `AdversarialArchitectEnv` to support parameterized rotation gates (Rx, Ry, Rz) in addition to Clifford, T, and CNOT gates. Set `include_rotations=True` when creating environments to enable this feature. This provides greater circuit expressiveness and parity with VQE approaches. Added utility functions for rotation gate creation, serialization, and metrics. Updated Saboteur environment to recognize rotation gates. Backward compatible - rotation gates are disabled by default.
- **VQE Architecture Search (Part 4)**: Fully implemented `VQEArchitectEnv` for molecular ground state optimization. Includes parameterized rotation gates (Rx, Ry, Rz), CNOT gates, classical optimization of rotation angles at episode end, comprehensive logging, and support for H2 and H4 molecules. Added example script and integration tests.
- **Environment Consolidation**: All environments and agents (ArchitectEnv, AdversarialArchitectEnv, Saboteur, VQEArchitectEnv) are now unified under `src/qas_gym/envs/`; duplicate definitions in `src/envs/` have been removed. Import from `src.qas_gym.envs` for all environment classes.


## Toffoli Gate Verification Harness

This repository includes a full verification harness for evaluating Toffoli (and n-controlled NOT) gate synthesis experiments using full basis-sweep fidelity.

### Overview

The verification harness computes average fidelity over all 2^n computational basis inputs by:
1. Enumerating all computational basis input states
2. For each input: preparing the state, simulating the candidate circuit (with optional noise)
3. Computing fidelity between output density matrix and expected output state (based on gate's truth table)
4. Averaging fidelities over all inputs

This ensures circuits are validated against the full gate behavior, not just a single input like |000...0>.
## Parameterized Rotation Gates

The `QuantumArchSearchEnv` and derived environments now support optional parameterized rotation gates (Rx, Ry, Rz) in addition to the standard Clifford and T gate set.

### Benefits

- **Greater Expressiveness**: Rotation gates allow for continuous parameterization, enabling circuits that can exactly prepare arbitrary quantum states.
- **VQE Parity**: Aligns the adversarial architecture search with VQE-style variational circuits, allowing comparison of discovered circuits with standard ansatzes.
- **Flexibility**: Rotation gates can be enabled/disabled based on the specific task requirements.

### Usage

```python
from src.utils.metrics import (
    full_basis_fidelity_toffoli,
    full_basis_fidelity,
    toffoli_truth_table,
)
import cirq

# Evaluate a circuit for Toffoli (CCNOT, 2 controls)
qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit(cirq.TOFFOLI(*qubits))
fidelity = full_basis_fidelity_toffoli(circuit, qubits, n_controls=2)
print(f"Toffoli fidelity: {fidelity:.4f}")  # Should be 1.0 for perfect gate

# Evaluate with noise
noise_model = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.01))
fidelity_noisy = full_basis_fidelity_toffoli(circuit, qubits, n_controls=2, noise_model=noise_model)
print(f"Noisy fidelity: {fidelity_noisy:.4f}")

# For 3-controlled NOT (CCCNOT)
qubits_4 = cirq.LineQubit.range(4)
cccnot_gate = cirq.X.controlled(num_controls=3)
circuit_4 = cirq.Circuit(cccnot_gate(*qubits_4))
fidelity_cccnot = full_basis_fidelity_toffoli(circuit_4, qubits_4, n_controls=3)
print(f"CCCNOT fidelity: {fidelity_cccnot:.4f}")

# For custom gates, use full_basis_fidelity with a custom truth table
single_qubit = cirq.LineQubit.range(1)
not_circuit = cirq.Circuit(cirq.X(single_qubit[0]))  # NOT gate
not_truth_fn = lambda x: x ^ 1  # NOT gate truth table: |0> -> |1>, |1> -> |0>
not_fidelity = full_basis_fidelity(not_circuit, single_qubit, not_truth_fn)
print(f"NOT gate fidelity: {not_fidelity:.4f}")  # Should be 1.0
```

### Available Functions

- `full_basis_fidelity(circuit, qubits, truth_table_fn, noise_model=None)`: General function for any gate with a given truth table
- `full_basis_fidelity_toffoli(circuit, qubits, n_controls, noise_model=None)`: Convenience wrapper for Toffoli family gates
- `toffoli_truth_table(n_controls)`: Generates truth table for n-controlled NOT gates
- `computational_basis_state(index, n_qubits)`: Creates computational basis state vectors

### Key Properties

- **Perfect gates** yield fidelity = 1.0
- **Identity/wrong circuits** yield fidelity < 1.0 (typically 0.75 for Toffoli because 6/8 states match)
- **Noisy circuits** yield reduced fidelity depending on noise level
- **Full basis sweep** ensures circuits that only work on |000...0> are correctly rejected
from src.qas_gym.envs import QuantumArchSearchEnv, ArchitectEnv
from src.qas_gym.utils import get_ghz_state
import numpy as np


# Create environment with rotation gates enabled
target = get_ghz_state(3)
# To use multiple allowed rotation angles for Rx, Ry, Rz gates, pass a list of angles:
allowed_angles = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi]
env = QuantumArchSearchEnv(
	target=target,
	fidelity_threshold=0.99,
	reward_penalty=0.01,
	max_timesteps=20,
	include_rotations=True,  # Enable Rx, Ry, Rz gates
	default_rotation_angle=allowed_angles  # Pass a list for multiple angles
)

# Standard usage
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# Get rotation gate information
circuit_info = env.get_circuit_info()
print(f"Rotation counts: {circuit_info['rotation_counts']}")
print(f"Rotation angles: {circuit_info['rotation_angles']}")

# Modify rotation angle for a specific gate
env.set_rotation_angle(gate_index=0, angle=np.pi/2)
```

### Action Space with Rotations


When `include_rotations=True`, the action space is expanded to include:
- All original Clifford/T gates (X, Y, Z, H, T, S on each qubit)
- Rx, Ry, Rz rotation gates on each qubit, for each angle in the list provided to `default_rotation_angle` (e.g., [0, 0.25π, 0.5π, 0.75π, π])
- CNOT gates for all qubit pairs

The action space remains discrete; each rotation gate action corresponds to a specific angle from the allowed list. For full continuous angle optimization at episode end, consider using `VQEArchitectEnv`.

### Metrics and Logging

Circuit metrics now include rotation gate information:

```python
from src.utils.metrics import evaluate_circuit, count_rotation_gates, get_rotation_angles

# Evaluate circuit including rotation metrics
metrics = evaluate_circuit(circuit, target_state, qubits)
# Returns: fidelity, cnot_count, total_gates, depth, rotation_counts, rotation_angles

# Count rotation gates
counts = count_rotation_gates(circuit)
# Returns: {'rx_count': N, 'ry_count': N, 'rz_count': N, 'total_rotations': N}

# Get detailed angle information
angles = get_rotation_angles(circuit)
# Returns: [{'gate_type': 'Rx', 'qubit': 'q(0)', 'angle': 0.785, 'index': 0}, ...]
```

### Caveats

- **Increased Complexity**: Adding rotation gates increases the action space, which may require longer training times.
- **Reproducibility**: Results with rotation gates may differ from Clifford-only experiments. Keep `include_rotations=False` (default) to maintain backward compatibility.
- **Continuous vs Discrete**: Currently uses a fixed default angle for rotation gates. For continuous angle optimization, use `VQEArchitectEnv` instead.


## Statistical Reporting Protocol

This repository follows best-practice statistical reporting for quantum architecture search experiments. All major experiments support multi-seed runs for statistical validity.

### Configuration

- **Number of seeds**: Configurable via `--n-seeds <int>` (default: 5, recommended: 10 for publication).
- **Seed control**: Use `--seed <int>` to set the base random seed for reproducibility.

### Running Multi-Seed Experiments

```bash
# Run full pipeline with 10 seeds per setting
python run_experiments.py --preset full --n-seeds 10 --seed 42

# Run specific experiment with custom seeds
python experiments/lambda_sweep_ghz.py --n-seeds 10 --n-qubits 4

# Parameter recovery with configurable repetitions
python experiments/parameter_recovery.py --n-repetitions 10 --baseline-circuit path/to/circuit.json
```

### Output Structure

Each experiment produces:
- **Per-seed results**: Individual JSON files for each seed (e.g., `lambda_0.001/seed_0_results.json`)
- **Aggregated results**: Combined statistics with mean +/- std
- **Summary files**:
  - `experiment_summary.json`: Machine-readable summary with all parameters
  - `experiment_summary.txt`: Human-readable summary
- **Plots with error bars**: All plots show mean +/- std with sample size annotation (n=...)

### Statistical Metrics

- **Aggregation method**: Mean +/- std (sample standard deviation with ddof=1)
- **Confidence intervals**: 95% CI using t-distribution
- **Success rate**: Wilson score interval for binomial proportions
- **Plots**: Error bars (mean +/- std), faint individual curves overlay, sample size annotation

### Checklist for Multi-Seed Experiments

1. Set `--n-seeds` to at least 5 (ideally 10)
2. Set `--seed` for reproducibility
3. Verify per-seed results are saved in experiment subdirectories
4. Check summary files contain all seeds used and hyperparameters
5. Confirm plots show error bars and sample size annotations
