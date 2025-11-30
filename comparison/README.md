# Comparison Workspace: DRL vs EA Quantum Architecture Search

This scaffold provides tools for comparing the Deep Reinforcement Learning (DRL) approach from [arXiv:2407.20147](https://arxiv.org/abs/2407.20147) with the coevolutionary (EA) agents implemented in this repository.

## Overview

The comparison workspace does **not** reimplement existing agents. Instead, it provides:

- **Diagnostics**: Fidelity computation utilities for verifying unitary comparisons
- **Analysis**: Tools to aggregate metrics from experiment logs
- **Configs**: Example YAML configurations for running DRL and EA experiments
- **Paper Metadata**: Machine-readable JSON with extracted hyperparameters from papers
- **Notebooks**: Jupyter notebooks for visualization and analysis
- **Tests**: Pytest tests for validation
- **Schema**: JSON schema for standardized log format

## Directory Structure

```
comparison/
├── README.md                           # This file
├── requirements.txt                    # Additional dependencies
├── paper_metadata/
│   └── quantum_ml_arch_search_2407.20147.json  # Paper metadata (classification)
├── diagnostics/
│   ├── __init__.py
│   └── diagnose_fidelity.py           # Fidelity computation functions
├── analysis/
│   ├── __init__.py
│   ├── compute_metrics.py             # Log aggregation and metrics
│   └── compute_classif_metrics.py     # Classification-specific metrics
├── experiments/
│   ├── __init__.py
│   └── configs/
│       ├── drl.yaml                   # DRL agent configuration (general)
│       ├── ea.yaml                    # EA agent configuration (general)
│       ├── drl_classification.yaml    # DRL classification config
│       └── ea_classification.yaml     # EA classification config
├── notebooks/
│   ├── compare_paper_vs_ea.ipynb      # General analysis notebook
│   └── compare_classification.ipynb    # Classification comparison notebook
├── tests/
│   ├── __init__.py
│   ├── test_diagnose_fidelity.py      # Diagnostics tests
│   ├── test_log_schema.py             # Schema validation tests
│   ├── test_paper_metadata.py         # Paper metadata tests
│   └── test_classif_metrics.py        # Classification metrics tests
└── logs/
    └── schema.json                    # JSON schema for logs
```

## Installation

Install additional dependencies for the comparison workspace:

```bash
pip install -r comparison/requirements.txt
```

## Quick Start

### 1. Run Sanity Checks

Verify fidelity computations are working:

```bash
cd /path/to/quantum-arch-search
python -m comparison.diagnostics.diagnose_fidelity
```

Expected output: JSON with fidelity metrics for identity and Toffoli matrices.

### 2. Run Tests

```bash
pytest comparison/tests/ -v
```

### 3. Run EA Experiments

Use the main repository pipeline:

```bash
python run_experiments.py --preset quick --n-qubits 3 --seed 42
```

### 4. Analyze Logs

After running experiments, compute metrics:

```bash
python -m comparison.analysis.compute_metrics \
    --input "results/**/logs/*.jsonl" \
    --out comparison/logs/analysis_output
```

### 5. Use the Notebook

Open and run the comparison notebook:

```bash
jupyter notebook comparison/notebooks/compare_paper_vs_ea.ipynb
```

## Log Format

All experiment logs should follow the schema in `logs/schema.json`. Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `eval_id` | integer | Unique evaluation identifier |
| `timestamp` | string | ISO 8601 timestamp |
| `method` | string | "drl" or "ea" |
| `seed` | integer | Random seed |
| `best_fidelity` | number | Best fidelity (0-1) |
| `fidelity_metric` | string | Type of fidelity metric |

Optional fields include `circuit_depth`, `gate_count`, `wall_time_s`, `cum_eval_count`, `unitary_hash`, and `notes`.

Example log entry:
```json
{
  "eval_id": 100,
  "timestamp": "2024-01-15T10:30:00Z",
  "method": "ea",
  "seed": 42,
  "best_fidelity": 0.9876,
  "fidelity_metric": "trace_overlap",
  "circuit_depth": 12,
  "gate_count": 25,
  "wall_time_s": 3.14,
  "cum_eval_count": 1000
}
```

## Adding a New Agent

To add a new agent's results to the comparison:

1. **Create a config file** in `experiments/configs/` following the YAML template
2. **Ensure log output** follows `logs/schema.json` format
3. **Place logs** in `logs/` with naming convention: `{method}_run_{seed}.jsonl`
4. **Update the notebook** to include your method in the analysis

### Example: Adding a DRL Agent

1. Copy `experiments/configs/drl.yaml` and modify:
   ```yaml
   entrypoint:
     script: "path/to/your/drl_agent.py"
   ```

2. Run your agent and save logs:
   ```bash
   python your_drl_agent.py --output comparison/logs/drl/drl_run_0.jsonl
   ```

3. Compute metrics:
   ```bash
   python -m comparison.analysis.compute_metrics \
       --input "comparison/logs/**/*.jsonl" \
       --out comparison/logs/analysis_output
   ```

## Naming Conventions

- **Log files**: `{method}_run_{seed}.jsonl` (e.g., `ea_run_42.jsonl`)
- **Circuit files**: `circuit_{method}_{seed}.json`
- **Metrics files**: `metrics_{method}_{seed}.json`

## Diagnostics Functions

The `diagnostics.diagnose_fidelity` module provides:

- `toffoli_matrix()`: Generate 3-qubit Toffoli unitary
- `trace_overlap_fidelity(U_target, U)`: Compute trace overlap fidelity
- `compute_fidelities(U_target, U)`: Comprehensive fidelity metrics
- `run_basic_sanity_checks()`: Verify computations

Example usage:
```python
from comparison.diagnostics import compute_fidelities, toffoli_matrix
import numpy as np

U_target = toffoli_matrix()
U_candidate = U_target * np.exp(1j * 0.5)  # Phase-shifted

result = compute_fidelities(U_target, U_candidate)
print(f"Phase-corrected fidelity: {result['phase_corrected_trace_f']:.4f}")
```

## Analysis Functions

The `analysis.compute_metrics` module provides:

- `load_logs(paths)`: Load JSON/JSONL log files
- `validate_logs(logs)`: Validate against schema
- `aggregate_metrics(logs)`: Compute per-run and cross-run statistics
- `save_summary(metrics, out_path)`: Save JSON and CSV summaries

## Troubleshooting

### "No module named comparison"

Ensure you're running from the repository root:
```bash
cd /path/to/quantum-arch-search
python -m comparison.diagnostics.diagnose_fidelity
```

### "jsonschema not installed"

Install comparison dependencies:
```bash
pip install -r comparison/requirements.txt
```

### Tests failing

Ensure numpy is installed:
```bash
pip install -r requirements.txt
```

## Contributing

When adding new comparison tools:

1. Follow the existing code style
2. Add tests in `comparison/tests/`
3. Update this README if adding new functionality
4. Ensure logs conform to `logs/schema.json`

## References

- [arXiv:2407.20147](https://arxiv.org/abs/2407.20147) - DRL for Quantum Circuit Architecture Search
- Repository README for coevolutionary agent documentation

---

## Classification Experiments

This section describes how to run and compare DRL and EA methods on **classification tasks**, as described in arXiv:2407.20147.

### Paper Metadata

The file `paper_metadata/quantum_ml_arch_search_2407.20147.json` contains machine-readable metadata extracted from the paper, including:

- Paper title, authors, and arXiv information
- Tasks/datasets used (make_classification, make_moons)
- DRL algorithm (N-step DDQN) and hyperparameters
- State representation and action space encoding
- Gate set (RX, RY, RZ, CNOT) and max circuit depth
- Inner-loop parameter optimization settings
- Reward function design with complexity penalties
- Reported metrics and compute budget

Load the metadata in Python:
```python
import json
with open('comparison/paper_metadata/quantum_ml_arch_search_2407.20147.json') as f:
    metadata = json.load(f)
print(metadata['paper_title'])
print(metadata['drl_algorithm']['name'])  # N-step Double Deep Q-Network (DDQN)
```

### Classification Configs

Two YAML configs are provided for classification experiments:

1. **`experiments/configs/drl_classification.yaml`** — DRL settings from the paper:
   - N-step DDQN with ε-greedy exploration
   - Discount factor γ = 0.005^(1/L) where L=20
   - Reward function with accuracy-based rewards and gate penalties
   - Adaptive search with incrementing ytarget

2. **`experiments/configs/ea_classification.yaml`** — EA settings matched to DRL:
   - Same gate set (RX, RY, RZ, CNOT)
   - Same max circuit depth (L=20)
   - Same evaluation budget (~800 evaluations)
   - Same inner-loop optimization (15 epochs, Adam)

### Running Classification Experiments

1. **Set the entrypoint** in each config file to your agent implementation:
   ```yaml
   entrypoint:
     script: "path/to/your/classification_agent.py"
   ```

2. **Run experiments** (example commands):
   ```bash
   # DRL classification (implement or adapt your DRL agent)
   python your_drl_agent.py --config comparison/experiments/configs/drl_classification.yaml
   
   # EA classification (adapt existing VQE environment for classification)
   python your_ea_agent.py --config comparison/experiments/configs/ea_classification.yaml
   ```

3. **Place logs** in the appropriate directories:
   - DRL: `comparison/logs/drl_classification/`
   - EA: `comparison/logs/ea_classification/`

### Computing Classification Metrics

Use the classification-specific metrics module:

```bash
python -m comparison.analysis.compute_classif_metrics \
    --input "comparison/logs/**/*.jsonl" \
    --out comparison/logs/classification_analysis
```

This computes:
- Final and best classification accuracy
- Evaluations to reach accuracy thresholds (70%, 80%, 90%)
- Gate count and circuit depth statistics
- Aggregated statistics by method

### Classification Log Format

Classification logs should include these fields:

| Field | Type | Description |
|-------|------|-------------|
| `eval_id` | integer | Evaluation identifier |
| `method` | string | "drl" or "ea" |
| `seed` | integer | Random seed |
| `train_accuracy` | number | Training accuracy (0-1) |
| `test_accuracy` | number | Test/validation accuracy (0-1) |
| `gate_count` | integer | Number of gates in circuit |
| `circuit_depth` | integer | Circuit depth |

Example log entry:
```json
{
  "eval_id": 100,
  "method": "drl",
  "seed": 0,
  "train_accuracy": 0.85,
  "test_accuracy": 0.82,
  "gate_count": 15,
  "circuit_depth": 12,
  "episode_reward": 1.2
}
```

### Classification Notebook

Open the classification comparison notebook:

```bash
jupyter notebook comparison/notebooks/compare_classification.ipynb
```

The notebook provides:
- Loading paper metadata and run summaries
- Accuracy vs evaluations learning curves
- ECDF of final accuracies across seeds
- Pareto plot: accuracy vs circuit depth
- Box/bar plots comparing methods
- Fair comparison checklist

### Fair Comparison Checklist

Before comparing DRL and EA results, verify:

- [ ] Same gate set (RX, RY, RZ, CNOT)
- [ ] Same max circuit depth (L=20)
- [ ] Same evaluation budget (~800 evaluations)
- [ ] Same inner-loop optimization (15 epochs max)
- [ ] Same dataset and train/test split
- [ ] Same data encoding (arctan embedding)
- [ ] Same number of seeds (≥5)
- [ ] Same gate penalty coefficient (0.01)

### Key Insights from the Paper

1. **Reward Design**: The paper uses a three-part reward function:
   - Success: Reward for reaching target accuracy with fewer gates
   - Failure: Penalty for not reaching minimum accuracy at max gates
   - Dynamic: Incremental accuracy improvement minus gate penalty

2. **Adaptive Search**: Target accuracy `ytarget` increases dynamically when the agent consistently succeeds, preventing premature convergence.

3. **Parameter Optimization**: The DRL agent selects gate types and positions; rotation angles are optimized classically after each gate addition (15 epochs max).

4. **Omitted Hyperparameters**: Several hyperparameters are not specified in the paper (learning rate, batch size, MLP architecture). The metadata file documents these omissions and provides reasonable assumptions.
