# Comparison Workspace: DRL vs EA Quantum Architecture Search

This scaffold provides tools for comparing the Deep Reinforcement Learning (DRL) approach from [arXiv:2407.20147](https://arxiv.org/abs/2407.20147) with the coevolutionary (EA) agents implemented in this repository.

## Overview

The comparison workspace does **not** reimplement existing agents. Instead, it provides:

- **Paper Metadata**: Extracted hyperparameters from arXiv:2407.20147 for reproducibility
- **Diagnostics**: Fidelity computation utilities for verifying unitary comparisons
- **Analysis**: Tools to aggregate metrics from experiment logs (fidelity and classification)
- **Configs**: Pre-filled YAML configurations for running DRL and EA experiments
- **Notebooks**: Jupyter notebooks for visualization and analysis
- **Tests**: Pytest tests for validation
- **Schema**: JSON schema for standardized log format

## Directory Structure

```
comparison/
├── README.md                           # This file
├── requirements.txt                    # Additional dependencies
├── paper_metadata/
│   └── quantum_ml_arch_search_2407.20147.json  # Extracted paper hyperparameters
├── diagnostics/
│   ├── __init__.py
│   └── diagnose_fidelity.py           # Fidelity computation functions
├── analysis/
│   ├── __init__.py
│   ├── compute_metrics.py             # Fidelity-based log aggregation
│   └── compute_classif_metrics.py     # Classification-specific metrics
├── experiments/
│   ├── __init__.py
│   └── configs/
│       ├── drl.yaml                   # DRL fidelity config
│       ├── ea.yaml                    # EA fidelity config
│       ├── drl_classification.yaml    # DRL classification config (paper hyperparams)
│       └── ea_classification.yaml     # EA classification config (matched to DRL)
├── notebooks/
│   ├── compare_paper_vs_ea.ipynb      # Fidelity analysis notebook
│   └── compare_classification.ipynb   # Classification analysis notebook
├── tests/
│   ├── __init__.py
│   ├── test_diagnose_fidelity.py      # Diagnostics tests
│   ├── test_log_schema.py             # Schema validation tests
│   ├── test_paper_metadata.py         # Paper metadata validation tests
│   └── test_compute_classif_metrics.py  # Classification metrics tests
└── logs/
    └── schema.json                    # JSON schema for logs (fidelity + classification)
```

## Installation

Install additional dependencies for the comparison workspace:

```bash
pip install -r comparison/requirements.txt
```

## Quick Start

### 1. Review Paper Metadata

The extracted hyperparameters from arXiv:2407.20147 are in:
```
comparison/paper_metadata/quantum_ml_arch_search_2407.20147.json
```

This file contains:
- DRL controller hyperparameters (algorithm, gamma, epsilon, etc.)
- Gate set and constraints
- Inner-loop optimization settings
- Reward design details
- Compute budget and reported metrics
- Notes on values not specified in the paper

### 2. Run Tests

```bash
pytest comparison/tests/ -v
```

### 3. Configure Entrypoint Commands

The YAML configs in `experiments/configs/` contain `entrypoint_command` placeholders.
To use them:

1. For DRL classification (`drl_classification.yaml`):
   ```yaml
   entrypoint_command: |
     python your_drl_agent.py \
       --config comparison/experiments/configs/drl_classification.yaml \
       --dataset make_classification \
       --output comparison/logs/drl/drl_classif_run_{seed}.jsonl \
       --seed {seed}
   ```

2. For EA classification (`ea_classification.yaml`):
   ```yaml
   entrypoint_command: |
     python run_experiments.py \
       --preset quick \
       --n-qubits 4 \
       --base-dir comparison/logs/ea \
       --seed {seed}
   ```

### 4. Generate Logs

Run your experiments with the configured entrypoints. Logs should follow the schema in `logs/schema.json`.

### 5. Analyze Classification Results

After running experiments, compute classification metrics:

```bash
python -m comparison.analysis.compute_classif_metrics \
    --input "comparison/logs/**/*classif*.jsonl" \
    --out comparison/logs/classif_analysis \
    --thresholds 0.70 0.80 0.90
```

### 6. Use the Notebook

Open and run the classification comparison notebook:

```bash
jupyter notebook comparison/notebooks/compare_classification.ipynb
```

## Log Format

All experiment logs should follow the schema in `logs/schema.json`. The schema supports both fidelity-based (unitary synthesis) and classification tasks.

### Base Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `eval_id` | integer | Unique evaluation identifier |
| `timestamp` | string | ISO 8601 timestamp |
| `method` | string | "drl" or "ea" |
| `seed` | integer | Random seed |

### Fidelity-Based Tasks (must include)

| Field | Type | Description |
|-------|------|-------------|
| `best_fidelity` | number | Best fidelity (0-1) |
| `fidelity_metric` | string | Type of fidelity metric |

### Classification Tasks (must include)

| Field | Type | Description |
|-------|------|-------------|
| `best_val_accuracy` | number | Best validation accuracy (0-1) |
| `best_test_accuracy` | number | Best test accuracy (0-1, optional) |

### Optional Fields

- `circuit_depth`, `gate_count`, `wall_time_s`, `cum_eval_count`
- `dataset`, `n_qubits`, `episode`, `generation`
- `unitary_hash`, `notes`

### Example Classification Log Entry
```json
{
  "eval_id": 100,
  "timestamp": "2024-01-15T10:30:00Z",
  "method": "drl",
  "seed": 42,
  "best_val_accuracy": 0.93,
  "best_test_accuracy": 0.91,
  "circuit_depth": 4,
  "gate_count": 4,
  "cum_eval_count": 1200,
  "dataset": "make_classification",
  "n_qubits": 4
}
```

## Alignment Checklist for Fair Comparison

Before running DRL vs EA comparison experiments, verify:

- [ ] **Gate Set**: Both use RX, RY, RZ, CNOT with same connectivity
- [ ] **Max Depth**: Same maximum circuit depth/gate count
- [ ] **Inner-Loop**: Same loss function, epochs, data encoding
- [ ] **Eval Budget**: Comparable total evaluations (e.g., 1200)
- [ ] **Seeds**: Same random seeds for reproducibility
- [ ] **Dataset**: Same dataset with same preprocessing

The `ea_classification.yaml` is pre-configured to match `drl_classification.yaml` settings from the paper.

## Classification Analysis Functions

The `analysis.compute_classif_metrics` module provides:

- `load_logs(paths)`: Load JSON/JSONL log files
- `compute_per_run_classification_metrics(logs)`: Per-run summaries
- `aggregate_classification_metrics(logs)`: Cross-run statistics
- `save_summary(metrics, out_path)`: Save JSON and CSV summaries

Computed metrics include:
- `final_val_accuracy`, `max_val_accuracy`
- `final_test_accuracy` (if present)
- `evals_to_70pct`, `evals_to_80pct`, `evals_to_90pct`
- `num_evals`, `final_gate_count`, `final_depth`

## Paper Metadata Fields

The `paper_metadata/quantum_ml_arch_search_2407.20147.json` contains:

| Field | Description |
|-------|-------------|
| `paper_title`, `authors`, `arxiv_id` | Basic paper info |
| `tasks` | Datasets and preprocessing |
| `drl_controller` | Algorithm, gamma, epsilon, etc. |
| `gate_set_and_constraints` | Allowed gates, max depth |
| `inner_loop_optimization` | Loss, epochs, optimizer |
| `reward_design` | Reward function details |
| `compute_budget_and_repeats` | Episodes, seeds |
| `reported_metrics` | Best results from paper |
| `notes` | Values not specified (set to null) |

### Fields Set to Null (Not in Paper)

The following fields are `null` in the metadata because they were not explicitly specified in arXiv:2407.20147:
- Learning rate for DDQN
- Batch size for RL training
- Number of seeds/repetitions
- Train/test split ratio
- Specific MLP layer sizes
- Inner-loop optimizer and learning rate

## Troubleshooting

### "No module named comparison"

Ensure you're running from the repository root:
```bash
cd /path/to/quantum-arch-search
python -m comparison.analysis.compute_classif_metrics --help
```

### "jsonschema not installed"

Install comparison dependencies:
```bash
pip install -r comparison/requirements.txt
```

### Tests failing

Ensure numpy and other dependencies are installed:
```bash
pip install -r requirements.txt
pip install -r comparison/requirements.txt
```

## Contributing

When adding new comparison tools:

1. Follow the existing code style
2. Add tests in `comparison/tests/`
3. Update this README if adding new functionality
4. Ensure logs conform to `logs/schema.json`

## References

- [arXiv:2407.20147](https://arxiv.org/abs/2407.20147) - Quantum Machine Learning Architecture Search via Deep Reinforcement Learning
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
