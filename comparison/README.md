# Comparison Workspace: DRL vs EA Quantum Architecture Search

This scaffold provides tools for comparing the Deep Reinforcement Learning (DRL) approach from [arXiv:2407.20147](https://arxiv.org/abs/2407.20147) with the coevolutionary (EA) agents implemented in this repository.

## Overview

The comparison workspace does **not** reimplement existing agents. Instead, it provides:

- **Diagnostics**: Fidelity computation utilities for verifying unitary comparisons
- **Analysis**: Tools to aggregate metrics from experiment logs
- **Configs**: Example YAML configurations for running DRL and EA experiments
- **Notebooks**: Jupyter notebooks for visualization and analysis
- **Tests**: Pytest tests for validation
- **Schema**: JSON schema for standardized log format

## Directory Structure

```
comparison/
├── README.md                           # This file
├── requirements.txt                    # Additional dependencies
├── diagnostics/
│   ├── __init__.py
│   └── diagnose_fidelity.py           # Fidelity computation functions
├── analysis/
│   ├── __init__.py
│   └── compute_metrics.py             # Log aggregation and metrics
├── experiments/
│   ├── __init__.py
│   └── configs/
│       ├── drl.yaml                   # DRL agent configuration
│       └── ea.yaml                    # EA agent configuration
├── notebooks/
│   └── compare_paper_vs_ea.ipynb      # Analysis notebook
├── tests/
│   ├── __init__.py
│   ├── test_diagnose_fidelity.py      # Diagnostics tests
│   └── test_log_schema.py             # Schema validation tests
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
