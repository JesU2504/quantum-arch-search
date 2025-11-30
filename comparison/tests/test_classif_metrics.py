"""
Tests for classification metrics computation.

These tests verify that the compute_classif_metrics module correctly
processes synthetic log data for classification experiments.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add comparison package to path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from comparison.analysis.compute_classif_metrics import (
    load_logs,
    validate_classification_logs,
    compute_per_run_classification_metrics,
    aggregate_classification_metrics,
    save_classification_summary,
)


@pytest.fixture
def synthetic_drl_logs():
    """Fixture providing synthetic DRL classification log entries."""
    logs = []
    # Simulate 10 episodes improving from 50% to 90% accuracy
    accuracies = [0.5, 0.55, 0.60, 0.65, 0.72, 0.78, 0.82, 0.87, 0.89, 0.91]
    gate_counts = [5, 7, 9, 11, 12, 13, 14, 14, 15, 15]
    
    for i, (acc, gc) in enumerate(zip(accuracies, gate_counts)):
        logs.append({
            "eval_id": i,
            "method": "drl",
            "seed": 0,
            "train_accuracy": acc - 0.02,  # Train slightly lower
            "test_accuracy": acc,
            "gate_count": gc,
            "circuit_depth": gc,
        })
    return logs


@pytest.fixture
def synthetic_ea_logs():
    """Fixture providing synthetic EA classification log entries."""
    logs = []
    # Simulate 10 generations improving from 55% to 85% accuracy
    accuracies = [0.55, 0.58, 0.62, 0.68, 0.72, 0.76, 0.79, 0.81, 0.83, 0.85]
    gate_counts = [4, 5, 6, 8, 10, 11, 12, 13, 14, 16]
    
    for i, (acc, gc) in enumerate(zip(accuracies, gate_counts)):
        logs.append({
            "eval_id": i,
            "method": "ea",
            "seed": 0,
            "train_accuracy": acc - 0.01,
            "test_accuracy": acc,
            "gate_count": gc,
            "circuit_depth": gc,
        })
    return logs


@pytest.fixture
def synthetic_logs_file(synthetic_drl_logs, synthetic_ea_logs):
    """Create a temporary JSONL file with synthetic logs."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for log in synthetic_drl_logs + synthetic_ea_logs:
            f.write(json.dumps(log) + '\n')
        return Path(f.name)


class TestLoadLogs:
    """Tests for log loading functionality."""

    def test_load_jsonl_file(self, synthetic_logs_file):
        """Should load JSONL file correctly."""
        logs = load_logs(str(synthetic_logs_file))
        assert len(logs) == 20  # 10 DRL + 10 EA
        
    def test_load_glob_pattern(self, synthetic_logs_file):
        """Should support glob pattern."""
        logs = load_logs(str(synthetic_logs_file.parent / "*.jsonl"))
        assert len(logs) >= 20

    def test_load_empty_pattern(self):
        """Should return empty list for non-matching pattern."""
        logs = load_logs("/nonexistent/path/*.jsonl")
        assert logs == []


class TestValidateLogs:
    """Tests for log validation functionality."""

    def test_valid_logs_pass(self, synthetic_drl_logs):
        """Valid logs should pass validation."""
        valid, errors = validate_classification_logs(synthetic_drl_logs)
        assert len(valid) == len(synthetic_drl_logs)
        assert len(errors) == 0

    def test_missing_required_field_fails(self):
        """Logs missing required fields should fail validation."""
        logs = [{"train_accuracy": 0.8}]  # Missing eval_id, method, seed
        valid, errors = validate_classification_logs(logs)
        assert len(valid) == 0
        assert len(errors) == 1

    def test_invalid_accuracy_range(self):
        """Accuracy out of [0,1] should be flagged."""
        logs = [{
            "eval_id": 0,
            "method": "drl",
            "seed": 0,
            "train_accuracy": 1.5  # Invalid
        }]
        valid, errors = validate_classification_logs(logs)
        assert len(valid) == 0
        assert len(errors) == 1


class TestPerRunMetrics:
    """Tests for per-run metrics computation."""

    def test_computes_final_accuracy(self, synthetic_drl_logs):
        """Should compute final accuracy correctly."""
        metrics = compute_per_run_classification_metrics(synthetic_drl_logs)
        run_key = "drl_seed0"
        assert run_key in metrics
        assert metrics[run_key]['final_test_accuracy'] == 0.91

    def test_computes_best_accuracy(self, synthetic_drl_logs):
        """Should compute best accuracy correctly."""
        metrics = compute_per_run_classification_metrics(synthetic_drl_logs)
        run_key = "drl_seed0"
        assert metrics[run_key]['best_test_accuracy'] == 0.91

    def test_computes_evals_to_threshold(self, synthetic_drl_logs):
        """Should compute evaluations to threshold correctly."""
        metrics = compute_per_run_classification_metrics(synthetic_drl_logs)
        run_key = "drl_seed0"
        
        # 70% reached at index 4 (accuracy 0.72), so evals = 5
        assert metrics[run_key]['evals_to_70_accuracy'] == 5
        
        # 80% reached at index 6 (accuracy 0.82), so evals = 7
        assert metrics[run_key]['evals_to_80_accuracy'] == 7
        
        # 90% reached at index 9 (accuracy 0.91), so evals = 10
        assert metrics[run_key]['evals_to_90_accuracy'] == 10

    def test_computes_gate_count_metrics(self, synthetic_drl_logs):
        """Should compute gate count metrics correctly."""
        metrics = compute_per_run_classification_metrics(synthetic_drl_logs)
        run_key = "drl_seed0"
        assert metrics[run_key]['final_gate_count'] == 15
        assert metrics[run_key]['min_gate_count'] == 5


class TestAggregateMetrics:
    """Tests for aggregate metrics computation."""

    def test_aggregates_by_method(self, synthetic_drl_logs, synthetic_ea_logs):
        """Should aggregate metrics by method."""
        all_logs = synthetic_drl_logs + synthetic_ea_logs
        metrics = aggregate_classification_metrics(all_logs)
        
        assert 'by_method' in metrics
        assert 'drl' in metrics['by_method']
        assert 'ea' in metrics['by_method']

    def test_computes_method_statistics(self, synthetic_drl_logs, synthetic_ea_logs):
        """Should compute statistics across runs."""
        all_logs = synthetic_drl_logs + synthetic_ea_logs
        metrics = aggregate_classification_metrics(all_logs)
        
        drl_stats = metrics['by_method']['drl']
        assert 'n_runs' in drl_stats
        assert 'mean_best_test_accuracy' in drl_stats

    def test_total_counts(self, synthetic_drl_logs, synthetic_ea_logs):
        """Should count totals correctly."""
        all_logs = synthetic_drl_logs + synthetic_ea_logs
        metrics = aggregate_classification_metrics(all_logs)
        
        assert metrics['total_logs'] == 20
        assert metrics['total_runs'] == 2


class TestSaveClassificationSummary:
    """Tests for summary saving functionality."""

    def test_saves_json_file(self, synthetic_drl_logs, synthetic_ea_logs, tmp_path):
        """Should save JSON summary file."""
        all_logs = synthetic_drl_logs + synthetic_ea_logs
        metrics = aggregate_classification_metrics(all_logs)
        
        json_path, csv_path = save_classification_summary(metrics, tmp_path)
        
        assert json_path.exists()
        assert json_path.suffix == '.json'
        
        # Verify JSON is valid
        with open(json_path) as f:
            loaded = json.load(f)
        assert 'by_method' in loaded

    def test_saves_csv_file(self, synthetic_drl_logs, synthetic_ea_logs, tmp_path):
        """Should save CSV per-run file."""
        all_logs = synthetic_drl_logs + synthetic_ea_logs
        metrics = aggregate_classification_metrics(all_logs)
        
        json_path, csv_path = save_classification_summary(metrics, tmp_path)
        
        assert csv_path.exists()
        assert csv_path.suffix == '.csv'
        
        # Verify CSV has content
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) > 1  # Header + at least one data row


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_logs(self):
        """Should handle empty logs gracefully."""
        metrics = aggregate_classification_metrics([])
        assert metrics['total_logs'] == 0
        assert metrics['total_runs'] == 0

    def test_missing_optional_fields(self):
        """Should handle logs with missing optional fields."""
        logs = [{
            "eval_id": 0,
            "method": "drl",
            "seed": 0,
            "test_accuracy": 0.8
            # No gate_count, circuit_depth
        }]
        
        metrics = compute_per_run_classification_metrics(logs)
        run_key = "drl_seed0"
        assert run_key in metrics
        assert metrics[run_key]['final_gate_count'] is None

    def test_single_entry_run(self):
        """Should handle runs with single entry."""
        logs = [{
            "eval_id": 0,
            "method": "drl",
            "seed": 0,
            "test_accuracy": 0.85,
            "gate_count": 10
        }]
        
        metrics = compute_per_run_classification_metrics(logs)
        run_key = "drl_seed0"
        assert metrics[run_key]['final_test_accuracy'] == 0.85
        assert metrics[run_key]['best_test_accuracy'] == 0.85


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
