"""
Tests for classification metrics computation module.

These tests verify that compute_classif_metrics.py correctly computes
classification-specific metrics from experiment logs.
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
    compute_per_run_classification_metrics,
    aggregate_classification_metrics,
    save_summary,
)


@pytest.fixture
def synthetic_drl_logs():
    """Fixture providing synthetic DRL classification log entries."""
    logs = []
    for i in range(20):
        logs.append({
            "eval_id": i,
            "timestamp": f"2024-01-15T10:{i:02d}:00Z",
            "method": "drl",
            "seed": 42,
            "best_val_accuracy": min(0.5 + i * 0.025, 0.95),
            "best_test_accuracy": min(0.48 + i * 0.024, 0.93),
            "gate_count": max(2, 10 - i // 4),
            "circuit_depth": max(2, 8 - i // 5),
            "cum_eval_count": (i + 1) * 10,
        })
    return logs


@pytest.fixture
def synthetic_ea_logs():
    """Fixture providing synthetic EA classification log entries."""
    logs = []
    for i in range(20):
        logs.append({
            "eval_id": i,
            "timestamp": f"2024-01-15T11:{i:02d}:00Z",
            "method": "ea",
            "seed": 42,
            "best_val_accuracy": min(0.45 + i * 0.028, 0.98),
            "gate_count": max(3, 12 - i // 3),
            "circuit_depth": max(3, 10 - i // 4),
            "cum_eval_count": (i + 1) * 10,
        })
    return logs


@pytest.fixture
def synthetic_logs(synthetic_drl_logs, synthetic_ea_logs):
    """Fixture combining DRL and EA logs."""
    return synthetic_drl_logs + synthetic_ea_logs


class TestLoadLogs:
    """Tests for log loading functionality."""

    def test_load_jsonl_file(self, synthetic_drl_logs):
        """Should load logs from JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for log in synthetic_drl_logs:
                f.write(json.dumps(log) + '\n')
            temp_path = f.name

        loaded = load_logs(temp_path)
        assert len(loaded) == len(synthetic_drl_logs)
        Path(temp_path).unlink()

    def test_load_json_file(self, synthetic_drl_logs):
        """Should load logs from JSON array file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(synthetic_drl_logs, f)
            temp_path = f.name

        loaded = load_logs(temp_path)
        assert len(loaded) == len(synthetic_drl_logs)
        Path(temp_path).unlink()

    def test_load_empty_pattern_returns_empty(self):
        """Should return empty list for non-matching pattern."""
        loaded = load_logs('/nonexistent/path/*.jsonl')
        assert loaded == []


class TestComputePerRunMetrics:
    """Tests for per-run metric computation."""

    def test_computes_final_val_accuracy(self, synthetic_drl_logs):
        """Should compute final_val_accuracy correctly."""
        runs = compute_per_run_classification_metrics(synthetic_drl_logs)
        assert len(runs) == 1

        run = list(runs.values())[0]
        assert 'final_val_accuracy' in run
        assert run['final_val_accuracy'] == pytest.approx(0.95, rel=0.01)

    def test_computes_max_val_accuracy(self, synthetic_drl_logs):
        """Should compute max_val_accuracy correctly."""
        runs = compute_per_run_classification_metrics(synthetic_drl_logs)
        run = list(runs.values())[0]

        assert 'max_val_accuracy' in run
        assert run['max_val_accuracy'] == pytest.approx(0.95, rel=0.01)

    def test_computes_final_test_accuracy(self, synthetic_drl_logs):
        """Should compute final_test_accuracy when present."""
        runs = compute_per_run_classification_metrics(synthetic_drl_logs)
        run = list(runs.values())[0]

        assert 'final_test_accuracy' in run
        assert run['final_test_accuracy'] == pytest.approx(0.93, rel=0.01)

    def test_computes_num_evals(self, synthetic_drl_logs):
        """Should compute num_evals correctly."""
        runs = compute_per_run_classification_metrics(synthetic_drl_logs)
        run = list(runs.values())[0]

        assert 'num_evals' in run
        assert run['num_evals'] == 20

    def test_computes_evals_to_thresholds(self, synthetic_drl_logs):
        """Should compute evals_to_threshold for each threshold."""
        thresholds = [0.70, 0.80, 0.90]
        runs = compute_per_run_classification_metrics(synthetic_drl_logs, thresholds=thresholds)
        run = list(runs.values())[0]

        # Check threshold keys exist
        for thresh in thresholds:
            key = f'evals_to_{int(thresh * 100)}pct'
            assert key in run

        # Check 70% threshold is reached
        assert run['evals_to_70pct'] is not None
        assert run['evals_to_70pct'] <= 20

    def test_handles_missing_test_accuracy(self, synthetic_ea_logs):
        """Should handle logs without test accuracy."""
        runs = compute_per_run_classification_metrics(synthetic_ea_logs)
        run = list(runs.values())[0]

        assert 'final_test_accuracy' in run
        assert run['final_test_accuracy'] is None

    def test_computes_gate_count_summaries(self, synthetic_drl_logs):
        """Should compute gate_count summaries."""
        runs = compute_per_run_classification_metrics(synthetic_drl_logs)
        run = list(runs.values())[0]

        assert 'final_gate_count' in run
        assert 'min_gate_count' in run
        assert run['final_gate_count'] is not None
        assert run['min_gate_count'] is not None

    def test_computes_depth_summaries(self, synthetic_drl_logs):
        """Should compute depth summaries."""
        runs = compute_per_run_classification_metrics(synthetic_drl_logs)
        run = list(runs.values())[0]

        assert 'final_depth' in run
        assert 'min_depth' in run


class TestAggregateMetrics:
    """Tests for aggregated metric computation."""

    def test_groups_by_method(self, synthetic_logs):
        """Should group runs by method."""
        metrics = aggregate_classification_metrics(synthetic_logs)

        assert 'by_method' in metrics
        assert 'drl' in metrics['by_method']
        assert 'ea' in metrics['by_method']

    def test_computes_mean_max_val_accuracy(self, synthetic_logs):
        """Should compute mean_max_val_accuracy per method."""
        metrics = aggregate_classification_metrics(synthetic_logs)

        for method in ['drl', 'ea']:
            assert 'mean_max_val_accuracy' in metrics['by_method'][method]
            assert metrics['by_method'][method]['mean_max_val_accuracy'] is not None

    def test_computes_threshold_stats(self, synthetic_logs):
        """Should compute threshold statistics per method."""
        thresholds = [0.70, 0.80, 0.90]
        metrics = aggregate_classification_metrics(synthetic_logs, thresholds=thresholds)

        for method in ['drl', 'ea']:
            for thresh in thresholds:
                pct = int(thresh * 100)
                assert f'n_reached_{pct}pct' in metrics['by_method'][method]

    def test_includes_total_counts(self, synthetic_logs):
        """Should include total_logs and total_runs."""
        metrics = aggregate_classification_metrics(synthetic_logs)

        assert 'total_logs' in metrics
        assert 'total_runs' in metrics
        assert metrics['total_logs'] == len(synthetic_logs)


class TestSaveSummary:
    """Tests for saving metrics summary."""

    def test_saves_json_and_csv(self, synthetic_logs):
        """Should save both JSON and CSV files."""
        metrics = aggregate_classification_metrics(synthetic_logs)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, csv_path = save_summary(metrics, tmpdir)

            assert json_path.exists()
            assert csv_path.exists()
            assert json_path.suffix == '.json'
            assert csv_path.suffix == '.csv'

    def test_json_is_valid(self, synthetic_logs):
        """Saved JSON should be valid and loadable."""
        metrics = aggregate_classification_metrics(synthetic_logs)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, _ = save_summary(metrics, tmpdir)

            with open(json_path) as f:
                loaded = json.load(f)

            assert 'by_method' in loaded
            assert 'total_logs' in loaded

    def test_csv_has_expected_columns(self, synthetic_logs):
        """Saved CSV should have expected columns."""
        metrics = aggregate_classification_metrics(synthetic_logs)

        with tempfile.TemporaryDirectory() as tmpdir:
            _, csv_path = save_summary(metrics, tmpdir)

            with open(csv_path) as f:
                header = f.readline().strip().split(',')

            assert 'run_key' in header
            assert 'method' in header
            assert 'seed' in header


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_logs_returns_empty_metrics(self):
        """Should handle empty log list gracefully."""
        metrics = aggregate_classification_metrics([])

        assert metrics['total_logs'] == 0
        assert metrics['total_runs'] == 0
        assert metrics['by_method'] == {}

    def test_single_log_entry(self):
        """Should handle single log entry."""
        logs = [{
            "eval_id": 0,
            "timestamp": "2024-01-15T10:00:00Z",
            "method": "drl",
            "seed": 0,
            "best_val_accuracy": 0.85,
        }]

        metrics = aggregate_classification_metrics(logs)

        assert metrics['total_logs'] == 1
        assert metrics['total_runs'] == 1

    def test_handles_fidelity_fallback(self):
        """Should use best_fidelity as fallback for accuracy."""
        logs = [{
            "eval_id": 0,
            "timestamp": "2024-01-15T10:00:00Z",
            "method": "drl",
            "seed": 0,
            "best_fidelity": 0.99,  # Fallback key
        }]

        runs = compute_per_run_classification_metrics(logs)
        run = list(runs.values())[0]

        # Should use best_fidelity as fallback
        assert run['final_val_accuracy'] == pytest.approx(0.99, rel=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
