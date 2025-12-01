"""
Tests for demo agent runners.

These tests verify that the demo DRL and EA agents generate valid
synthetic log data matching the expected schema.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parent.parent.parent


class TestDRLAgent:
    """Tests for the demo DRL agent."""

    def test_runs_without_error(self, tmp_path):
        """Demo DRL agent should run successfully."""
        output_file = tmp_path / "test_drl.jsonl"
        config_file = _repo_root / "comparison" / "experiments" / "configs" / "drl_classification.yaml"
        
        result = subprocess.run(
            [
                sys.executable,
                str(_repo_root / "tools" / "run_drl_agent.py"),
                "--config", str(config_file),
                "--seed", "42",
                "--output", str(output_file),
                "--episodes", "5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0, f"DRL agent failed: {result.stderr}"

    def test_generates_jsonl_output(self, tmp_path):
        """Demo DRL agent should generate valid JSONL output."""
        output_file = tmp_path / "test_drl.jsonl"
        config_file = _repo_root / "comparison" / "experiments" / "configs" / "drl_classification.yaml"
        
        subprocess.run(
            [
                sys.executable,
                str(_repo_root / "tools" / "run_drl_agent.py"),
                "--config", str(config_file),
                "--seed", "42",
                "--output", str(output_file),
                "--episodes", "5",
            ],
            capture_output=True,
            timeout=30,
        )
        
        assert output_file.exists()
        
        with open(output_file) as f:
            lines = f.readlines()
        
        assert len(lines) == 5
        
        for line in lines:
            entry = json.loads(line)
            assert "eval_id" in entry
            assert "method" in entry
            assert entry["method"] == "drl"
            assert "seed" in entry
            assert entry["seed"] == 42
            assert "test_accuracy" in entry
            assert 0.0 <= entry["test_accuracy"] <= 1.0

    def test_uses_seed_for_reproducibility(self, tmp_path):
        """Same seed should produce same results."""
        output1 = tmp_path / "test_drl1.jsonl"
        output2 = tmp_path / "test_drl2.jsonl"
        config_file = _repo_root / "comparison" / "experiments" / "configs" / "drl_classification.yaml"
        
        for output in [output1, output2]:
            subprocess.run(
                [
                    sys.executable,
                    str(_repo_root / "tools" / "run_drl_agent.py"),
                    "--config", str(config_file),
                    "--seed", "123",
                    "--output", str(output),
                    "--episodes", "3",
                ],
                capture_output=True,
                timeout=30,
            )
        
        with open(output1) as f1, open(output2) as f2:
            logs1 = [json.loads(line) for line in f1]
            logs2 = [json.loads(line) for line in f2]
        
        # Accuracy and gate_count should be identical for same seed
        for log1, log2 in zip(logs1, logs2):
            assert log1["test_accuracy"] == log2["test_accuracy"]
            assert log1["gate_count"] == log2["gate_count"]


class TestEAAgent:
    """Tests for the demo EA agent."""

    def test_runs_without_error(self, tmp_path):
        """Demo EA agent should run successfully."""
        output_file = tmp_path / "test_ea.jsonl"
        config_file = _repo_root / "comparison" / "experiments" / "configs" / "ea_classification.yaml"
        
        result = subprocess.run(
            [
                sys.executable,
                str(_repo_root / "tools" / "run_ea_agent.py"),
                "--config", str(config_file),
                "--seed", "42",
                "--output", str(output_file),
                "--generations", "5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0, f"EA agent failed: {result.stderr}"

    def test_generates_jsonl_output(self, tmp_path):
        """Demo EA agent should generate valid JSONL output."""
        output_file = tmp_path / "test_ea.jsonl"
        config_file = _repo_root / "comparison" / "experiments" / "configs" / "ea_classification.yaml"
        
        subprocess.run(
            [
                sys.executable,
                str(_repo_root / "tools" / "run_ea_agent.py"),
                "--config", str(config_file),
                "--seed", "42",
                "--output", str(output_file),
                "--generations", "5",
            ],
            capture_output=True,
            timeout=30,
        )
        
        assert output_file.exists()
        
        with open(output_file) as f:
            lines = f.readlines()
        
        assert len(lines) == 5
        
        for line in lines:
            entry = json.loads(line)
            assert "eval_id" in entry
            assert "method" in entry
            assert entry["method"] == "ea"
            assert "seed" in entry
            assert entry["seed"] == 42
            assert "test_accuracy" in entry
            assert 0.0 <= entry["test_accuracy"] <= 1.0
            assert "generation" in entry  # EA-specific field
            assert "population_best_fitness" in entry  # EA-specific field

    def test_uses_seed_for_reproducibility(self, tmp_path):
        """Same seed should produce same results."""
        output1 = tmp_path / "test_ea1.jsonl"
        output2 = tmp_path / "test_ea2.jsonl"
        config_file = _repo_root / "comparison" / "experiments" / "configs" / "ea_classification.yaml"
        
        for output in [output1, output2]:
            subprocess.run(
                [
                    sys.executable,
                    str(_repo_root / "tools" / "run_ea_agent.py"),
                    "--config", str(config_file),
                    "--seed", "123",
                    "--output", str(output),
                    "--generations", "3",
                ],
                capture_output=True,
                timeout=30,
            )
        
        with open(output1) as f1, open(output2) as f2:
            logs1 = [json.loads(line) for line in f1]
            logs2 = [json.loads(line) for line in f2]
        
        # Accuracy and gate_count should be identical for same seed
        for log1, log2 in zip(logs1, logs2):
            assert log1["test_accuracy"] == log2["test_accuracy"]
            assert log1["gate_count"] == log2["gate_count"]


class TestRunExperimentsScript:
    """Tests for the comparison run_experiments.sh script."""

    def test_help_flag(self):
        """Script should show help when --help is used."""
        result = subprocess.run(
            [str(_repo_root / "comparison" / "run_experiments.sh"), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        assert result.returncode == 0
        assert "method" in result.stdout.lower()
        assert "seeds" in result.stdout.lower()
        assert "dry-run" in result.stdout.lower()

    def test_dry_run_flag(self, tmp_path):
        """Script should show commands without executing in dry-run mode."""
        result = subprocess.run(
            [
                str(_repo_root / "comparison" / "run_experiments.sh"),
                "--dry-run",
                "--method", "drl",
                "--seeds", "42",
                "--log-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(_repo_root),
        )
        
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "Would execute:" in result.stdout
        assert "python tools/run_drl_agent.py" in result.stdout

    def test_dry_run_does_not_execute(self, tmp_path):
        """Dry-run mode should not create output files."""
        result = subprocess.run(
            [
                str(_repo_root / "comparison" / "run_experiments.sh"),
                "--dry-run",
                "--method", "both",
                "--seeds", "42",
                "--log-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(_repo_root),
        )
        
        assert result.returncode == 0
        # No log files should be created in dry-run mode (directories may exist)
        drl_logs = list((tmp_path / "drl").glob("*.log")) if (tmp_path / "drl").exists() else []
        ea_logs = list((tmp_path / "ea").glob("*.log")) if (tmp_path / "ea").exists() else []
        assert len(drl_logs) == 0, "Dry-run should not create DRL log files"
        assert len(ea_logs) == 0, "Dry-run should not create EA log files"

    def test_full_pipeline_execution(self, tmp_path):
        """Script should execute full pipeline with demo runners."""
        result = subprocess.run(
            [
                str(_repo_root / "comparison" / "run_experiments.sh"),
                "--method", "both",
                "--seeds", "42",
                "--log-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(_repo_root),
        )
        
        assert result.returncode == 0
        assert "completed successfully" in result.stdout
        
        # Check that log files were created
        drl_log = tmp_path / "drl" / "drl_classif_seed42.log"
        ea_log = tmp_path / "ea" / "ea_classif_seed42.log"
        assert drl_log.exists(), f"DRL log file not created: {drl_log}"
        assert ea_log.exists(), f"EA log file not created: {ea_log}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
