"""
Tests for log schema validation.

These tests verify that log entries conform to the expected JSON schema
for fair comparison between DRL and EA methods.
"""

import json
import sys
from pathlib import Path

import pytest

# Add comparison package to path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Try to import jsonschema
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def load_schema():
    """Load the log schema from the comparison/logs directory."""
    schema_path = Path(__file__).parent.parent / 'logs' / 'schema.json'
    with open(schema_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def schema():
    """Fixture to load the schema."""
    return load_schema()


@pytest.fixture
def valid_log_entry():
    """Fixture providing a valid log entry."""
    return {
        "eval_id": 1,
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


@pytest.fixture
def valid_log_entry_with_optional():
    """Fixture providing a valid log entry with optional fields."""
    return {
        "eval_id": 1,
        "timestamp": "2024-01-15T10:30:00Z",
        "method": "drl",
        "seed": 0,
        "best_fidelity": 0.9999,
        "fidelity_metric": "average_gate",
        "circuit_depth": 8,
        "gate_count": 15,
        "wall_time_s": 1.5,
        "cum_eval_count": 500,
        "unitary_hash": "abc123def456",
        "notes": "Test run with PPO algorithm"
    }


class TestSchemaExists:
    """Tests for schema file existence and structure."""

    def test_schema_file_exists(self):
        """Schema file should exist."""
        schema_path = Path(__file__).parent.parent / 'logs' / 'schema.json'
        assert schema_path.exists(), f"Schema file not found at {schema_path}"

    def test_schema_is_valid_json(self):
        """Schema file should be valid JSON."""
        schema_path = Path(__file__).parent.parent / 'logs' / 'schema.json'
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        assert isinstance(schema, dict)

    def test_schema_has_required_fields(self, schema):
        """Schema should define required fields."""
        assert 'required' in schema
        required = schema['required']
        expected_required = [
            'eval_id', 'timestamp', 'method', 'seed',
            'best_fidelity', 'fidelity_metric'
        ]
        for field in expected_required:
            assert field in required, f"Required field '{field}' not in schema"


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestSchemaValidation:
    """Tests for schema validation with jsonschema."""

    def test_valid_entry_passes(self, schema, valid_log_entry):
        """Valid log entry should pass validation."""
        jsonschema.validate(valid_log_entry, schema)

    def test_valid_entry_with_optional_passes(self, schema, valid_log_entry_with_optional):
        """Valid log entry with optional fields should pass validation."""
        jsonschema.validate(valid_log_entry_with_optional, schema)

    def test_missing_required_field_fails(self, schema, valid_log_entry):
        """Missing required field should fail validation."""
        del valid_log_entry['best_fidelity']
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(valid_log_entry, schema)

    def test_invalid_fidelity_type_fails(self, schema, valid_log_entry):
        """Invalid fidelity type should fail validation."""
        valid_log_entry['best_fidelity'] = "not a number"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(valid_log_entry, schema)

    def test_fidelity_out_of_range_fails(self, schema, valid_log_entry):
        """Fidelity > 1 should fail validation."""
        valid_log_entry['best_fidelity'] = 1.5
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(valid_log_entry, schema)

    def test_negative_fidelity_fails(self, schema, valid_log_entry):
        """Negative fidelity should fail validation."""
        valid_log_entry['best_fidelity'] = -0.1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(valid_log_entry, schema)

    def test_invalid_method_fails(self, schema, valid_log_entry):
        """Invalid method should fail validation."""
        valid_log_entry['method'] = "invalid_method"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(valid_log_entry, schema)

    def test_negative_eval_id_fails(self, schema, valid_log_entry):
        """Negative eval_id should fail validation."""
        valid_log_entry['eval_id'] = -1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(valid_log_entry, schema)

    def test_negative_gate_count_fails(self, schema, valid_log_entry):
        """Negative gate_count should fail validation."""
        valid_log_entry['gate_count'] = -5
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(valid_log_entry, schema)


class TestSyntheticLogValidation:
    """Tests for synthetic log entries (no jsonschema required)."""

    def test_synthetic_drl_entry_structure(self):
        """Synthetic DRL log entry should have expected structure."""
        entry = {
            "eval_id": 100,
            "timestamp": "2024-01-15T12:00:00Z",
            "method": "drl",
            "seed": 1,
            "best_fidelity": 0.95,
            "fidelity_metric": "trace_overlap",
            "circuit_depth": 10,
            "gate_count": 20,
            "wall_time_s": 5.0,
            "cum_eval_count": 10000
        }

        # Basic structural checks
        assert entry['method'] in ['drl', 'ea']
        assert 0 <= entry['best_fidelity'] <= 1
        assert entry['eval_id'] >= 0
        assert entry['gate_count'] >= 0
        assert entry['circuit_depth'] >= 0
        assert entry['wall_time_s'] >= 0

    def test_synthetic_ea_entry_structure(self):
        """Synthetic EA log entry should have expected structure."""
        entry = {
            "eval_id": 50,
            "timestamp": "2024-01-15T14:30:00Z",
            "method": "ea",
            "seed": 2,
            "best_fidelity": 0.999,
            "fidelity_metric": "average_gate",
            "circuit_depth": 6,
            "gate_count": 12,
            "wall_time_s": 2.5,
            "cum_eval_count": 5000,
            "notes": "Adversarial coevolution run"
        }

        # Basic structural checks
        assert entry['method'] in ['drl', 'ea']
        assert 0 <= entry['best_fidelity'] <= 1
        assert entry['eval_id'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
