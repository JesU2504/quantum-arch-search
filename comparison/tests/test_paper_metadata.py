"""
Tests for paper metadata JSON file.

These tests verify that the paper metadata file exists and contains
the required fields for fair comparison between DRL and EA methods.
"""

import json
import sys
from pathlib import Path

import pytest

# Add comparison package to path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


METADATA_PATH = Path(__file__).parent.parent / 'paper_metadata' / 'quantum_ml_arch_search_2407.20147.json'


class TestPaperMetadataExists:
    """Tests for paper metadata file existence."""

    def test_metadata_file_exists(self):
        """Paper metadata file should exist."""
        assert METADATA_PATH.exists(), f"Metadata file not found at {METADATA_PATH}"

    def test_metadata_is_valid_json(self):
        """Metadata file should be valid JSON."""
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        assert isinstance(metadata, dict)


class TestPaperMetadataRequiredFields:
    """Tests for required fields in paper metadata."""

    @pytest.fixture
    def metadata(self):
        """Load metadata fixture."""
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)

    def test_has_paper_title(self, metadata):
        """Metadata should have paper_title."""
        assert 'paper_title' in metadata
        assert isinstance(metadata['paper_title'], str)
        assert len(metadata['paper_title']) > 0

    def test_has_authors(self, metadata):
        """Metadata should have authors list."""
        assert 'authors' in metadata
        assert isinstance(metadata['authors'], list)
        assert len(metadata['authors']) > 0

    def test_has_arxiv_id(self, metadata):
        """Metadata should have arxiv_id."""
        assert 'arxiv_id' in metadata
        assert metadata['arxiv_id'] == '2407.20147'

    def test_has_arxiv_url(self, metadata):
        """Metadata should have arxiv_url."""
        assert 'arxiv_url' in metadata
        # Verify this is a valid arXiv URL (not sanitizing user input, just validating metadata)
        url = metadata['arxiv_url']
        assert url.startswith('https://arxiv.org/abs/') or url.startswith('http://arxiv.org/abs/')

    def test_has_tasks(self, metadata):
        """Metadata should have tasks/datasets information."""
        assert 'tasks' in metadata
        assert isinstance(metadata['tasks'], list)
        assert len(metadata['tasks']) > 0
        
        # Check each task has required fields
        for task in metadata['tasks']:
            assert 'name' in task
            assert 'description' in task

    def test_has_drl_algorithm(self, metadata):
        """Metadata should have DRL algorithm information."""
        assert 'drl_algorithm' in metadata
        assert 'name' in metadata['drl_algorithm']
        assert 'DDQN' in metadata['drl_algorithm']['name'] or 'DQN' in metadata['drl_algorithm']['name']

    def test_has_policy_network(self, metadata):
        """Metadata should have policy network architecture."""
        assert 'policy_network_architecture' in metadata
        assert 'type' in metadata['policy_network_architecture']

    def test_has_training_hyperparameters(self, metadata):
        """Metadata should have training hyperparameters."""
        assert 'training_hyperparameters' in metadata
        
    def test_has_state_representation(self, metadata):
        """Metadata should have state representation."""
        assert 'state_representation' in metadata
        assert 'type' in metadata['state_representation']
        assert 'shape' in metadata['state_representation']

    def test_has_action_space(self, metadata):
        """Metadata should have action space information."""
        assert 'action_space' in metadata
        assert 'gate_set' in metadata['action_space']
        gates = metadata['action_space']['gate_set']
        assert 'RX' in gates or 'RY' in gates or 'RZ' in gates
        assert 'CNOT' in gates

    def test_has_max_depth_termination(self, metadata):
        """Metadata should have max depth/termination conditions."""
        assert 'max_depth_termination' in metadata
        assert 'max_gates' in metadata['max_depth_termination']

    def test_has_inner_loop_optimization(self, metadata):
        """Metadata should have inner-loop optimization details."""
        assert 'inner_loop_optimization' in metadata
        inner_loop = metadata['inner_loop_optimization']
        assert 'max_epochs_per_step' in inner_loop

    def test_has_reward_function(self, metadata):
        """Metadata should have reward function information."""
        assert 'reward_function' in metadata
        reward = metadata['reward_function']
        assert 'type' in reward

    def test_has_compute_budget(self, metadata):
        """Metadata should have compute budget information."""
        assert 'compute_budget' in metadata

    def test_has_reported_metrics(self, metadata):
        """Metadata should have reported metrics information."""
        assert 'reported_metrics' in metadata
        assert 'primary' in metadata['reported_metrics']


class TestPaperMetadataValues:
    """Tests for specific values in paper metadata."""

    @pytest.fixture
    def metadata(self):
        """Load metadata fixture."""
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)

    def test_task_type_is_classification(self, metadata):
        """Task type should be classification."""
        assert 'task_type' in metadata
        assert metadata['task_type'] == 'classification'

    def test_datasets_are_sklearn(self, metadata):
        """Datasets should be sklearn classification datasets."""
        task_names = [t['name'] for t in metadata['tasks']]
        assert 'make_classification' in task_names or 'make_moons' in task_names

    def test_gate_set_includes_rotation_gates(self, metadata):
        """Gate set should include rotation gates."""
        gates = metadata['action_space']['gate_set']
        rotation_gates = ['RX', 'RY', 'RZ']
        assert any(g in gates for g in rotation_gates)

    def test_max_gates_is_reasonable(self, metadata):
        """Max gates should be in reasonable range."""
        max_gates = metadata['max_depth_termination']['max_gates']['L_values_used']
        assert isinstance(max_gates, list)
        assert all(1 <= g <= 100 for g in max_gates)

    def test_inner_loop_epochs_reasonable(self, metadata):
        """Inner loop epochs should be reasonable."""
        epochs = metadata['inner_loop_optimization']['max_epochs_per_step']
        assert 1 <= epochs <= 100

    def test_has_notes_for_omissions(self, metadata):
        """Metadata should document omissions and assumptions."""
        assert 'notes_and_omissions' in metadata
        omissions = metadata['notes_and_omissions']
        assert 'omitted_hyperparameters' in omissions
        assert 'assumptions_for_reproduction' in omissions


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
