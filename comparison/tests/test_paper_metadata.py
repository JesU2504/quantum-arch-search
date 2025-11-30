"""
Tests for paper metadata JSON file.

These tests verify that the paper metadata JSON exists and contains
all required keys for fair comparison between DRL and EA methods.
"""

import json
import sys
from pathlib import Path

import pytest

# Add comparison package to path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def load_paper_metadata():
    """Load the paper metadata JSON file."""
    metadata_path = Path(__file__).parent.parent / 'paper_metadata' / 'quantum_ml_arch_search_2407.20147.json'
    with open(metadata_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def metadata():
    """Fixture to load the paper metadata."""
    return load_paper_metadata()


class TestPaperMetadataExists:
    """Tests for paper metadata file existence and structure."""

    def test_metadata_file_exists(self):
        """Paper metadata file should exist."""
        metadata_path = Path(__file__).parent.parent / 'paper_metadata' / 'quantum_ml_arch_search_2407.20147.json'
        assert metadata_path.exists(), f"Metadata file not found at {metadata_path}"

    def test_metadata_is_valid_json(self):
        """Paper metadata file should be valid JSON."""
        metadata_path = Path(__file__).parent.parent / 'paper_metadata' / 'quantum_ml_arch_search_2407.20147.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert isinstance(metadata, dict)


class TestRequiredTopLevelKeys:
    """Tests for required top-level keys in paper metadata."""

    REQUIRED_KEYS = [
        'paper_title',
        'authors',
        'arxiv_id',
        'arxiv_url',
        'paper_pdf',
        'tasks',
        'drl_controller',
        'controller_action_state',
        'gate_set_and_constraints',
        'inner_loop_optimization',
        'reward_design',
        'compute_budget_and_repeats',
        'reported_metrics',
        'notes',
    ]

    def test_has_all_required_keys(self, metadata):
        """Metadata should contain all required top-level keys."""
        for key in self.REQUIRED_KEYS:
            assert key in metadata, f"Required key '{key}' not in metadata"

    def test_paper_title_is_string(self, metadata):
        """paper_title should be a non-empty string."""
        assert isinstance(metadata['paper_title'], str)
        assert len(metadata['paper_title']) > 0

    def test_authors_is_list(self, metadata):
        """authors should be a non-empty list of strings."""
        assert isinstance(metadata['authors'], list)
        assert len(metadata['authors']) > 0
        for author in metadata['authors']:
            assert isinstance(author, str)

    def test_arxiv_id_format(self, metadata):
        """arxiv_id should match expected format."""
        arxiv_id = metadata['arxiv_id']
        assert isinstance(arxiv_id, str)
        assert '2407.20147' in arxiv_id

    def test_arxiv_url_format(self, metadata):
        """arxiv_url should be a valid arXiv URL."""
        arxiv_url = metadata['arxiv_url']
        assert isinstance(arxiv_url, str)
        assert 'arxiv.org' in arxiv_url


class TestTasksSection:
    """Tests for the tasks section of paper metadata."""

    def test_tasks_has_required_keys(self, metadata):
        """tasks section should have required keys."""
        tasks = metadata['tasks']
        assert 'task_type' in tasks
        assert 'datasets' in tasks
        assert 'preprocessing' in tasks

    def test_datasets_is_list(self, metadata):
        """datasets should be a non-empty list."""
        datasets = metadata['tasks']['datasets']
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_each_dataset_has_name(self, metadata):
        """Each dataset should have a name field."""
        for dataset in metadata['tasks']['datasets']:
            assert 'name' in dataset
            assert isinstance(dataset['name'], str)


class TestDRLControllerSection:
    """Tests for the drl_controller section of paper metadata."""

    def test_drl_controller_has_algorithm(self, metadata):
        """drl_controller should specify algorithm."""
        assert 'algorithm' in metadata['drl_controller']
        assert isinstance(metadata['drl_controller']['algorithm'], str)

    def test_drl_controller_has_policy_architecture(self, metadata):
        """drl_controller should specify policy_architecture."""
        assert 'policy_architecture' in metadata['drl_controller']

    def test_drl_controller_has_gamma(self, metadata):
        """drl_controller should specify gamma (discount factor)."""
        assert 'gamma' in metadata['drl_controller']


class TestGateSetSection:
    """Tests for the gate_set_and_constraints section."""

    def test_has_gate_set(self, metadata):
        """gate_set_and_constraints should have gate_set."""
        gate_section = metadata['gate_set_and_constraints']
        assert 'gate_set' in gate_section
        assert isinstance(gate_section['gate_set'], list)

    def test_gate_set_not_empty(self, metadata):
        """gate_set should not be empty."""
        gate_set = metadata['gate_set_and_constraints']['gate_set']
        assert len(gate_set) > 0

    def test_has_max_depth(self, metadata):
        """gate_set_and_constraints should have max_depth."""
        assert 'max_depth' in metadata['gate_set_and_constraints']


class TestInnerLoopSection:
    """Tests for the inner_loop_optimization section."""

    def test_has_loss_function(self, metadata):
        """inner_loop_optimization should specify loss function."""
        inner_loop = metadata['inner_loop_optimization']
        assert 'loss' in inner_loop


class TestRewardDesignSection:
    """Tests for the reward_design section."""

    def test_has_primary_reward(self, metadata):
        """reward_design should specify primary_reward."""
        assert 'primary_reward' in metadata['reward_design']


class TestNotesSection:
    """Tests for the notes section."""

    def test_notes_is_list(self, metadata):
        """notes should be a list."""
        assert isinstance(metadata['notes'], list)

    def test_notes_documents_missing_values(self, metadata):
        """notes should document any values not found in paper."""
        notes = metadata['notes']
        # Should mention learning rate is not specified
        notes_text = ' '.join(notes).lower()
        assert 'not specified' in notes_text or 'null' in notes_text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
