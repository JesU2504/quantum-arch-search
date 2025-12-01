import pytest
from experiments import config
from experiments.config import get_metric_key_for_task_mode


def test_state_preparation_metric():
    assert get_metric_key_for_task_mode('state_preparation') == 'state_preparation_metric'


def test_unitary_preparation_metric():
    assert get_metric_key_for_task_mode('unitary_preparation') == 'unitary_preparation_metric'


def test_default_task_mode_returns_correct_metric():
    # When called without an argument, should use the module-level TASK_MODE
    expected_metric = config._METRIC_FOR_TASK_MODE[config.TASK_MODE]
    assert get_metric_key_for_task_mode() == expected_metric


def test_invalid_task_mode_raises_value_error():
    with pytest.raises(ValueError):
        get_metric_key_for_task_mode('invalid_mode')
