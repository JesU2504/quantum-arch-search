# Experiments

This directory contains experiment configuration and training scripts for quantum architecture search.

## Metric Key Helper

The `get_metric_key_for_task_mode` helper function returns the evaluation metric key for a given task mode. Use this function instead of hard-coding metric names to ensure consistency across the codebase.

```python
from experiments.config import get_metric_key_for_task_mode

metric_key = get_metric_key_for_task_mode()
```
