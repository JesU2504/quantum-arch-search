#!/usr/bin/env python3
"""
Classification metrics computation module for DRL vs EA comparison.

This module reads JSONL logs from classification experiments, validates them,
and computes classification-specific metrics including:
- Final validation/test accuracy
- Best accuracy achieved
- Evaluations-to-threshold accuracies (70%, 80%, 90%)
- Complexity metrics (gate_count, circuit_depth)
- Aggregated statistics across seeds

Usage:
    python -m comparison.analysis.compute_classif_metrics --input "logs/*.jsonl" --out results/

Functions:
    load_logs(paths): Load logs from file paths (JSON or JSONL)
    validate_classification_logs(logs): Validate logs have classification fields
    compute_classification_metrics(logs): Compute classification-specific metrics
    save_classification_summary(metrics, out_path): Save JSON and CSV summaries
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Schema fields required for classification logs
REQUIRED_FIELDS = ['eval_id', 'method', 'seed']
CLASSIFICATION_FIELDS = ['train_accuracy', 'test_accuracy']
OPTIONAL_FIELDS = ['gate_count', 'circuit_depth', 'episode', 'generation', 'wall_time_s']


def load_logs(paths):
    """
    Load logs from file paths (JSON or JSONL format).

    Args:
        paths: List of file paths or glob pattern string

    Returns:
        list: List of log entry dictionaries
    """
    if isinstance(paths, str):
        paths = glob.glob(paths, recursive=True)

    logs = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue

        with open(path, 'r') as f:
            if path.suffix == '.jsonl':
                # JSONL format: one JSON object per line
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entry['_source_file'] = str(path)
                            entry['_source_line'] = line_num
                            logs.append(entry)
                        except json.JSONDecodeError:
                            continue
            else:
                # JSON format: array of objects or single object
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            entry['_source_file'] = str(path)
                            logs.append(entry)
                    else:
                        data['_source_file'] = str(path)
                        logs.append(data)
                except json.JSONDecodeError:
                    continue

    return logs


def validate_classification_logs(logs, strict=False):
    """
    Validate logs have required classification fields.

    Args:
        logs: List of log entry dictionaries
        strict: If True, require all classification fields

    Returns:
        tuple: (valid_logs, errors) where errors is list of (index, error_msg)
    """
    valid_logs = []
    errors = []

    for i, log in enumerate(logs):
        missing_required = [f for f in REQUIRED_FIELDS if f not in log]
        if missing_required:
            errors.append((i, f"Missing required fields: {missing_required}"))
            continue

        # Check for at least one accuracy field
        has_accuracy = any(f in log for f in ['train_accuracy', 'test_accuracy', 
                                               'best_accuracy', 'accuracy'])
        if strict and not has_accuracy:
            errors.append((i, "No accuracy field found"))
            continue

        # Validate accuracy ranges
        is_valid = True
        for field in ['train_accuracy', 'test_accuracy', 'best_accuracy', 'accuracy']:
            if field in log:
                val = log[field]
                if val is not None and (val < 0 or val > 1):
                    errors.append((i, f"{field} out of range [0,1]: {val}"))
                    is_valid = False
                    break

        if is_valid:
            valid_logs.append(log)

    return valid_logs, errors


def compute_per_run_classification_metrics(logs):
    """
    Compute classification metrics for each unique run.

    Args:
        logs: List of validated log entries

    Returns:
        dict: Metrics grouped by run identifier (method + seed)
    """
    runs: Dict[str, Dict[str, Any]] = {}

    for log in logs:
        method = log.get('method', 'unknown')
        seed = log.get('seed', 0)
        run_key = f"{method}_seed{seed}"

        if run_key not in runs:
            runs[run_key] = {
                'method': method,
                'seed': seed,
                'entries': [],
                'train_accuracies': [],
                'test_accuracies': [],
                'gate_counts': [],
                'depths': [],
                'wall_times': [],
                'eval_ids': [],
            }

        runs[run_key]['entries'].append(log)

        # Extract eval_id for ordering
        if 'eval_id' in log:
            runs[run_key]['eval_ids'].append(log['eval_id'])
        elif 'episode' in log:
            runs[run_key]['eval_ids'].append(log['episode'])
        elif 'generation' in log:
            runs[run_key]['eval_ids'].append(log['generation'])

        # Extract accuracies
        train_acc = log.get('train_accuracy') or log.get('accuracy')
        if train_acc is not None:
            runs[run_key]['train_accuracies'].append(train_acc)

        test_acc = log.get('test_accuracy') or log.get('validation_accuracy')
        if test_acc is not None:
            runs[run_key]['test_accuracies'].append(test_acc)

        # Extract complexity metrics
        if 'gate_count' in log:
            runs[run_key]['gate_counts'].append(log['gate_count'])
        if 'circuit_depth' in log:
            runs[run_key]['depths'].append(log['circuit_depth'])
        if 'wall_time_s' in log:
            runs[run_key]['wall_times'].append(log['wall_time_s'])

    # Compute per-run summary metrics
    for run_key, run_data in runs.items():
        train_accs = run_data['train_accuracies']
        test_accs = run_data['test_accuracies']

        # Final and best accuracies
        run_data['final_train_accuracy'] = train_accs[-1] if train_accs else None
        run_data['best_train_accuracy'] = max(train_accs) if train_accs else None
        run_data['mean_train_accuracy'] = sum(train_accs) / len(train_accs) if train_accs else None

        run_data['final_test_accuracy'] = test_accs[-1] if test_accs else None
        run_data['best_test_accuracy'] = max(test_accs) if test_accs else None
        run_data['mean_test_accuracy'] = sum(test_accs) / len(test_accs) if test_accs else None

        run_data['total_evals'] = len(run_data['entries'])

        # Evaluations to reach accuracy thresholds
        thresholds = [0.70, 0.80, 0.90, 0.95]
        # Use best accuracy seen so far for threshold computation
        best_so_far = 0.0
        for thresh in thresholds:
            run_data[f'evals_to_{int(thresh*100)}_accuracy'] = None

        accuracies_to_check = test_accs if test_accs else train_accs
        for i, acc in enumerate(accuracies_to_check):
            best_so_far = max(best_so_far, acc)
            for thresh in thresholds:
                key = f'evals_to_{int(thresh*100)}_accuracy'
                if run_data[key] is None and best_so_far >= thresh:
                    run_data[key] = i + 1

        # Complexity metrics
        gc = run_data['gate_counts']
        dp = run_data['depths']
        run_data['final_gate_count'] = gc[-1] if gc else None
        run_data['min_gate_count'] = min(gc) if gc else None
        run_data['mean_gate_count'] = sum(gc) / len(gc) if gc else None

        run_data['final_depth'] = dp[-1] if dp else None
        run_data['min_depth'] = min(dp) if dp else None
        run_data['mean_depth'] = sum(dp) / len(dp) if dp else None

        # Best model complexity (at best test accuracy)
        if test_accs and gc:
            best_idx = test_accs.index(max(test_accs))
            if best_idx < len(gc):
                run_data['best_model_gate_count'] = gc[best_idx]
            if best_idx < len(dp):
                run_data['best_model_depth'] = dp[best_idx]

        # Wall clock
        wt = run_data['wall_times']
        run_data['total_wall_time_s'] = max(wt) if wt else None

        # Clean up intermediate lists
        del run_data['train_accuracies']
        del run_data['test_accuracies']
        del run_data['gate_counts']
        del run_data['depths']
        del run_data['wall_times']
        del run_data['eval_ids']

    return runs


def aggregate_classification_metrics(logs):
    """
    Compute aggregated classification metrics across all logs.

    Args:
        logs: List of validated log entries

    Returns:
        dict: Aggregated metrics including per-run and cross-run statistics
    """
    per_run = compute_per_run_classification_metrics(logs)

    # Group by method
    methods: Dict[str, Dict[str, Any]] = {}
    for run_key, run_data in per_run.items():
        method = run_data['method']
        if method not in methods:
            methods[method] = {
                'runs': [],
                'best_test_accuracies': [],
                'final_test_accuracies': [],
                'best_train_accuracies': [],
                'evals_to_70': [],
                'evals_to_80': [],
                'evals_to_90': [],
                'final_gate_counts': [],
                'min_gate_counts': [],
            }

        methods[method]['runs'].append(run_data)

        if run_data.get('best_test_accuracy') is not None:
            methods[method]['best_test_accuracies'].append(run_data['best_test_accuracy'])
        if run_data.get('final_test_accuracy') is not None:
            methods[method]['final_test_accuracies'].append(run_data['final_test_accuracy'])
        if run_data.get('best_train_accuracy') is not None:
            methods[method]['best_train_accuracies'].append(run_data['best_train_accuracy'])
        if run_data.get('evals_to_70_accuracy') is not None:
            methods[method]['evals_to_70'].append(run_data['evals_to_70_accuracy'])
        if run_data.get('evals_to_80_accuracy') is not None:
            methods[method]['evals_to_80'].append(run_data['evals_to_80_accuracy'])
        if run_data.get('evals_to_90_accuracy') is not None:
            methods[method]['evals_to_90'].append(run_data['evals_to_90_accuracy'])
        if run_data.get('final_gate_count') is not None:
            methods[method]['final_gate_counts'].append(run_data['final_gate_count'])
        if run_data.get('min_gate_count') is not None:
            methods[method]['min_gate_counts'].append(run_data['min_gate_count'])

    # Compute cross-run statistics
    aggregated = {
        'per_run': per_run,
        'by_method': {},
        'total_logs': len(logs),
        'total_runs': len(per_run),
    }

    for method, method_data in methods.items():
        n_runs = len(method_data['runs'])

        aggregated['by_method'][method] = {
            'n_runs': n_runs,

            # Accuracy statistics
            'mean_best_test_accuracy': _mean(method_data['best_test_accuracies']),
            'std_best_test_accuracy': _std(method_data['best_test_accuracies']),
            'mean_final_test_accuracy': _mean(method_data['final_test_accuracies']),
            'std_final_test_accuracy': _std(method_data['final_test_accuracies']),
            'mean_best_train_accuracy': _mean(method_data['best_train_accuracies']),
            'std_best_train_accuracy': _std(method_data['best_train_accuracies']),

            # Evals-to-threshold statistics
            'mean_evals_to_70': _mean(method_data['evals_to_70']),
            'std_evals_to_70': _std(method_data['evals_to_70']),
            'success_rate_70': len(method_data['evals_to_70']) / n_runs if n_runs > 0 else None,

            'mean_evals_to_80': _mean(method_data['evals_to_80']),
            'std_evals_to_80': _std(method_data['evals_to_80']),
            'success_rate_80': len(method_data['evals_to_80']) / n_runs if n_runs > 0 else None,

            'mean_evals_to_90': _mean(method_data['evals_to_90']),
            'std_evals_to_90': _std(method_data['evals_to_90']),
            'success_rate_90': len(method_data['evals_to_90']) / n_runs if n_runs > 0 else None,

            # Complexity statistics
            'mean_final_gate_count': _mean(method_data['final_gate_counts']),
            'std_final_gate_count': _std(method_data['final_gate_counts']),
            'mean_min_gate_count': _mean(method_data['min_gate_counts']),
            'std_min_gate_count': _std(method_data['min_gate_counts']),
        }

    return aggregated


def _mean(values):
    """Compute mean of list, handling empty lists."""
    if not values:
        return None
    return sum(values) / len(values)


def _std(values):
    """Compute sample standard deviation."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def save_classification_summary(metrics, out_path):
    """
    Save classification summary to JSON and CSV files.

    Args:
        metrics: Aggregated metrics dictionary
        out_path: Output directory or file path (without extension)

    Returns:
        tuple: (json_path, csv_path) paths to saved files
    """
    out_path = Path(out_path)
    if out_path.suffix:
        out_dir = out_path.parent
        base_name = out_path.stem
    else:
        out_dir = out_path
        base_name = 'classif_metrics_summary'

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = out_dir / f'{base_name}.json'
    serializable_metrics = _make_serializable(metrics)
    with open(json_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    # Save CSV for per-run metrics
    csv_path = out_dir / f'{base_name}_per_run.csv'
    per_run = metrics.get('per_run', {})
    if per_run:
        # Get all unique keys from runs
        all_keys: Set[str] = set()
        for run_data in per_run.values():
            all_keys.update(k for k in run_data.keys() if k != 'entries')

        fieldnames = ['run_key'] + sorted(all_keys)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for run_key, run_data in per_run.items():
                row = {'run_key': run_key}
                for key in all_keys:
                    val = run_data.get(key, '')
                    row[key] = val if val is not None else ''
                writer.writerow(row)

    # Save by-method summary CSV
    method_csv_path = out_dir / f'{base_name}_by_method.csv'
    by_method = metrics.get('by_method', {})
    if by_method:
        all_keys = set()
        for method_data in by_method.values():
            all_keys.update(method_data.keys())

        fieldnames = ['method'] + sorted(all_keys)
        with open(method_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for method, method_data in by_method.items():
                row = {'method': method}
                for key in all_keys:
                    val = method_data.get(key, '')
                    row[key] = val if val is not None else ''
                writer.writerow(row)

    return json_path, csv_path


def _make_serializable(obj):
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items() if k != 'entries'}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Compute classification metrics from experiment logs for DRL vs EA comparison.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Glob pattern or path to input log files (JSON or JSONL)'
    )
    parser.add_argument(
        '--out', '-o',
        default='.',
        help='Output directory for summary files (default: current directory)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate logs and report errors'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Require all classification fields (stricter validation)'
    )
    args = parser.parse_args()

    # Load logs
    logs = load_logs(args.input)
    if not logs:
        print(f"No logs found matching pattern: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(logs)} log entries")

    # Validate
    valid_logs, errors = validate_classification_logs(logs, strict=args.strict)
    if args.validate and errors:
        print(f"Validation errors ({len(errors)}):")
        for idx, err in errors[:10]:
            print(f"  Entry {idx}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    elif not errors:
        print("All logs validated successfully")

    if not valid_logs:
        print("No valid logs after validation", file=sys.stderr)
        sys.exit(1)

    print(f"Using {len(valid_logs)} valid log entries")

    # Compute and save metrics
    metrics = aggregate_classification_metrics(valid_logs)
    json_path, csv_path = save_classification_summary(metrics, args.out)

    print(f"\nSaved JSON summary to: {json_path}")
    print(f"Saved CSV per-run metrics to: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Total logs: {metrics['total_logs']}")
    print(f"Total runs: {metrics['total_runs']}")

    for method, stats in metrics.get('by_method', {}).items():
        print(f"\n--- Method: {method} ---")
        print(f"  Runs: {stats['n_runs']}")

        if stats.get('mean_best_test_accuracy') is not None:
            std = stats.get('std_best_test_accuracy') or 0
            print(f"  Best test accuracy: {stats['mean_best_test_accuracy']:.4f} ± {std:.4f}")

        if stats.get('mean_final_test_accuracy') is not None:
            std = stats.get('std_final_test_accuracy') or 0
            print(f"  Final test accuracy: {stats['mean_final_test_accuracy']:.4f} ± {std:.4f}")

        if stats.get('success_rate_90') is not None:
            print(f"  Success rate (90%): {stats['success_rate_90']*100:.1f}%")
            if stats.get('mean_evals_to_90') is not None:
                print(f"  Mean evals to 90%: {stats['mean_evals_to_90']:.1f}")

        if stats.get('mean_final_gate_count') is not None:
            std = stats.get('std_final_gate_count') or 0
            print(f"  Final gate count: {stats['mean_final_gate_count']:.1f} ± {std:.1f}")


if __name__ == '__main__':
    main()
