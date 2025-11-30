#!/usr/bin/env python3
"""
Classification metrics computation module for DRL vs EA comparison.

This module reads JSONL logs (or multiple JSON files), validates them against
the schema, and computes per-run classification metrics including:
- final_val_accuracy, max_val_accuracy
- final_test_accuracy (if present)
- evals_to_thresholds (70%, 80%, 90%)
- num_evals, gate_count/depth summaries

Usage:
    python -m comparison.analysis.compute_classif_metrics \\
        --input "comparison/logs/*.jsonl" \\
        --out comparison/logs/classif_analysis

Functions:
    load_logs(paths): Load logs from file paths (JSON or JSONL)
    compute_classification_metrics(logs): Compute classification-specific metrics
    save_summary(metrics, out_path): Save summary to JSON and CSV
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Optional


def load_logs(paths):
    """
    Load logs from file paths (JSON or JSONL format).

    Args:
        paths: List of file paths or glob pattern string

    Returns:
        list: List of log entry dictionaries
    """
    if isinstance(paths, str):
        paths = glob.glob(paths)

    logs = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue

        with open(path, 'r') as f:
            if path.suffix == '.jsonl':
                # JSONL format: one JSON object per line
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            else:
                # JSON format: array of objects or single object
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        logs.extend(data)
                    else:
                        logs.append(data)
                except json.JSONDecodeError:
                    continue

    return logs


def _get_accuracy(log: dict, key_priority: list) -> Optional[float]:
    """Get accuracy value from log entry using key priority list."""
    for key in key_priority:
        if key in log and log[key] is not None:
            return float(log[key])
    return None


def compute_per_run_classification_metrics(logs, thresholds=None):
    """
    Compute classification metrics for each unique run.

    Args:
        logs: List of validated log entries
        thresholds: List of accuracy thresholds to compute evals_to_threshold
                   Default: [0.70, 0.80, 0.90]

    Returns:
        dict: Metrics grouped by run identifier (method + seed)
    """
    if thresholds is None:
        thresholds = [0.70, 0.80, 0.90]

    runs: dict[str, dict[str, Any]] = {}

    # Keys to try for validation accuracy (in priority order)
    val_acc_keys = ['best_val_accuracy', 'val_accuracy', 'best_fidelity', 'accuracy']
    # Keys to try for test accuracy
    test_acc_keys = ['best_test_accuracy', 'test_accuracy', 'final_test_accuracy']

    for log in logs:
        method = log.get('method', 'unknown')
        seed = log.get('seed', 0)
        run_key = f"{method}_seed{seed}"

        if run_key not in runs:
            runs[run_key] = {
                'method': method,
                'seed': seed,
                'entries': [],
                'val_accuracies': [],
                'test_accuracies': [],
                'gate_counts': [],
                'depths': [],
                'eval_counts': [],
            }

        runs[run_key]['entries'].append(log)

        val_acc = _get_accuracy(log, val_acc_keys)
        if val_acc is not None:
            runs[run_key]['val_accuracies'].append(val_acc)

        test_acc = _get_accuracy(log, test_acc_keys)
        if test_acc is not None:
            runs[run_key]['test_accuracies'].append(test_acc)

        if 'gate_count' in log and log['gate_count'] is not None:
            runs[run_key]['gate_counts'].append(log['gate_count'])
        if 'circuit_depth' in log and log['circuit_depth'] is not None:
            runs[run_key]['depths'].append(log['circuit_depth'])
        if 'cum_eval_count' in log and log['cum_eval_count'] is not None:
            runs[run_key]['eval_counts'].append(log['cum_eval_count'])

    # Compute per-run summary metrics
    for run_key, run_data in runs.items():
        val_accs = run_data['val_accuracies']
        test_accs = run_data['test_accuracies']

        # Validation accuracy metrics
        run_data['final_val_accuracy'] = val_accs[-1] if val_accs else None
        run_data['max_val_accuracy'] = max(val_accs) if val_accs else None
        run_data['mean_val_accuracy'] = sum(val_accs) / len(val_accs) if val_accs else None

        # Test accuracy metrics
        run_data['final_test_accuracy'] = test_accs[-1] if test_accs else None
        run_data['max_test_accuracy'] = max(test_accs) if test_accs else None

        # Number of evaluations
        run_data['num_evals'] = len(run_data['entries'])
        if run_data['eval_counts']:
            run_data['total_cum_evals'] = max(run_data['eval_counts'])

        # Evaluations to reach thresholds (validation accuracy)
        for thresh in thresholds:
            thresh_key = f'evals_to_{int(thresh * 100)}pct'
            run_data[thresh_key] = None
            for i, acc in enumerate(val_accs):
                if acc >= thresh:
                    run_data[thresh_key] = i + 1
                    break

        # Gate count and depth summaries
        gc = run_data['gate_counts']
        dp = run_data['depths']
        run_data['final_gate_count'] = gc[-1] if gc else None
        run_data['min_gate_count'] = min(gc) if gc else None
        run_data['final_depth'] = dp[-1] if dp else None
        run_data['min_depth'] = min(dp) if dp else None

        # Clean up intermediate lists (keep entries for debugging if needed)
        del run_data['val_accuracies']
        del run_data['test_accuracies']
        del run_data['gate_counts']
        del run_data['depths']
        del run_data['eval_counts']

    return runs


def aggregate_classification_metrics(logs, thresholds=None):
    """
    Compute aggregated classification metrics across all logs.

    Args:
        logs: List of validated log entries
        thresholds: List of accuracy thresholds

    Returns:
        dict: Aggregated metrics including per-run and cross-run statistics
    """
    if thresholds is None:
        thresholds = [0.70, 0.80, 0.90]

    per_run = compute_per_run_classification_metrics(logs, thresholds)

    # Group by method
    methods: dict[str, dict[str, Any]] = {}
    for run_key, run_data in per_run.items():
        method = run_data['method']
        if method not in methods:
            methods[method] = {
                'runs': [],
                'max_val_accuracies': [],
                'final_val_accuracies': [],
                'final_test_accuracies': [],
                'num_evals': [],
            }
        methods[method]['runs'].append(run_data)
        if run_data['max_val_accuracy'] is not None:
            methods[method]['max_val_accuracies'].append(run_data['max_val_accuracy'])
        if run_data['final_val_accuracy'] is not None:
            methods[method]['final_val_accuracies'].append(run_data['final_val_accuracy'])
        if run_data['final_test_accuracy'] is not None:
            methods[method]['final_test_accuracies'].append(run_data['final_test_accuracy'])
        methods[method]['num_evals'].append(run_data['num_evals'])

    # Compute cross-run statistics
    aggregated = {
        'per_run': per_run,
        'by_method': {},
        'total_logs': len(logs),
        'total_runs': len(per_run),
        'thresholds_used': thresholds,
    }

    for method, method_data in methods.items():
        n_runs = len(method_data['runs'])
        max_vals = method_data['max_val_accuracies']
        final_vals = method_data['final_val_accuracies']
        final_tests = method_data['final_test_accuracies']

        aggregated['by_method'][method] = {
            'n_runs': n_runs,
            # Validation accuracy stats
            'mean_max_val_accuracy': sum(max_vals) / len(max_vals) if max_vals else None,
            'std_max_val_accuracy': _std(max_vals) if len(max_vals) > 1 else None,
            'mean_final_val_accuracy': sum(final_vals) / len(final_vals) if final_vals else None,
            'std_final_val_accuracy': _std(final_vals) if len(final_vals) > 1 else None,
            # Test accuracy stats
            'mean_final_test_accuracy': sum(final_tests) / len(final_tests) if final_tests else None,
            'std_final_test_accuracy': _std(final_tests) if len(final_tests) > 1 else None,
            # Eval counts
            'mean_num_evals': sum(method_data['num_evals']) / n_runs if n_runs else None,
        }

        # Compute mean evals_to_threshold for each threshold
        for thresh in thresholds:
            thresh_key = f'evals_to_{int(thresh * 100)}pct'
            evals = [r[thresh_key] for r in method_data['runs'] if r.get(thresh_key) is not None]
            aggregated['by_method'][method][f'mean_{thresh_key}'] = (
                sum(evals) / len(evals) if evals else None
            )
            aggregated['by_method'][method][f'n_reached_{int(thresh * 100)}pct'] = len(evals)

    return aggregated


def _std(values):
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def save_summary(metrics, out_path):
    """
    Save summary to JSON and CSV files.

    Args:
        metrics: Aggregated metrics dictionary
        out_path: Output directory or file path (without extension)

    Returns:
        tuple: (json_path, csv_path)
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
        all_keys: set[str] = set()
        for run_data in per_run.values():
            all_keys.update(run_data.keys())
        all_keys.discard('entries')  # Don't include raw entries

        fieldnames = ['run_key'] + sorted(all_keys)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for run_key, run_data in per_run.items():
                row = {'run_key': run_key}
                for key in all_keys:
                    row[key] = run_data.get(key, '')
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
        '--thresholds', '-t',
        type=float,
        nargs='+',
        default=[0.70, 0.80, 0.90],
        help='Accuracy thresholds for evals_to_threshold (default: 0.70 0.80 0.90)'
    )
    args = parser.parse_args()

    # Load logs
    logs = load_logs(args.input)
    if not logs:
        print(f"No logs found matching pattern: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(logs)} log entries")

    # Compute and save metrics
    metrics = aggregate_classification_metrics(logs, thresholds=args.thresholds)
    json_path, csv_path = save_summary(metrics, args.out)

    print(f"Saved JSON summary to: {json_path}")
    print(f"Saved CSV per-run metrics to: {csv_path}")

    # Print summary
    print("\n--- Classification Metrics Summary ---")
    print(f"Total logs: {metrics['total_logs']}")
    print(f"Total runs: {metrics['total_runs']}")
    print(f"Thresholds: {metrics['thresholds_used']}")

    for method, stats in metrics.get('by_method', {}).items():
        print(f"\nMethod: {method}")
        print(f"  Runs: {stats['n_runs']}")
        if stats['mean_max_val_accuracy'] is not None:
            std = stats.get('std_max_val_accuracy', 0) or 0
            print(f"  Mean max val accuracy: {stats['mean_max_val_accuracy']:.4f} ± {std:.4f}")
        if stats['mean_final_val_accuracy'] is not None:
            std = stats.get('std_final_val_accuracy', 0) or 0
            print(f"  Mean final val accuracy: {stats['mean_final_val_accuracy']:.4f} ± {std:.4f}")
        if stats['mean_final_test_accuracy'] is not None:
            std = stats.get('std_final_test_accuracy', 0) or 0
            print(f"  Mean final test accuracy: {stats['mean_final_test_accuracy']:.4f} ± {std:.4f}")

        # Print threshold stats
        for thresh in args.thresholds:
            pct = int(thresh * 100)
            n_reached = stats.get(f'n_reached_{pct}pct', 0)
            mean_evals = stats.get(f'mean_evals_to_{pct}pct')
            if mean_evals is not None:
                print(f"  Reached {pct}%: {n_reached}/{stats['n_runs']} runs, mean evals: {mean_evals:.1f}")


if __name__ == '__main__':
    main()
