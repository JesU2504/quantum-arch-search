#!/usr/bin/env python3
"""
Metrics computation module for DRL vs EA comparison.

This module reads JSONL logs (or multiple JSON files), validates them against
the schema, and computes per-run metrics including final fidelity, max fidelity,
evaluations-to-thresholds, gate_count/depth summaries, and wall-clock time.

Usage:
    python -m comparison.analysis.compute_metrics --input "logs/*.jsonl" --out results/

Functions:
    load_logs(paths): Load logs from file paths (JSON or JSONL)
    validate_logs(logs): Validate logs against schema
    aggregate_metrics(logs): Compute aggregated metrics
    save_summary(metrics, out_path): Save summary to JSON and CSV
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

# Optional: import jsonschema if available
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def load_schema():
    """Load the log schema from the comparison/logs directory."""
    schema_path = Path(__file__).parent.parent / 'logs' / 'schema.json'
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            return json.load(f)
    return None


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


def validate_logs(logs, schema=None):
    """
    Validate logs against the JSON schema.

    Args:
        logs: List of log entry dictionaries
        schema: Optional schema dict (loads default if None)

    Returns:
        tuple: (valid_logs, errors) where errors is list of (index, error_msg)
    """
    if schema is None:
        schema = load_schema()

    if schema is None or not HAS_JSONSCHEMA:
        # If no schema or jsonschema not available, return all logs as valid
        return logs, []

    valid_logs = []
    errors = []

    for i, log in enumerate(logs):
        try:
            jsonschema.validate(log, schema)
            valid_logs.append(log)
        except jsonschema.ValidationError as e:
            errors.append((i, str(e.message)))

    return valid_logs, errors


def compute_per_run_metrics(logs):
    """
    Compute metrics for each unique run.

    Args:
        logs: List of validated log entries

    Returns:
        dict: Metrics grouped by run identifier (method + seed)
    """
    runs: dict[str, dict[str, Any]] = {}

    for log in logs:
        method = log.get('method', 'unknown')
        seed = log.get('seed', 0)
        run_key = f"{method}_seed{seed}"

        if run_key not in runs:
            runs[run_key] = {
                'method': method,
                'seed': seed,
                'entries': [],
                'fidelities': [],
                'gate_counts': [],
                'depths': [],
                'wall_times': [],
            }

        runs[run_key]['entries'].append(log)
        if 'best_fidelity' in log:
            runs[run_key]['fidelities'].append(log['best_fidelity'])
        if 'gate_count' in log:
            runs[run_key]['gate_counts'].append(log['gate_count'])
        if 'circuit_depth' in log:
            runs[run_key]['depths'].append(log['circuit_depth'])
        if 'wall_time_s' in log:
            runs[run_key]['wall_times'].append(log['wall_time_s'])

    # Compute per-run summary metrics
    for run_key, run_data in runs.items():
        fids = run_data['fidelities']
        run_data['final_fidelity'] = fids[-1] if fids else None
        run_data['max_fidelity'] = max(fids) if fids else None
        run_data['mean_fidelity'] = sum(fids) / len(fids) if fids else None
        run_data['total_evals'] = len(run_data['entries'])

        # Evaluations to reach thresholds
        thresholds = [0.9, 0.95, 0.99, 0.999]
        evals_to_threshold = {}
        for thresh in thresholds:
            evals_to_threshold[f'evals_to_{thresh}'] = None
            for i, f in enumerate(fids):
                if f >= thresh:
                    evals_to_threshold[f'evals_to_{thresh}'] = i + 1
                    break
        run_data.update(evals_to_threshold)

        # Gate count and depth summaries
        gc = run_data['gate_counts']
        dp = run_data['depths']
        run_data['final_gate_count'] = gc[-1] if gc else None
        run_data['min_gate_count'] = min(gc) if gc else None
        run_data['final_depth'] = dp[-1] if dp else None
        run_data['min_depth'] = min(dp) if dp else None

        # Wall clock
        wt = run_data['wall_times']
        run_data['total_wall_time_s'] = sum(wt) if wt else None
        run_data['final_wall_time_s'] = wt[-1] if wt else None

        # Clean up intermediate lists
        del run_data['fidelities']
        del run_data['gate_counts']
        del run_data['depths']
        del run_data['wall_times']

    return runs


def aggregate_metrics(logs):
    """
    Compute aggregated metrics across all logs.

    Args:
        logs: List of validated log entries

    Returns:
        dict: Aggregated metrics including per-run and cross-run statistics
    """
    per_run = compute_per_run_metrics(logs)

    # Group by method
    methods: dict[str, dict[str, Any]] = {}
    for run_key, run_data in per_run.items():
        method = run_data['method']
        if method not in methods:
            methods[method] = {
                'runs': [],
                'max_fidelities': [],
                'final_fidelities': [],
                'total_evals': [],
            }
        methods[method]['runs'].append(run_data)
        if run_data['max_fidelity'] is not None:
            methods[method]['max_fidelities'].append(run_data['max_fidelity'])
        if run_data['final_fidelity'] is not None:
            methods[method]['final_fidelities'].append(run_data['final_fidelity'])
        methods[method]['total_evals'].append(run_data['total_evals'])

    # Compute cross-run statistics
    aggregated = {
        'per_run': per_run,
        'by_method': {},
        'total_logs': len(logs),
        'total_runs': len(per_run),
    }

    for method, method_data in methods.items():
        n_runs = len(method_data['runs'])
        max_fids = method_data['max_fidelities']
        final_fids = method_data['final_fidelities']

        aggregated['by_method'][method] = {
            'n_runs': n_runs,
            'mean_max_fidelity': sum(max_fids) / len(max_fids) if max_fids else None,
            'std_max_fidelity': _std(max_fids) if len(max_fids) > 1 else None,
            'mean_final_fidelity': sum(final_fids) / len(final_fids) if final_fids else None,
            'std_final_fidelity': _std(final_fids) if len(final_fids) > 1 else None,
            'mean_total_evals': sum(method_data['total_evals']) / n_runs if n_runs else None,
        }

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
    """
    out_path = Path(out_path)
    if out_path.suffix:
        # If extension provided, use parent dir
        out_dir = out_path.parent
        base_name = out_path.stem
    else:
        out_dir = out_path
        base_name = 'metrics_summary'

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = out_dir / f'{base_name}.json'
    # Convert metrics to JSON-serializable format
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
        description='Compute metrics from experiment logs for DRL vs EA comparison.'
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
        help='Validate logs against schema and report errors'
    )
    args = parser.parse_args()

    # Load logs
    logs = load_logs(args.input)
    if not logs:
        print(f"No logs found matching pattern: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(logs)} log entries")

    # Validate if requested
    if args.validate:
        valid_logs, errors = validate_logs(logs)
        if errors:
            print(f"Validation errors ({len(errors)}):")
            for idx, err in errors[:10]:  # Show first 10 errors
                print(f"  Entry {idx}: {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print("All logs validated successfully")
        logs = valid_logs

    # Compute and save metrics
    metrics = aggregate_metrics(logs)
    json_path, csv_path = save_summary(metrics, args.out)

    print(f"Saved JSON summary to: {json_path}")
    print(f"Saved CSV per-run metrics to: {csv_path}")

    # Print summary
    print("\n--- Summary ---")
    print(f"Total logs: {metrics['total_logs']}")
    print(f"Total runs: {metrics['total_runs']}")
    for method, stats in metrics.get('by_method', {}).items():
        print(f"\nMethod: {method}")
        print(f"  Runs: {stats['n_runs']}")
        if stats['mean_max_fidelity'] is not None:
            print(f"  Mean max fidelity: {stats['mean_max_fidelity']:.4f}")
        if stats['std_max_fidelity'] is not None:
            print(f"  Std max fidelity: {stats['std_max_fidelity']:.4f}")


if __name__ == '__main__':
    main()
