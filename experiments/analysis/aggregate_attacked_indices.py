#!/usr/bin/env python3
"""
Scan a results directory for JSON files containing 'attacked_indices' and
produce a histogram (counts per gate index). This helps visualize whether
attacks concentrate on early gates.

Usage:
  python experiments/analysis/aggregate_attacked_indices.py --results-root results/your_run

Outputs:
  - Prints histogram to stdout
  - Writes CSV to results_root/attacked_indices_hist.csv
"""
from pathlib import Path
import argparse
import json
from collections import Counter


def find_attacked_indices_in_obj(obj):
    """Recursively search object for 'attacked_indices' keys and yield lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'attacked_indices' and isinstance(v, list):
                yield v
            else:
                yield from find_attacked_indices_in_obj(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from find_attacked_indices_in_obj(item)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', type=str, required=True)
    p.add_argument('--out-csv', type=str, default=None)
    args = p.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise SystemExit(f"Results root not found: {root}")

    counter = Counter()
    total_found = 0
    for path in root.rglob('*.json'):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for lst in find_attacked_indices_in_obj(data):
            for idx in lst:
                counter[int(idx)] += 1
                total_found += 1

    if total_found == 0:
        print("No attacked_indices found in JSON files under", root)
        return

    print(f"Found {total_found} attacked indices across JSON files under {root}")
    print("Index,Count")
    for idx, cnt in sorted(counter.items()):
        print(f"{idx},{cnt}")

    out_csv = args.out_csv or (root / 'attacked_indices_hist.csv')
    with open(out_csv, 'w') as f:
        f.write('index,count\n')
        for idx, cnt in sorted(counter.items()):
            f.write(f"{idx},{cnt}\n")
    print('Wrote histogram to', out_csv)


if __name__ == '__main__':
    main()
