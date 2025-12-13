#!/usr/bin/env python3
import argparse
import csv
import json
import os
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def count_ops_from_cirq_json(path):
    try:
        j = json.load(open(path))
    except Exception:
        return None
    cnt = 0
    for m in j.get('moments', []):
        cnt += len(m.get('operations', []))
    return cnt


def load_gate_counts(compare_dir):
    # map (run_idx, circuit_type) -> gate_count
    mapping = {}
    run_dirs = sorted([p for p in Path(compare_dir).glob('run_*') if p.is_dir()])
    for run_dir in run_dirs:
        try:
            run_idx = int(run_dir.name.split('_')[-1])
        except Exception:
            continue
        for ctype in ['vanilla', 'robust', 'quantumnas']:
            path = run_dir / f'circuit_{ctype}.json'
            if path.exists():
                mapping[(run_idx, ctype)] = count_ops_from_cirq_json(path)
    return mapping


def read_attacked_samples(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                run_idx = int(r['run_idx'])
                ctype = r['circuit_type']
                val = float(r['attacked_fidelity'])
            except Exception:
                continue
            rows.append({'run_idx': run_idx, 'circuit_type': ctype, 'attacked': val})
    return rows


def fit_regression(xs, dummies):
    # xs is array shape (n,) gate_count; dummies is dict of dummy arrays shape (n,)
    n = xs.shape[0]
    Xcols = [np.ones(n), xs]
    for k in sorted(dummies.keys()):
        Xcols.append(dummies[k])
    X = np.vstack(Xcols).T
    y = attacked_arr
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X.dot(coef)
    ssr = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else float('nan')
    return coef, r2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results_dir')
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    compare_dir = results_dir / 'compare'
    attacked_csv = compare_dir / 'attacked_fidelity_samples.csv'
    if not attacked_csv.exists():
        print('No attacked_fidelity_samples.csv in', compare_dir)
        raise SystemExit(1)

    mapping = load_gate_counts(compare_dir)
    rows = read_attacked_samples(attacked_csv)

    gate_counts = []
    types = []
    attacked_vals = []
    missing = 0
    for r in rows:
        k = (r['run_idx'], r['circuit_type'])
        gc = mapping.get(k)
        if gc is None or gc == 0:
            # try fallback: use aggregated mean from experiment_summary.json
            gc = None
            missing += 1
        gate_counts.append(gc if gc is not None else math.nan)
        types.append(r['circuit_type'])
        attacked_vals.append(r['attacked'])

    # Convert arrays and drop nan gate_counts
    gate_arr = np.array(gate_counts, dtype=float)
    attacked_arr = np.array(attacked_vals, dtype=float)
    types_arr = np.array(types)
    mask = ~np.isnan(gate_arr)
    print(f"Total samples: {len(attacked_arr)}, missing gate counts: {np.sum(np.isnan(gate_arr))}")

    gate_arr = gate_arr[mask]
    attacked_arr = attacked_arr[mask]
    types_arr = types_arr[mask]

    # Build dummy variables (robust, quantumnas), with vanilla as baseline
    robust_dummy = (types_arr == 'robust').astype(float)
    qnas_dummy = (types_arr == 'quantumnas').astype(float)

    # Fit linear model: attacked ~ gate_count + robust + quantumnas
    X = np.vstack([gate_arr, robust_dummy, qnas_dummy]).T
    # Add intercept
    Xmat = np.hstack([np.ones((X.shape[0], 1)), X])
    coef, *_ = np.linalg.lstsq(Xmat, attacked_arr, rcond=None)
    intercept = coef[0]
    coef_gate = coef[1]
    coef_robust = coef[2]
    coef_qnas = coef[3]
    yhat = Xmat.dot(coef)
    ssr = np.sum((attacked_arr - yhat) ** 2)
    sst = np.sum((attacked_arr - attacked_arr.mean()) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else float('nan')

    print('\nLinear regression: attacked_fidelity = intercept + coef_gate * gate_count + coef_robust*I_robust + coef_qnas*I_qnas')
    print(f'intercept={intercept:.6f}, coef_gate={coef_gate:.6f}, coef_robust={coef_robust:.6f}, coef_qnas={coef_qnas:.6f}, R2={r2:.4f}')

    # Plot attacked_fidelity vs gate_count colored by type
    out_dir = Path(args.out_dir) if args.out_dir else compare_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,4))
    for label, col in [('vanilla','C0'), ('robust','C1'), ('quantumnas','C2')]:
        sel = types_arr == label
        plt.scatter(gate_arr[sel], attacked_arr[sel], label=label, alpha=0.6, color=col, s=10)
    # plot fitted line vs gate_count (for vanilla baseline)
    xs = np.linspace(np.nanmin(gate_arr), np.nanmax(gate_arr), 100)
    ys = intercept + coef_gate * xs  # baseline
    plt.plot(xs, ys, color='k', linestyle='--', label='fit (vanilla baseline)')
    plt.xlabel('Gate count')
    plt.ylabel('Attacked fidelity')
    plt.legend()
    plt.tight_layout()
    p1 = out_dir / 'attacked_vs_gatecount.png'
    plt.savefig(p1, dpi=150)
    print('Wrote', p1)

    # Compute loss_per_gate = (1 - attacked)/gate_count and boxplot by type
    loss_per_gate = (1.0 - attacked_arr) / gate_arr
    vals_by_type = [loss_per_gate[types_arr == t] for t in ['vanilla','robust','quantumnas']]

    plt.figure(figsize=(6,4))
    plt.boxplot(vals_by_type, labels=['vanilla','robust','quantumnas'], showfliers=False)
    plt.ylabel('Loss per gate = (1 - attacked_fidelity)/gate_count')
    plt.tight_layout()
    p2 = out_dir / 'loss_per_gate_boxplot.png'
    plt.savefig(p2, dpi=150)
    print('Wrote', p2)

    # Print simple summary stats
    for t, arr in zip(['vanilla','robust','quantumnas'], vals_by_type):
        if len(arr) == 0:
            print(f'{t}: no samples')
            continue
        print(f"{t}: loss_per_gate mean={np.mean(arr):.6f}, std={np.std(arr, ddof=1):.6f}, n={len(arr)}")

    print('\nDone')
