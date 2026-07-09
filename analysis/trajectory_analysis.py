"""
trajectory_analysis.py
======================

Parse and analyze TRAJECTORY_LEVEL=summary logs from main10_d.jl.

Each log line looks like:
    [decision] t=15 idx=3 k=7 pos=(5,8) chose=up(agent7) tag=OK best=-19.0000 gap=1.0000
    [candidates] t=15 idx=3 k=7 left(agent7)=-17.0000 wait(agent7)=-18.0000 ... best=up(agent7)

`tag` is one of HEUR, OK, FALLBACK, EMPTY, SOLE.

Usage:
    decisions = parse_run_dir('data/simulations/real_L1_traj_<ts>/raw/trajectories')
    decisions_h = parse_run_dir('data/simulations/L1_heuristic_traj_<ts>/raw/trajectories')
    diff = compare_models(decisions, decisions_h)
    summary = aggregate_per_map(diff)
"""

import os
import re
import pandas as pd
import numpy as np
from glob import glob

DECISION_RE = re.compile(
    r'\[decision\]\s+t=(\d+)\s+idx=(\d+)\s+k=(\d+)\s+'
    r'pos=\(([0-9-]+),([0-9-]+)\)\s+'
    r'chose=(\S+)\s+tag=(\w+)'
    r'(?:\s+best=([-\d.]+)\s+gap=([-\d.NaN]+))?'
)


def parse_log(path):
    """Parse a single trajectory log file. Returns list of dicts.

    Filename: trajectory_<map>_run<N>.log -> we extract map and run.
    """
    fn = os.path.basename(path)
    m = re.match(r'trajectory_(.+)_run(\d+)\.log', fn)
    if not m:
        return []
    map_name, run = m.group(1), int(m.group(2))

    rows = []
    with open(path) as f:
        for line in f:
            m = DECISION_RE.match(line)
            if not m:
                continue
            t, idx, k, x, y, chose, tag, best, gap = m.groups()
            rows.append({
                'map': map_name,
                'run': run,
                't': int(t),
                'idx': int(idx),
                'k': int(k),
                'pos': f'({x},{y})',
                'chose': chose,
                'tag': tag,
                'best': float(best) if best else np.nan,
                'gap': float(gap) if gap and gap != 'NaN' else np.nan,
            })
    return rows


def parse_run_dir(traj_dir):
    """Parse all trajectory logs in a directory. Returns long-form DataFrame."""
    if not os.path.isdir(traj_dir):
        raise ValueError(f"Not a directory: {traj_dir}")
    all_rows = []
    files = sorted(glob(os.path.join(traj_dir, 'trajectory_*.log')))
    for fp in files:
        rows = parse_log(fp)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    if not df.empty:
        n_maps = df['map'].nunique()
        n_decisions = len(df)
        print(f"  parsed {len(files)} files, {n_decisions} decisions, "
              f"{n_maps} maps from {os.path.basename(traj_dir)}")
    return df


def compare_models(decisions_a, decisions_b, tag_a='real_L1', tag_b='L1_heur'):
    """
    Match decisions on (map, run, t, idx, k) and compare chosen actions.
    Returns merged DataFrame with columns:
        map, run, t, idx, k,
        chose_a, chose_b, tag_a, tag_b, gap_a (only real_L1 has gap),
        agreed (bool)
    """
    a = decisions_a.rename(columns={
        'chose': f'chose_{tag_a}',
        'tag': f'tag_{tag_a}',
        'best': f'best_{tag_a}',
        'gap': f'gap_{tag_a}',
    })
    b = decisions_b.rename(columns={
        'chose': f'chose_{tag_b}',
        'tag': f'tag_{tag_b}',
    })[['map', 'run', 't', 'idx', 'k', f'chose_{tag_b}', f'tag_{tag_b}']]

    merged = a.merge(b, on=['map', 'run', 't', 'idx', 'k'], how='inner')
    merged['agreed'] = (merged[f'chose_{tag_a}'] == merged[f'chose_{tag_b}'])
    return merged


def aggregate_per_map(diff_df, tag_a='real_L1', gap_threshold=1.0):
    """
    Per-map summary: how often models disagree, what fraction are close calls.
    """
    out = []
    for m, sub in diff_df.groupby('map'):
        n_total = len(sub)
        n_disagree = (~sub['agreed']).sum()
        n_close = ((~sub['agreed']) & (sub[f'gap_{tag_a}'] < gap_threshold)).sum()
        n_fallback = (sub[f'tag_{tag_a}'] == 'FALLBACK').sum()
        out.append({
            'map': m,
            'n_decisions': n_total,
            'pct_disagree': 100 * n_disagree / max(n_total, 1),
            'pct_close_of_disagree': 100 * n_close / max(n_disagree, 1),
            'pct_fallback': 100 * n_fallback / max(n_total, 1),
            'mean_gap_disagree': sub.loc[~sub['agreed'], f'gap_{tag_a}'].mean(),
        })
    return pd.DataFrame(out).sort_values('pct_disagree', ascending=False)


def overall_summary(diff_df, tag_a='real_L1', gap_threshold=1.0):
    """One-line summary across all decisions."""
    n_total = len(diff_df)
    n_disagree = (~diff_df['agreed']).sum()
    n_close = ((~diff_df['agreed']) & (diff_df[f'gap_{tag_a}'] < gap_threshold)).sum()
    n_fallback = (diff_df[f'tag_{tag_a}'] == 'FALLBACK').sum()
    print(f"Overall ({n_total} decisions):")
    print(f"  disagreement:        {n_disagree:>6d} ({100*n_disagree/n_total:.1f}%)")
    print(f"  close-call disagree: {n_close:>6d} ({100*n_close/max(n_disagree,1):.1f}% of disagreements)")
    print(f"  fallback:            {n_fallback:>6d} ({100*n_fallback/n_total:.1f}%)")
