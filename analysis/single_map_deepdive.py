"""
single_map_deepdive.py
======================

For one map (e.g. yes_line_8), produce:
  - visualization of the map with agent positions
  - first N decisions side-by-side (real_L1 vs L1h)
  - per-decision candidate evaluations from the trajectory log
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from map_analysis import load_map, render_image


DECISION_RE = re.compile(
    r'\[decision\]\s+t=(\d+)\s+idx=(\d+)\s+k=(\d+)\s+'
    r'pos=\(([0-9-]+),([0-9-]+)\)\s+'
    r'chose=(\S+)\s+tag=(\w+)'
    r'(?:\s+best=([-\d.]+)\s+gap=([-\d.NaN]+))?'
)
CAND_RE = re.compile(
    r'\[candidates\]\s+t=(\d+)\s+idx=(\d+)\s+k=(\d+)\s+(.+)'
)


def parse_trajectory_with_candidates(log_path):
    """Parse a trajectory log, returning both decisions AND their preceding
    candidate evaluations.
    """
    rows = []
    with open(log_path) as f:
        last_candidates = None
        for line in f:
            cm = CAND_RE.match(line)
            if cm:
                t = int(cm.group(1)); idx = int(cm.group(2)); k = int(cm.group(3))
                rest = cm.group(4)
                # Parse "action_name(agentN)=value action_name(agentN)=value ..."
                pairs = {}
                for am in re.finditer(r'(\S+?)=([-\d.Inf]+)', rest):
                    action = am.group(1)
                    try:
                        val = float(am.group(2))
                    except ValueError:
                        val = float('-inf') if 'Inf' in am.group(2) else float('nan')
                    pairs[action] = val
                last_candidates = (t, idx, k, pairs)
                continue
            dm = DECISION_RE.match(line)
            if dm:
                t = int(dm.group(1)); idx = int(dm.group(2)); k = int(dm.group(3))
                x, y = int(dm.group(4)), int(dm.group(5))
                chose = dm.group(6); tag = dm.group(7)
                best = float(dm.group(8)) if dm.group(8) else np.nan
                gap_s = dm.group(9)
                gap = float(gap_s) if gap_s and gap_s != 'NaN' else np.nan
                cands = None
                if last_candidates and last_candidates[:3] == (t, idx, k):
                    cands = last_candidates[3]
                rows.append({
                    't': t, 'idx': idx, 'k': k,
                    'pos': (x, y), 'chose': chose, 'tag': tag,
                    'best': best, 'gap': gap, 'candidates': cands,
                })
    return rows


def compare_first_n_decisions(real_log, heur_log, n=20, show_candidates=True):
    """Side-by-side comparison of first N decisions in two logs."""
    r_rows = parse_trajectory_with_candidates(real_log)
    h_rows = parse_trajectory_with_candidates(heur_log)
    r_by_key = {(d['t'], d['idx'], d['k']): d for d in r_rows}
    h_by_key = {(d['t'], d['idx'], d['k']): d for d in h_rows}
    keys = sorted(set(r_by_key.keys()) & set(h_by_key.keys()))[:n]
    
    output = []
    first_disagreement_seen = False
    for key in keys:
        r = r_by_key[key]
        h = h_by_key[key]
        t, idx, k = key
        agreed = r['chose'] == h['chose']
        marker = '   ' if agreed else '** '
        if not agreed and not first_disagreement_seen:
            marker = '>> '
            first_disagreement_seen = True
        line = (f"{marker}t={t:2d} idx={idx} k={k} pos={r['pos']} | "
                f"real_L1: {r['chose']:20s} tag={r['tag']:8s} gap={r['gap']:.3f} | "
                f"L1_heur: {h['chose']:20s}")
        output.append(line)
        if not agreed and show_candidates and r.get('candidates'):
            cand_str = ', '.join(f"{a}={v:.2f}" if v != float('-inf') else f"{a}=-inf"
                                  for a, v in sorted(r['candidates'].items(),
                                                      key=lambda x: -x[1]))
            output.append(f"    real_L1 candidates: {cand_str}")
            output.append(f"    L1h chose {h['chose']} → real_L1 valued it as "
                          f"{r['candidates'].get(h['chose'], 'N/A')}")
    return '\n'.join(output)


def find_first_divergence_with_candidates(real_log, heur_log):
    """Find the first (t, idx, k) where they diverge. Return full info."""
    r_rows = parse_trajectory_with_candidates(real_log)
    h_rows = parse_trajectory_with_candidates(heur_log)
    r_by_key = {(d['t'], d['idx'], d['k']): d for d in r_rows}
    h_by_key = {(d['t'], d['idx'], d['k']): d for d in h_rows}
    keys = sorted(set(r_by_key.keys()) & set(h_by_key.keys()))
    for key in keys:
        if r_by_key[key]['chose'] != h_by_key[key]['chose']:
            return {'real_L1': r_by_key[key], 'L1_heur': h_by_key[key]}
    return None


def visualize_map_with_divergence(map_obj, divergence, save_path=None, title=None):
    """Render map with annotation at the position where models first diverged."""
    fig, ax = plt.subplots(figsize=(0.5*map_obj.n_cols+1, 0.5*map_obj.n_rows+1))
    render_image(map_obj, ax=ax, title=title)
    if divergence:
        r = divergence['real_L1']
        x, y = r['pos']
        ax.scatter(x, y, c='yellow', s=300, marker='*', 
                    edgecolor='black', linewidth=1.2, zorder=5)
        ax.annotate(f"First divergence:\n"
                    f"agent {r['k']} at t={r['t']}\n"
                    f"real_L1: {r['chose']}\n"
                    f"L1h: {divergence['L1_heur']['chose']}",
                    xy=(x, y), xytext=(x+1, y+0.5),
                    fontsize=8, color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax
