"""
cross_map_analysis.py
=====================

Combine trajectory analysis with structural map features.

Use:
    from cross_map_analysis import build_full_table, identify_first_divergence

    # Build per-map table with outcome, trajectory, and structural features
    df = build_full_table(diff, sims, map_dir='maps/')

    # For one map, find when real_L1 and L1h first diverge
    first_d = identify_first_divergence(dec_real, dec_heur, 'yes_line_8', run=1)
"""

import os
import numpy as np
import pandas as pd
from map_analysis import load_map, structural_features


def _macro_type(name):
    name = str(name)
    if 'no_line' in name: return 'no_line'
    if 'yes_line' in name: return 'yes_line'
    return 'maybe/other'


def per_map_outcomes(sims_real, sims_heur, metric='univ_aggregate_welfare'):
    """Aggregate outcomes per map.

    sims_real / sims_heur: summary DataFrames (e.g. sims['real L1']) with 
    columns 'map', metric, metric+'_sd'.
    """
    r = sims_real[['map', metric, f'{metric}_sd']].rename(
        columns={metric: 'metric_real', f'{metric}_sd': 'sd_real'}
    )
    h = sims_heur[['map', metric, f'{metric}_sd']].rename(
        columns={metric: 'metric_heur', f'{metric}_sd': 'sd_heur'}
    )
    out = r.merge(h, on='map')
    out['gap'] = out['metric_real'] - out['metric_heur']
    return out


def per_map_traj_metrics(diff_df):
    """Trajectory-level metrics per map."""
    rows = []
    for m, sub in diff_df.groupby('map'):
        n_total = len(sub)
        n_disagree = (~sub['agreed']).sum()
        n_fb = (sub['tag_real_L1'] == 'FALLBACK').sum()
        nofb = sub[sub['tag_real_L1'] == 'OK']
        n_nofb_disagree = (~nofb['agreed']).sum()
        # t=1 only (clean comparison)
        t1 = sub[sub['t'] == 1]
        t1_nofb = t1[t1['tag_real_L1'] == 'OK']
        n_t1_disagree = (~t1_nofb['agreed']).sum() if len(t1_nofb) else 0
        rows.append({
            'map': m,
            'n_decisions': n_total,
            'pct_disagree_all': round(100 * n_disagree / max(n_total, 1), 1),
            'pct_fallback': round(100 * n_fb / max(n_total, 1), 1),
            'pct_disagree_nofb': round(100 * n_nofb_disagree / max(len(nofb), 1), 1),
            'n_t1': len(t1),
            'pct_disagree_t1_nofb': round(100 * n_t1_disagree / max(len(t1_nofb), 1), 1),
        })
    return pd.DataFrame(rows)


def per_map_structural(map_dir, map_names=None):
    """Compute structural features for each PDDL map in map_dir."""
    rows = []
    for fn in sorted(os.listdir(map_dir)):
        if not fn.endswith('.pddl'): continue
        if fn.startswith('debug'): continue
        name = fn[:-5]  # strip .pddl
        if map_names is not None and name not in map_names:
            continue
        try:
            m = load_map(f"{map_dir}/{fn}")
            feats = structural_features(m)
            feats['map'] = name
            feats['n_agents'] = len(m.agents)
            feats['n_wells'] = len(m.wells)
            feats['n_tanks'] = len(m.tanks)
            rows.append(feats)
        except Exception as e:
            print(f"  skip {name}: {e}")
    return pd.DataFrame(rows)


def build_full_table(diff_df, sims, map_dir, 
                     metric='univ_aggregate_welfare'):
    """Build the full per-map analysis table.
    
    Returns DataFrame with columns from:
      - per-map outcomes (real vs heur U_AW, gap, SD)
      - trajectory metrics (disagreement, fallback)
      - structural features (walls, corridors, distance)
    """
    out = per_map_outcomes(sims['real L1'], sims['L1 heuristic'], metric)
    out = out.merge(per_map_traj_metrics(diff_df), on='map', how='outer')
    struct = per_map_structural(map_dir, map_names=out['map'].tolist())
    out = out.merge(struct, on='map', how='left')
    out['macro_type'] = out['map'].apply(_macro_type)
    out = out.sort_values('gap', ascending=True).reset_index(drop=True)
    return out


def identify_first_divergence(dec_real, dec_heur, map_name, run=1):
    """For one (map, run), find first decision where real_L1 and L1h chose differently.
    Returns a dict with details, or None if no disagreement.
    """
    r = dec_real[(dec_real['map'] == map_name) & (dec_real['run'] == run)].copy()
    h = dec_heur[(dec_heur['map'] == map_name) & (dec_heur['run'] == run)].copy()
    merged = r.merge(h, on=['map', 'run', 't', 'idx', 'k'],
                     suffixes=('_real', '_heur'))
    merged = merged.sort_values(['t', 'idx'])
    merged['agreed'] = merged['chose_real'] == merged['chose_heur']

    disagrees = merged[~merged['agreed']]
    if len(disagrees) == 0:
        return None
    first = disagrees.iloc[0]
    return {
        't': int(first['t']),
        'idx': int(first['idx']),
        'k': int(first['k']),
        'pos': first['pos_real'],
        'chose_real_L1': first['chose_real'],
        'chose_L1_heur': first['chose_heur'],
        'tag_real_L1': first['tag_real'],
        'best_value': first['best_real'],
        'gap': first['gap_real'],
        'n_disagreements_total': len(disagrees),
        'n_total_in_run': len(merged),
    }


def compare_maps_summary(full_table, top_n=8):
    """Print top-N divergent maps and their key features."""
    cols = ['map', 'macro_type', 'gap', 'sd_real',
            'pct_fallback', 'pct_disagree_t1_nofb',
            'wall_density', 'n_corridors', 'n_adj_agent_pairs',
            'mean_dist_to_water']
    available = [c for c in cols if c in full_table.columns]
    print("Most divergent maps (largest negative gap = real_L1 worse):")
    print(full_table[available].head(top_n).to_string(index=False))
    print()
    print("Least divergent (similar performance or real_L1 better):")
    print(full_table[available].tail(top_n).to_string(index=False))


def correlation_with_gap(full_table, candidate_features=None):
    """For each candidate feature, correlate with the gap."""
    if candidate_features is None:
        candidate_features = ['wall_density', 'n_corridors', 'n_deadends',
                               'n_adj_agent_pairs', 'mean_dist_to_water',
                               'pct_fallback', 'pct_disagree_t1_nofb',
                               'sd_real']
    rows = []
    for f in candidate_features:
        if f not in full_table.columns: continue
        sub = full_table[['gap', f]].dropna()
        # drop infs
        sub = sub[~np.isinf(sub[f])]
        if len(sub) < 3: continue
        r = sub.corr().loc['gap', f]
        rows.append({'feature': f, 'r_with_gap': round(r, 3), 'n': len(sub)})
    return pd.DataFrame(rows).sort_values('r_with_gap')
