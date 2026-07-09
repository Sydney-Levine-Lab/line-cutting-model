"""
Bootstrap significance tests for simulation-vs-experiment comparisons.

Usage:
    from bootstrap_significance import (
        bootstrap_over_participants,
        bootstrap_delta_r2,
        bootstrap_pairwise,
    )

    # CIs on Pearson r
    ci_df, boot_rs = bootstrap_over_participants(dm["joe_univ"], datasets["joe_univ"])

    # CIs on ΔR² (the main one for papers)
    ci_df, boot_dr2 = bootstrap_delta_r2(dm["joe_univ"], datasets["joe_univ"])

    # Pairwise: is depth1 > depth0?
    result = bootstrap_pairwise(dm["joe_univ"], datasets["joe_univ"], 'depth1', 'depth0')
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

UAW_COL = "univ_aggregate_welfare"
DEFAULT_BASELINE_COLS = ("aggregate_welfare", "ordinal_harm_blind", "inequality")


# =====================================================================
# Helper: recompute stimulus means from resampled participants
# =====================================================================

def _resample_means(long, participants, pcol, jcol, scol, rng):
    """Resample participants with replacement, return stimulus means."""
    resampled = rng.choice(participants, size=len(participants), replace=True)
    counts = pd.Series(resampled).value_counts()
    frames = [long[long[pcol] == p] for p in counts.index
              for _ in range(counts[p])]
    boot_long = pd.concat(frames, ignore_index=True)
    return boot_long.groupby(scol)[jcol].mean()


# =====================================================================
# 1. Bootstrap CIs on Pearson r (resampling participants)
# =====================================================================

def bootstrap_over_participants(design_mats, experiment, n_boot=1000, seed=42):
    """
    Bootstrap over participants to get CIs on each model's r with human data.

    Returns (summary_df, boot_rs_dict)
    """
    if experiment.long_df is None:
        raise ValueError(f"Need long_df for '{experiment.name}'")

    long = experiment.long_df.copy()
    pcol = experiment.participant_col
    jcol = experiment.judgment_col
    scol = experiment.stimulus_col
    participants = long[pcol].unique()
    rng = np.random.RandomState(seed)

    # Precompute U_AW per stimulus for each model
    uaw_per_tag = {}
    for tag, dm in design_mats.items():
        uaw = dm[["stimulus", UAW_COL]].drop_duplicates("stimulus")
        uaw_per_tag[tag] = uaw.set_index("stimulus")[UAW_COL]

    tags = list(design_mats.keys())
    boot_rs = {tag: [] for tag in tags}

    for _ in range(n_boot):
        boot_means = _resample_means(long, participants, pcol, jcol, scol, rng)
        for tag in tags:
            uaw = uaw_per_tag[tag]
            common = boot_means.index.intersection(uaw.index)
            bm = boot_means[common]
            ua = uaw[common]
            valid = np.isfinite(bm) & np.isfinite(ua)
            if valid.sum() >= 3:
                r, _ = stats.pearsonr(bm[valid], ua[valid])
                boot_rs[tag].append(r)
            else:
                boot_rs[tag].append(np.nan)

    rows = []
    for tag in tags:
        rs = np.array(boot_rs[tag])
        rs = rs[~np.isnan(rs)]
        rows.append({
            "tag": tag, "metric": "r",
            "mean": np.mean(rs), "median": np.median(rs),
            "ci_lo": np.percentile(rs, 2.5),
            "ci_hi": np.percentile(rs, 97.5),
            "se": np.std(rs),
        })
    return pd.DataFrame(rows), boot_rs


# =====================================================================
# 2. Bootstrap CIs on ΔR² (resampling participants)
# =====================================================================

def bootstrap_delta_r2(design_mats, experiment,
                       baseline_cols=DEFAULT_BASELINE_COLS,
                       n_boot=1000, seed=42):
    """
    Bootstrap over participants to get CIs on each model's ΔR².

    For each resample:
      - Resample participants → recompute stimulus-level DV means
      - For each model, rebuild design matrix with resampled DV
      - Fit baseline regression (outcome vars only)
      - Fit full regression (outcome vars + U_AW)
      - ΔR² = R²_full - R²_baseline

    Returns (summary_df, boot_dr2_dict)
    """
    if experiment.long_df is None:
        raise ValueError(f"Need long_df for '{experiment.name}'")

    long = experiment.long_df.copy()
    pcol = experiment.participant_col
    jcol = experiment.judgment_col
    scol = experiment.stimulus_col
    dv_col = experiment.dv_col
    participants = long[pcol].unique()
    rng = np.random.RandomState(seed)
    bl = list(baseline_cols)

    # Precompute: for each model, get the non-DV columns of the design matrix
    # (stimulus, U_AW, baseline predictors) — these don't change across resamples
    dm_fixed = {}
    for tag, dm in design_mats.items():
        keep_cols = ["stimulus"] + [c for c in bl if c in dm.columns] + [UAW_COL]
        keep_cols = [c for c in keep_cols if c in dm.columns]
        dm_fixed[tag] = dm[keep_cols].drop_duplicates("stimulus").set_index("stimulus")

    tags = list(design_mats.keys())
    boot_dr2 = {tag: [] for tag in tags}
    boot_r = {tag: [] for tag in tags}

    for _ in range(n_boot):
        boot_means = _resample_means(long, participants, pcol, jcol, scol, rng)

        for tag in tags:
            fixed = dm_fixed[tag]
            common = boot_means.index.intersection(fixed.index)

            if len(common) < len(bl) + 3:
                boot_dr2[tag].append(np.nan)
                boot_r[tag].append(np.nan)
                continue

            df = fixed.loc[common].copy()
            df[dv_col] = boot_means[common]
            df = df.dropna()

            if len(df) < len(bl) + 3:
                boot_dr2[tag].append(np.nan)
                boot_r[tag].append(np.nan)
                continue

            try:
                y = df[dv_col]
                bl_present = [c for c in bl if c in df.columns]

                X_base = sm.add_constant(df[bl_present])
                base = sm.OLS(y, X_base).fit()

                X_full = sm.add_constant(df[bl_present + [UAW_COL]])
                full = sm.OLS(y, X_full).fit()

                dr2 = full.rsquared - base.rsquared
                r, _ = stats.pearsonr(df[UAW_COL].values, y.values)

                boot_dr2[tag].append(dr2)
                boot_r[tag].append(r)
            except Exception:
                boot_dr2[tag].append(np.nan)
                boot_r[tag].append(np.nan)

    rows = []
    for tag in tags:
        dr2s = np.array(boot_dr2[tag])
        dr2s = dr2s[~np.isnan(dr2s)]
        rs = np.array(boot_r[tag])
        rs = rs[~np.isnan(rs)]
        rows.append({
            "tag": tag,
            "dr2_mean": np.mean(dr2s) if len(dr2s) > 0 else np.nan,
            "dr2_median": np.median(dr2s) if len(dr2s) > 0 else np.nan,
            "dr2_ci_lo": np.percentile(dr2s, 2.5) if len(dr2s) > 0 else np.nan,
            "dr2_ci_hi": np.percentile(dr2s, 97.5) if len(dr2s) > 0 else np.nan,
            "dr2_se": np.std(dr2s) if len(dr2s) > 0 else np.nan,
            "r_mean": np.mean(rs) if len(rs) > 0 else np.nan,
            "r_ci_lo": np.percentile(rs, 2.5) if len(rs) > 0 else np.nan,
            "r_ci_hi": np.percentile(rs, 97.5) if len(rs) > 0 else np.nan,
        })
    return pd.DataFrame(rows), boot_dr2


# =====================================================================
# 3. Pairwise comparison (on r or ΔR²)
# =====================================================================

def bootstrap_pairwise(design_mats, experiment, tag_a, tag_b,
                       metric="r", n_boot=1000, seed=42):
    """
    Test whether model A has significantly higher r (or ΔR²) than model B.

    metric: "r" or "delta_r2"

    Returns dict with diff_mean, ci_lo, ci_hi, p_value, significant
    p_value = proportion of resamples where A ≤ B (low = A wins)
    """
    sub_mats = {tag_a: design_mats[tag_a], tag_b: design_mats[tag_b]}

    if metric == "r":
        _, boots = bootstrap_over_participants(sub_mats, experiment,
                                               n_boot=n_boot, seed=seed)
    else:
        _, boots = bootstrap_delta_r2(sub_mats, experiment,
                                      n_boot=n_boot, seed=seed)

    rs_a = np.array(boots[tag_a])
    rs_b = np.array(boots[tag_b])
    valid = ~(np.isnan(rs_a) | np.isnan(rs_b))
    rs_a, rs_b = rs_a[valid], rs_b[valid]
    diffs = rs_a - rs_b

    return {
        "tag_a": tag_a, "tag_b": tag_b, "metric": metric,
        "diff_mean": np.mean(diffs),
        "diff_median": np.median(diffs),
        "ci_lo": np.percentile(diffs, 2.5),
        "ci_hi": np.percentile(diffs, 97.5),
        "p_value": np.mean(diffs <= 0),
        "significant": np.mean(diffs <= 0) < 0.05,
        "n_boot_valid": len(diffs),
    }


# =====================================================================
# 4. Directional hypothesis tests
# =====================================================================

def test_hypotheses(design_mats, experiment, metric="r", n_boot=1000, seed=42):
    """
    Run the key directional hypothesis tests.
    
    Tests whether adding complexity improves fit:
      depth0 → depth1 (does reasoning help?)
      depth1 → depth2_v3 (does forward simulation help?)
      depth1 → l2_d1 (does deeper ToM help?)
      l2_d1 → l3_d1 (does even deeper ToM help?)
      depth1 → l2_d2 (does both help?)
    """
    contrasts = [
        ('depth1',    'depth0',    'L1 heuristic vs L0 (does reasoning help?)'),
        ('depth2_v3', 'depth1',    'Real L1 vs L1 heuristic (does depth help?)'),
        ('l2_d1',    'depth1',     'L2 heuristic vs L1 heuristic (does ToM help?)'),
        ('l3_d1',    'l2_d1',      'L3 heuristic vs L2 heuristic (more ToM?)'),
        ('l2_d2',    'depth1',     'Real L2 vs L1 heuristic (both ToM+depth?)'),
    ]

    results = []
    for better, base, desc in contrasts:
        if better not in design_mats or base not in design_mats:
            print(f"  Skipping: {desc} (missing model)")
            continue
        res = bootstrap_pairwise(design_mats, experiment, better, base,
                                  metric=metric, n_boot=n_boot, seed=seed)
        res['description'] = desc
        results.append(res)

        sig = ("***" if res['p_value'] < 0.001 else
               "**" if res['p_value'] < 0.01 else
               "*" if res['p_value'] < 0.05 else "n.s.")
        arrow = "↑" if res['diff_mean'] > 0 else "↓"
        print(f"{desc}")
        print(f"  Δ{metric} = {res['diff_mean']:+.4f}  "
              f"[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
              f"p={res['p_value']:.3f} {sig} {arrow}")
        print()

    return results


# =====================================================================
# 5. Full pairwise comparison table
# =====================================================================

def pairwise_comparison_table(design_mats, experiment,
                              tags=None, metric="r",
                              n_boot=1000, seed=42):
    """
    Compare all pairs of models. Returns DataFrame.
    """
    if tags is None:
        tags = list(design_mats.keys())

    if metric == "r":
        _, boots = bootstrap_over_participants(
            design_mats, experiment, n_boot=n_boot, seed=seed)
    else:
        _, boots = bootstrap_delta_r2(
            design_mats, experiment, n_boot=n_boot, seed=seed)

    rows = []
    for i, tag_a in enumerate(tags):
        for tag_b in tags[i+1:]:
            rs_a = np.array(boots[tag_a])
            rs_b = np.array(boots[tag_b])
            valid = ~(np.isnan(rs_a) | np.isnan(rs_b))
            diffs = rs_a[valid] - rs_b[valid]
            rows.append({
                "model_a": tag_a, "model_b": tag_b,
                "diff_mean": np.mean(diffs),
                "ci_lo": np.percentile(diffs, 2.5),
                "ci_hi": np.percentile(diffs, 97.5),
                "p_a_gt_b": np.mean(diffs <= 0),
                "significant": bool(np.percentile(diffs, 2.5) > 0 or
                                    np.percentile(diffs, 97.5) < 0),
            })
    return pd.DataFrame(rows)
