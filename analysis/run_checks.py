"""
run_checks.py
=============

Small helpers to sanity-check the flattened cluster runs before you trust
any model comparison. Import from the notebook:

    from run_checks import check_runs, check_variance

    check_runs({"L0": "L0_fixedorder_zap_flat",
                "L1 heuristic": "L1h_fixedorder_zap_flat",
                "real L1": "realL1_fixedorder_zap_flat"})
"""

from pathlib import Path
import pandas as pd
import numpy as np

SIM_ROOT = Path("../data/simulations")
AGENT_COLS = [f"agent_{i}" for i in range(1, 9)]


def check_runs(runs, sim_root=SIM_ROOT, expected_runs=5, expected_maps=28):
    """
    For each run_label, report per-map run counts and flag anything that
    isn't `expected_runs`. Unequal n across maps biases model comparison:
    a map with 3 runs has a noisier mean than one with 5, and the noise
    is not random with respect to map difficulty (hard maps are the ones
    that get preempted).
    """
    rows = []
    for tag, label in runs.items():
        raw = Path(sim_root) / label / "raw"
        if not raw.is_dir():
            print(f"  {tag:16s} MISSING: {raw}")
            continue
        for csv in sorted(raw.glob("*.csv")):
            df = pd.read_csv(csv)
            stuck = int((df[AGENT_COLS] <= 0).any(axis=1).sum()) if \
                all(c in df.columns for c in AGENT_COLS) else 0
            rows.append({"model": tag, "map": csv.stem,
                         "n_runs": len(df), "n_stuck": stuck,
                         "n_valid": len(df) - stuck})

    out = pd.DataFrame(rows)
    if out.empty:
        print("No data found.")
        return out

    print("Per-model coverage:")
    for tag, sub in out.groupby("model", sort=False):
        n_maps = len(sub)
        bad = sub[sub["n_runs"] != expected_runs]
        stuck = sub["n_stuck"].sum()
        flag = "" if (n_maps == expected_maps and bad.empty) else "   <-- CHECK"
        print(f"  {tag:16s} {n_maps:2d}/{expected_maps} maps, "
              f"{sub['n_runs'].sum():3d} runs, {stuck} stuck{flag}")
        for _, r in bad.iterrows():
            print(f"      {r['map']:16s} {r['n_runs']}/{expected_runs} runs")

    # Maps where models disagree on n — the dangerous case for delta-R^2
    piv = out.pivot_table(index="map", columns="model", values="n_valid")
    ragged = piv[piv.nunique(axis=1) > 1]
    if len(ragged):
        print("\nMaps with UNEQUAL valid-run counts across models:")
        print(ragged.to_string())
        print("  Model means on these maps rest on different samples.")
        print("  Consider dropping them, or resubmit the array to fill.")

    return out


def check_variance(runs, sim_root=SIM_ROOT, metric="mean"):
    """
    Per-map SD of a per-run metric, across runs, per model.

    With fixed play order the only remaining randomness is Boltzmann
    tie-breaking at temp 1e-4. Any substantial SD is therefore generated
    by the *map*, not the noise parameter — congested maps force genuine
    ties, and each coin-flip cascades.

    Returns a wide table: rows = maps, cols = models, values = SD of
    per-run mean fill time. Sort by any column to see the congestion
    gradient.
    """
    frames = []
    for tag, label in runs.items():
        proc = Path(sim_root) / label / "processed"
        if not proc.is_dir():
            print(f"  {tag}: no processed/ dir — run get_universalization_summary(recompute=True) first")
            continue
        for csv in sorted(proc.glob("*.csv")):
            df = pd.read_csv(csv)
            if metric not in df.columns:
                continue
            valid = df[~df.get("stuck", False).astype(bool)] if "stuck" in df else df
            if len(valid) < 2:
                continue
            frames.append({"model": tag, "map": csv.stem,
                           "sd": valid[metric].std(ddof=1),
                           "mean": valid[metric].mean(),
                           "n": len(valid)})

    out = pd.DataFrame(frames)
    if out.empty:
        return out
    wide = out.pivot_table(index="map", columns="model", values="sd")
    return wide.sort_values(wide.columns[0], ascending=False)
