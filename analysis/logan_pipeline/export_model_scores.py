"""
Export per-stimulus universalization scores (U_AW) for each simulation model, in
the shape Logan's judgment model consumes: one row per stimulus, one column per
model. This is the only thing that crosses from the simulation repo into the
statistical pipeline.

    stimulus, map, condition, in_d_optimal, joe, L0, L1h, L2h, realL1, realL2

`joe` is Joe Kwon's 2023 score, copied from Logan's per-stimulus-measures.csv
(column univ.agg.welfare) when that file is given, so the reproduction baseline
and the substitutions live in one table.

Bad/badder stimuli share a map with their 1cut sibling and therefore share every
model's U_AW; only the outcome measures (joined later, from Joe's table) differ.

Usage (from the simulation repo root):
    python analysis/logan_pipeline/export_model_scores.py \
        --out analysis/logan_pipeline/model_scores.csv \
        [--joe-measures ../logan-line-cutting/data/per-stimulus-measures.csv]
"""

import argparse
from pathlib import Path

import pandas as pd

# Final poster configuration (poster_figures_FINAL_v4.ipynb, RUNS).
RUNS = {
    "L0": "L0_ghost_flat",
    "L1h": "L1h_ghost_flat",
    "L2h": "L2h_ghost_flat",
    "realL1": "realL1_fixed_flat",
    "realL2": "realL2_fixed_flat",
}
SCORE_COL = "univ_aggregate_welfare"


def main(sim_root: Path, lookup: Path, out: Path, joe_measures: Path | None,
         summary_name: str = "summary_universalization_metrics.csv") -> None:
    lk = pd.read_csv(lookup)
    table = lk[["preferred_label", "map", "condition", "in_d_optimal"]].rename(
        columns={"preferred_label": "stimulus"})

    if joe_measures is not None:
        joe = pd.read_csv(joe_measures)
        joe = joe.merge(lk[["cogsci 23", "preferred_label"]], left_on="context",
                        right_on="cogsci 23", how="left")
        assert joe["preferred_label"].notna().all(), "unmapped Joe contexts"
        table = table.merge(joe[["preferred_label", "univ.agg.welfare"]]
                            .rename(columns={"preferred_label": "stimulus",
                                             "univ.agg.welfare": "joe"}),
                            on="stimulus", how="left")

    for tag, run in RUNS.items():
        s = pd.read_csv(sim_root / run / summary_name)
        assert s["map"].is_unique
        if "n_stuck" in s and s["n_stuck"].sum():
            print(f"  note: {tag} has {int(s['n_stuck'].sum())} stuck runs excluded from its means")
        table = table.merge(s[["map", SCORE_COL]].rename(columns={SCORE_COL: tag}),
                            on="map", how="left")

    missing = table[list(RUNS)].isna().any(axis=1)
    assert not missing.any(), f"stimuli without a model score: {table.loc[missing, 'stimulus'].tolist()}"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"wrote {out}: {len(table)} stimuli, {table['map'].nunique()} maps, "
          f"{int(table['in_d_optimal'].sum())} in D-optimal set")
    print(table[["joe", *RUNS]].corr().round(3).to_string() if "joe" in table else table[list(RUNS)].corr().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", type=Path, default=Path("data/simulations"))
    ap.add_argument("--lookup", type=Path, default=Path("data/scenarios/scenario_lookup.csv"))
    ap.add_argument("--out", type=Path, default=Path("analysis/logan_pipeline/model_scores.csv"))
    ap.add_argument("--joe-measures", type=Path, default=None)
    ap.add_argument("--summary-name", default="summary_universalization_metrics.csv")
    a = ap.parse_args()
    main(a.sim_root, a.lookup, a.out, a.joe_measures, a.summary_name)
