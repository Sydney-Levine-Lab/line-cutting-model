"""Where do the model differences live? (n=50, poster pipeline)
1. per-stimulus contribution to the fit difference between two models, in R² units:
   (res_B² − res_A²)/SStot, so that the contributions sum to R²_A − R²_B.
   - universalizability (Joe): y ~ U_AW
   - moral US (Joe) and moral 20-country (Logan pooled means): y ~ outcomes + U_AW  (poster ΔR² model)
2. leave-one-stimulus-out and leave-one-map-out point estimates of Δr (univ) for the two contrasts
3. map disagreement ranking: |U_A − U_B| / run SE, with restricted-set flag
Outputs in logan_pipeline/results/: subset_per_stimulus.csv, subset_loo_univ.csv, subset_map_disagreement.csv"""
import numpy as np, pandas as pd
from new_comparison_utils import (load_experiments, load_simulations, load_outcome_metrics,
                                  build_all_design_matrices, DEFAULT_BASELINE_COLS, UAW_COL)
from logan_individual import load_logan_individual
from subset_utils import load_stimulus_flag
RUNS = {"L0": "L0_ghost_flat", "L1 heuristic": "L1h_ghost_flat", "L2 heuristic": "L2h_ghost_flat", "real L1": "realL1_fixed_flat"}
CONTRASTS = [("L1 heuristic", "L0"), ("L1 heuristic", "real L1"), ("L2 heuristic", "L1 heuristic")]
keep = load_stimulus_flag()
ds = load_experiments(joe_moral_agg="joe_moral_agg.csv", joe_moral_long="joe_moral_long.csv",
                      joe_univ_agg="joe_univ_agg.csv", joe_univ_long="joe_univ_long.csv")
pooled, _ = load_logan_individual(); ds["logan_moral"] = pooled
sims = load_simulations(RUNS, recompute=False); out_df = load_outcome_metrics()
dm = {k: build_all_design_matrices(sims, out_df, e) for k, e in ds.items()}

def ols_resid(df, y, xcols):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in xcols]); yv = df[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None); return yv - X @ beta, yv

rows = []
for key, xextra, label in [("joe_univ", [], "univ (US)"), ("joe_moral", list(DEFAULT_BASELINE_COLS), "moral US"), ("logan_moral", list(DEFAULT_BASELINE_COLS), "moral 20 countries")]:
    dv = ds[key].dv_col
    res = {}
    for tag in RUNS:
        d = dm[key][tag].sort_values("stimulus").reset_index(drop=True)
        r, yv = ols_resid(d, dv, xextra + [UAW_COL]); res[tag] = (d, r, yv)
    for a, b in CONTRASTS:
        da, ra, yv = res[a]; _, rb, _ = res[b]; sst = ((yv - yv.mean()) ** 2).sum()
        for i, st in enumerate(da["stimulus"]):
            rows.append(dict(outcome=label, contrast=f"{a} − {b}", stimulus=st, map=da["map"].iloc[i],
                             retained=st in keep, contrib_R2=(rb[i] ** 2 - ra[i] ** 2) / sst))
per = pd.DataFrame(rows); per.to_csv("logan_pipeline/results/subset_per_stimulus.csv", index=False)
summ = per.groupby(["outcome", "contrast", "retained"]).contrib_R2.sum().unstack("retained")
summ["total"] = summ.sum(axis=1); summ["share_dropped"] = summ[False] / summ["total"]
print("\nΔR² decomposition (A − B), sum over retained 20 / dropped 19:\n", summ.round(3).to_string())

# leave-one-out on the univ Pearson-r contrasts (point estimates)
d = {t: dm["joe_univ"][t].sort_values("stimulus").reset_index(drop=True) for t in RUNS}
dv = ds["joe_univ"].dv_col
def dr(mask, a, b):
    return (np.corrcoef(d[a][UAW_COL][mask], d[a][dv][mask])[0, 1] - np.corrcoef(d[b][UAW_COL][mask], d[b][dv][mask])[0, 1])
base = d["L0"]; lo = []
full = {c: dr(np.ones(len(base), bool), *c) for c in CONTRASTS[:2]}
for unit, col in [("stimulus", "stimulus"), ("map", "map")]:
    for v in base[col].unique():
        m = (base[col] != v).to_numpy()
        lo.append({"unit": unit, "removed": v, **{f"Δr {a}−{b}": dr(m, a, b) - full[(a, b)] for a, b in CONTRASTS[:2]}})
lo = pd.DataFrame(lo); lo.to_csv("logan_pipeline/results/subset_loo_univ.csv", index=False)
print("\nFull-sample Δr:", {f"{a}−{b}": round(v, 3) for (a, b), v in full.items()})
for c in [f"Δr {a}−{b}" for a, b in CONTRASTS[:2]]:
    print(f"\nLeave-one-out change in {c} (most influential):")
    print(lo.reindex(lo[c].abs().sort_values(ascending=False).index).head(6)[["unit", "removed", c]].round(3).to_string(index=False))

# map disagreement
S = {t: pd.read_csv(f"../data/simulations/{l}/summary_universalization_metrics.csv").set_index("map") for t, l in RUNS.items()}
lk = pd.read_csv("../data/scenarios/scenario_lookup.csv"); map_in = lk.groupby("map")["in_d_optimal"].any()
md = pd.DataFrame(index=S["L0"].index)
for a, b in CONTRASTS:
    dd = S[a][UAW_COL] - S[b][UAW_COL]; se = np.sqrt(S[a][f"{UAW_COL}_sd"] ** 2 / S[a].n_valid + S[b][f"{UAW_COL}_sd"] ** 2 / S[b].n_valid)
    md[f"d {a}−{b}"] = dd; md[f"z {a}−{b}"] = dd / se
md["in_restricted"] = map_in.reindex(md.index); md["n_stimuli"] = lk.groupby("map").size().reindex(md.index)
md = md.sort_values("z L1 heuristic−L0", key=np.abs, ascending=False); md.to_csv("logan_pipeline/results/subset_map_disagreement.csv")
print("\nMap disagreement (z = score difference / run SE):\n", md.round(2).to_string())
