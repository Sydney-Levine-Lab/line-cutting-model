"""Per-model universalizability fit (Pearson r, participant bootstrap) and Δr vs L1 shallow,
all 39 vs restricted 20, 50-run sims. Same functions/seeds as the n50 notebook (section 4/9)."""
import copy, numpy as np, pandas as pd
from new_comparison_utils import load_experiments, load_simulations, load_outcome_metrics, build_all_design_matrices, evaluate_fits
from bootstrap_significance import bootstrap_over_participants
from subset_utils import load_stimulus_flag, restrict_all
import fast_bootstrap; fast_bootstrap.install()
RUNS = {"L0": "L0_ghost_flat", "L1 heuristic": "L1h_ghost_flat", "L2 heuristic": "L2h_ghost_flat", "real L1": "realL1_fixed_flat", "real L2": "realL2_fixed_flat"}
ds = load_experiments(joe_moral_agg="joe_moral_agg.csv", joe_moral_long="joe_moral_long.csv",
                      joe_univ_agg="joe_univ_agg.csv", joe_univ_long="joe_univ_long.csv")
ds = {"joe_univ": ds["joe_univ"]}
sims = load_simulations(RUNS, recompute=False); out_df = load_outcome_metrics()
dm = {"joe_univ": build_all_design_matrices(sims, out_df, ds["joe_univ"])}
rows = []
for sub in ("all", "dopt"):
    d, m = (ds, dm) if sub == "all" else restrict_all(copy.deepcopy(ds), copy.deepcopy(dm), load_stimulus_flag(), verbose=False)
    _, boot = bootstrap_over_participants(m["joe_univ"], d["joe_univ"], n_boot=2000, seed=42)
    ev = evaluate_fits(m["joe_univ"], d["joe_univ"]).set_index("tag")
    ref = np.asarray(boot["L1 heuristic"])
    for t in RUNS:
        b = np.asarray(boot[t]); dd = b - ref
        rows.append(dict(subset=sub, model=t, r=ev.loc[t, "r_U_AW"], r_lo=np.percentile(b, 2.5), r_hi=np.percentile(b, 97.5),
                         d_vs_L1h=ev.loc[t, "r_U_AW"] - ev.loc["L1 heuristic", "r_U_AW"],
                         d_lo=np.percentile(dd, 2.5), d_hi=np.percentile(dd, 97.5)))
out = pd.DataFrame(rows); out.to_csv("logan_pipeline/results/n50_univ_r.csv", index=False); print(out.round(3).to_string())
