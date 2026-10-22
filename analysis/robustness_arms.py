"""Robustness arms vs baseline. Usage: python robustness_arms.py <arm>   (arm in ARMS)
Per model: correlation of the 28 map scores with the 50-run baseline, stuck runs; then the three poster-pipeline
claims with participant bootstrap: Δr univ (shallow − L0, rollout − shallow), ΔΔR² moral US and 20 countries.
Appends to logan_pipeline/results/robustness_arms.csv. 'baseline5' = first 5 runs (same seeds as the arms)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from new_comparison_utils import load_experiments, load_simulations, load_outcome_metrics, build_all_design_matrices, evaluate_fits, UAW_COL
from bootstrap_significance import bootstrap_over_participants, bootstrap_delta_r2
from logan_individual import load_logan_individual
from build_utils import compute_metrics
import fast_bootstrap; fast_bootstrap.install()
ARMS = {
 "baseline50": {"L0": "L0_ghost_flat", "L1 heuristic": "L1h_ghost_flat", "real L1": "realL1_fixed_flat"},
 "baseline5":  {"L0": "L0_ghost_flat", "L1 heuristic": "L1h_ghost_flat", "real L1": "realL1_fixed_flat"},
 "ghost_T05":  {"L0": "L0_ghost_T05_flat", "L1 heuristic": "L1h_ghost_T05_flat", "real L1": "realL1_fixed_T05_flat"},
 "ghost_rev":  {"L0": "L0_ghost_rev_flat", "L1 heuristic": "L1h_ghost_rev_flat", "real L1": "realL1_fixed_rev_flat"},
 "zap_T05":    {"L0": "L0_zap_T05_flat", "L1 heuristic": "L1h_zap_T05_flat", "real L1": "realL1_zap_T05_flat"},
 "zap":        {"L0": "L0_zap_flat", "L1 heuristic": "L1h_zap_flat", "real L1": "realL1_zap_flat"},
}
arm = sys.argv[1]; NB = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SIM = Path("../data/simulations"); AG = [f"agent_{i}" for i in range(1, 9)]
line = pd.read_csv("../data/scenarios/completion_times.csv"); line = line[line.condition == "line"].set_index("map")

def first5(label):
    rows = []
    for f in sorted((SIM / label / "processed").glob("*.csv")):
        d = pd.read_csv(f); d = d[d.run <= 5]; lt = line.loc[f.stem, AG].to_numpy(float); vals = []
        for _, r in d.iterrows():
            w = r[AG].to_numpy(float)
            if (w <= 0).any(): continue
            vals.append(compute_metrics(world=w, line=lt)["aggregate_welfare"])
        rows.append(dict(map=f.stem, **{UAW_COL: np.mean(vals)}, n_stuck=int((d[AG] <= 0).any(axis=1).sum())))
    return pd.DataFrame(rows)

base = load_simulations(ARMS["baseline50"], recompute=False)
sims = {t: first5(l) for t, l in ARMS["baseline5"].items()} if arm == "baseline5" else load_simulations(ARMS[arm], recompute=(arm not in ("baseline50",)))
ds = load_experiments(joe_moral_agg="joe_moral_agg.csv", joe_moral_long="joe_moral_long.csv", joe_univ_agg="joe_univ_agg.csv", joe_univ_long="joe_univ_long.csv")
pooled, _ = load_logan_individual(); ds["logan_moral"] = pooled
out_df = load_outcome_metrics()
row = {"arm": arm, "n_boot": NB}
for t in ARMS[arm]:
    a = sims[t].set_index("map")[UAW_COL]; b = base[t].set_index("map")[UAW_COL]; j = pd.concat([a, b], axis=1).dropna()
    row[f"corr_{t}"] = np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]; row[f"stuck_{t}"] = int(sims[t]["n_stuck"].sum()) if "n_stuck" in sims[t] else np.nan
dm = {k: build_all_design_matrices(sims, out_df, e) for k, e in ds.items()}
ev = evaluate_fits(dm["joe_univ"], ds["joe_univ"]).set_index("tag")
for t in ARMS[arm]: row[f"r_univ_{t}"] = ev.loc[t, "r_U_AW"]
_, boot = bootstrap_over_participants(dm["joe_univ"], ds["joe_univ"], n_boot=NB, seed=42)
def ci(x): return np.percentile(x, [2.5, 97.5])
for name, (a, b) in {"univ_shallow-L0": ("L1 heuristic", "L0"), "univ_rollout-shallow": ("real L1", "L1 heuristic")}.items():
    d = np.asarray(boot[a]) - np.asarray(boot[b]); row[name] = ev.loc[a, "r_U_AW"] - ev.loc[b, "r_U_AW"]; row[name + "_lo"], row[name + "_hi"] = ci(d)
for key, lab in [("joe_moral", "moralUS"), ("logan_moral", "moral20")]:
    _, bd = bootstrap_delta_r2(dm[key], ds[key], n_boot=NB, seed=42)
    for name, (a, b) in {f"{lab}_shallow-L0": ("L1 heuristic", "L0"), f"{lab}_rollout-shallow": ("real L1", "L1 heuristic")}.items():
        d = np.asarray(bd[a]) - np.asarray(bd[b]); row[name] = d.mean(); row[name + "_lo"], row[name + "_hi"] = ci(d)
out = Path("logan_pipeline/results/robustness_arms.csv")
df = pd.concat([pd.read_csv(out), pd.DataFrame([row])]) if out.exists() else pd.DataFrame([row])
df = df.drop_duplicates(subset="arm", keep="last"); df.to_csv(out, index=False)
print(pd.DataFrame([row]).T.to_string(header=False))
