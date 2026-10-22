"""Robustness of the universalizability fit to how runs are aggregated into a per-map score.
Rules: mean_excl_stuck (current) | median_excl_stuck | mean_incl_stuck (unfinished agents scored at TIME_MAX).
Recomputes U_AW from agent completion times in each model's processed files, then reruns evaluate_fits +
participant bootstrap (all 39 stimuli). Output: logan_pipeline/results/stats_choice_robustness.csv"""
import glob, numpy as np, pandas as pd
from pathlib import Path
from new_comparison_utils import load_experiments, load_simulations, load_outcome_metrics, build_all_design_matrices, evaluate_fits
from bootstrap_significance import bootstrap_over_participants
from build_utils import compute_metrics
RUNS = {"L0": "L0_ghost_flat", "L1 heuristic": "L1h_ghost_flat", "L2 heuristic": "L2h_ghost_flat", "real L1": "realL1_fixed_flat"}
SIM = Path("../data/simulations"); TIME_MAX = 1000
line = pd.read_csv("../data/scenarios/completion_times.csv"); line = line[line.condition == "line"].set_index("map")
AG = [f"agent_{i}" for i in range(1, 9)]

def per_map_scores(label, rule):
    out = {}
    for f in sorted((SIM / label / "processed").glob("*.csv")):
        d = pd.read_csv(f); m = f.stem; lt = line.loc[m, AG].to_numpy(float)
        vals = []
        for _, r in d.iterrows():
            w = r[AG].to_numpy(float)
            if (w <= 0).any():
                if rule != "mean_incl_stuck": continue
                w = np.where(w <= 0, TIME_MAX, w)
            vals.append(compute_metrics(world=w, line=lt)["aggregate_welfare"])
        out[m] = np.median(vals) if rule == "median_excl_stuck" else np.mean(vals)
    return pd.Series(out)

ds = load_experiments(joe_moral_agg="joe_moral_agg.csv", joe_moral_long="joe_moral_long.csv",
                      joe_univ_agg="joe_univ_agg.csv", joe_univ_long="joe_univ_long.csv")
exp = ds["joe_univ"]; out_df = load_outcome_metrics()
sims0 = load_simulations(RUNS, recompute=False)
rows = []
for rule in ["mean_excl_stuck", "median_excl_stuck", "mean_incl_stuck"]:
    sims = {}
    for tag, label in RUNS.items():
        s = sims0[tag].copy(); sc = per_map_scores(label, rule)
        s["univ_aggregate_welfare"] = s["map"].map(sc).to_numpy(); sims[tag] = s
    dm = build_all_design_matrices(sims, out_df, exp)
    ev = evaluate_fits(dm, exp).set_index("tag")
    _, boot = bootstrap_over_participants(dm, exp, n_boot=2000, seed=42)
    b = {t: np.asarray(boot[t]) for t in RUNS}
    def ci(x): return np.percentile(x, [2.5, 97.5])
    for t in RUNS:
        rows.append(dict(rule=rule, model=t, r=ev.loc[t, "r_U_AW"], r_lo=ci(b[t])[0], r_hi=ci(b[t])[1]))
    for a, c in [("L1 heuristic", "L0"), ("real L1", "L1 heuristic"), ("L2 heuristic", "L1 heuristic")]:
        d = b[a] - b[c]
        rows.append(dict(rule=rule, model=f"{a} − {c}", r=ev.loc[a, "r_U_AW"] - ev.loc[c, "r_U_AW"], r_lo=ci(d)[0], r_hi=ci(d)[1]))
res = pd.DataFrame(rows); res.to_csv("logan_pipeline/results/stats_choice_robustness.csv", index=False)
print(res.pivot(index="model", columns="rule", values=["r", "r_lo", "r_hi"]).round(3).to_string())
