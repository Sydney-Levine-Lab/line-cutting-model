"""Simulated worlds per map: universalization score (U_AW, group time saved vs a queue, mean over 50 runs) for
L0, L1 shallow, L1 rollout on every map, ordered by congestion (L1 shallow mean fill time). Right: fill time."""
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from n50_style import *
t = pd.read_csv("results/per_map_n50.csv").set_index("map")
z = pd.read_csv("results/subset_map_zgap.csv").set_index("map")
t["retained"] = z.retained; t = t.sort_values("mean_t_L1h")
y = np.arange(len(t))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.6), facecolor=SURF, sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]; clean(ax)
for m, col, lab in [("U_L0", COL["L0"], "L0"), ("U_L1h", COL["L1h"], "L1 shallow"), ("U_roll", COL["realL1"], "L1 rollout")]:
    sd = t["sd_" + m.split("_")[1]] / np.sqrt(50)
    ax.errorbar(t[m], y, xerr=1.96 * sd, fmt="o", ms=5.5, color=col, mec=SURF, mew=1, elinewidth=1.2, label=lab, zorder=3)
for i, (m, r) in enumerate(t.iterrows()):
    ax.plot([min(r.U_L0, r.U_L1h, r.U_roll), max(r.U_L0, r.U_L1h, r.U_roll)], [i, i], color=GRID, lw=1, zorder=1)
ax.axvline(0, color=INK2, lw=.8, zorder=1)
ax.set_yticks(y); ax.set_yticklabels([f"{m}" + ("" if r.retained else "  ○") for m, r in t.iterrows()], fontsize=8.5, color=INK)
ax.set_xlabel("Universalization score (group time saved vs. a queue, normalized; mean of 50 runs, 95% CI)", color=INK2, fontsize=8.5)
ax.legend(loc="lower left", frameon=False, fontsize=9); ax.set_ylim(-.7, len(t) - .3)
ax = axes[1]; clean(ax)
for m, col, lab in [("mean_t_L0", COL["L0"], "L0"), ("mean_t_L1h", COL["L1h"], "L1 shallow"), ("mean_t_roll", COL["realL1"], "L1 rollout")]:
    ax.plot(t[m], y, "o", ms=5.5, color=col, mec=SURF, mew=1, zorder=3, label=lab)
for i, (m, r) in enumerate(t.iterrows()):
    ax.plot([min(r.mean_t_L0, r.mean_t_L1h, r.mean_t_roll), max(r.mean_t_L0, r.mean_t_L1h, r.mean_t_roll)], [i, i], color=GRID, lw=1, zorder=1)
    ax.annotate(f"{r.stuck_L0:d}", (ax.get_xlim()[1], i), xytext=(0, 0), textcoords="offset points", fontsize=7.5, color=COL["L0"], ha="left", va="center") if r.stuck_L0 else None
ax.set_xlabel("Mean fill time per agent (timesteps), finished runs only", color=INK2, fontsize=8.5)
ax.set_xlim(10, 185); ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.text(1.0, 1.01, "L0 stuck runs/50", transform=ax.transAxes, fontsize=7.5, color=COL["L0"], ha="right")
fig.text(.012, .01, "Maps ordered by congestion (L1 shallow fill time, bottom = easiest). ○ = map not in the restricted 20. Current implementation, 50 runs/map, stuck runs excluded. "
         "\nModels only differ on congested maps; rollout is slower than shallow on 27/28 maps; L0's jams (red counts) are confined to five maps, where its finished runs are also slow.", color=INK2, fontsize=7.6)
fig.tight_layout(rect=(0, .05, 1, 1))
fig.savefig("results/n50_per_map_worlds.png", dpi=200, facecolor=SURF)
