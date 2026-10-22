"""Where the model differences live: per-stimulus contribution to each contrast, retained vs dropped by the
restricted-20 set. Top: universalizability fit (US), contribution to ΔR² (sums to R²_A − R²_B).
Bottom: moral judgments, 20 countries, Logan's model, per-stimulus ΔELPD."""
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from n50_style import *
u = pd.read_csv("results/subset_per_stimulus.csv"); u = u[u.outcome == "univ (US)"]
g = pd.read_csv("../../../logan-line-cutting/output/n50_per_stimulus_delta_elpd.csv"); g = g[g.subset == "all"]
def short(s): return s.replace("_1cut", "").replace("new_", "n_").replace("yesline", "yes_line").replace("noline", "no_line")
PANELS = {  # (title, color, sign) ; sign flips the stored A−B so that positive = first-named model fits better
    "L0":   ("L1 shallow − L0\n(does modelling others' next move help?)", COL["L1h"]),
    "roll": ("L1 rollout − L1 shallow\n(does planning further ahead help?)", COL["realL1"])}
DATA = {("univ", "L0"): (u[u.contrast == "L1 heuristic − L0"], "contrib_R2", 1),
        ("univ", "roll"): (u[u.contrast == "L1 heuristic − real L1"], "contrib_R2", -1),
        ("elpd", "L0"): (g[g.contrast == "L1h − L0"], "delta_elpd", 1),
        ("elpd", "roll"): (g[g.contrast == "L1h − realL1"], "delta_elpd", -1)}
ROWS = {"univ": ("Universalizability judgments (US)\ncontribution to ΔR²", "{:+.3f}"), "elpd": ("Moral judgments, 20 countries\n(Logan's model) ΔELPD", "{:+.0f}")}
fig, axes = plt.subplots(2, 2, figsize=(13, 7.4), facecolor=SURF)
for i, r in enumerate(ROWS):
    for j, c in enumerate(PANELS):
        ax = axes[i, j]; d, col, sign = DATA[(r, c)]
        d = d.assign(v=d[col] * sign).sort_values("v").reset_index(drop=True)
        x = np.arange(len(d)); color = PANELS[c][1]
        for k, row in d.iterrows():
            if row.retained: ax.bar(k, row.v, color=color, width=.8, zorder=2)
            else: ax.bar(k, row.v, facecolor=SURF, edgecolor=color, hatch="////", lw=.8, width=.8, zorder=2)
        ax.axhline(0, color=INK2, lw=.8, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([short(s) for s in d.stimulus], rotation=90, fontsize=6.3, color=INK2)
        ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=.8, zorder=0); ax.set_facecolor(SURF)
        for s in ax.spines.values(): s.set_visible(False)
        tot, drop, keep = d.v.sum(), d.v[~d.retained].sum(), d.v[d.retained].sum()
        f = ROWS[r][1]
        ax.text(.02, .96, f"total {f.format(tot)}   retained 20: {f.format(keep)}   dropped 19: {f.format(drop)}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
        if i == 0: ax.set_title(PANELS[c][0], color=INK, fontsize=10.5, loc="left", pad=10)
        if j == 0: ax.set_ylabel(ROWS[r][0], color=INK2, fontsize=8.5)
        ax.set_xlim(-.8, len(d) - .2)
# legend
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=INK2, label="retained in the new wave (20)"), Patch(facecolor=SURF, edgecolor=INK2, hatch="////", label="dropped (19)")],
           loc="lower center", frameon=False, fontsize=8.5, ncol=2, bbox_to_anchor=(.5, .028))
fig.text(.012, .006, "Positive = first-named model fits humans better on that stimulus. Each panel sums to the pooled contrast. 50 simulation runs per map; "
         "Logan's model: quick fits (2 chains × 500 draws), all-39 fit.", color=INK2, fontsize=7.6)
fig.tight_layout(rect=(0, .055, 1, 1), h_pad=1.5)
fig.savefig("results/n50_where_differences_live.png", dpi=200, facecolor=SURF)
