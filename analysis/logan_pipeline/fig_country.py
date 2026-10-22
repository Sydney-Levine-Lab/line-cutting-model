"""Per-country forest: universalization slope in Logan's model, by country, for L0 / L1 shallow / L1 rollout,
all 39 vs restricted 20 (n=50 quick fits)."""
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from n50_style import *
s = pd.read_csv("../../../logan-line-cutting/output/n50_per_country_slopes.csv")  # run from analysis/logan_pipeline
MODELS = [("L0", "L0"), ("L1h", "L1 shallow"), ("realL1", "L1 rollout")]
order = s[(s.subset == "all") & (s.model == "L1h")].sort_values("slope").country.tolist()
fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.2), facecolor=SURF, sharey=True)
for ax, sub in zip(axes, ["all", "dopt"]):
    clean(ax); ax.axvline(0, color=INK2, lw=1, zorder=1)
    for k, (m, lab) in enumerate(MODELS):
        off = (k - 1) * 0.27
        d = s[(s.subset == sub) & (s.model == m)].set_index("country").loc[order]
        y = np.arange(len(order)) + off
        ax.hlines(y, d.lo, d.hi, color=COL[m], lw=1.6, zorder=2)
        ax.plot(d.slope, y, ls="", marker=SETS[sub][1], ms=5.5, mfc=COL[m] if SETS[sub][2] else SURF, mec=COL[m], mew=1.4, zorder=3, label=lab)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8.5, color=INK)
    ax.set_ylim(-.7, len(order) - .3); ax.set_xlim(-.3, 1.2)
    ax.set_title(SETS[sub][0], color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Universalization slope (logits per SD)", color=INK2, fontsize=8.5)
axes[0].legend(loc="lower right", frameon=False, fontsize=8.5)
fig.text(.012, .012, "Logan's hierarchical ordinal model, per-country slopes (partially pooled); 95% posterior intervals; quick fits (2 chains × 500 draws); "
         "50 simulation runs per map.\nCountries ordered by the L1 shallow slope on all 39. Compare within a panel only (scores standardized per stimulus set).",
         color=INK2, fontsize=7.6)
fig.tight_layout(rect=(0, .05, 1, 1))
fig.savefig("results/n50_per_country_slopes.png", dpi=200, facecolor=SURF)
