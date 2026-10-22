"""Resource-rationality figure: compute cost per run (log s, median over maps; whiskers = IQR across maps)
vs (left) fit to universalizability judgments, (right) ΔELPD vs L1 shallow in Logan's model. All 39 stimuli, n=50."""
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from n50_style import *
c = pd.read_csv("results/compute_per_map_n50.csv").groupby("model").sec_mean.agg(med="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
u = pd.read_csv("results/n50_univ_r.csv"); u = u[u.subset == "all"].set_index("model").rename(index={"L1 heuristic": "L1h", "L2 heuristic": "L2h", "real L1": "realL1", "real L2": "realL2"})
g = pd.read_csv("../../../logan-line-cutting/output/n50_quick_summary_v2.csv"); g = g[g.subset == "all"].set_index("model")
LAB = {"L0": "L0", "L1h": "L1 shallow", "L2h": "L2 shallow", "realL1": "L1 rollout", "realL2": "L2 rollout*"}
OFF = {"L0": (12, -3, "left"), "L1h": (0, -16, "center"), "L2h": (0, 12, "center"), "realL1": (0, -16, "center"), "realL2": (0, 12, "center")}
OFF2 = {"L0": (12, -3, "left"), "L1h": (12, -3, "left"), "L2h": (12, -3, "left"), "realL1": (12, -3, "left"), "realL2": (-12, -3, "right")}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
for ax, off, (ycol, lo, hi, src, ylab, hline) in zip(axes, [OFF, OFF2], [("r", "r_lo", "r_hi", u, "Fit to universalizability judgments (US, Pearson r)", None),
                                                     ("d_elpd_vs_L1h", None, None, g, "Moral judgments, 20 countries: ΔELPD vs L1 shallow", 0)]):
    clean(ax); ax.set_xscale("log")
    if hline is not None: ax.axhline(hline, color=INK2, lw=.8, zorder=1)
    for m in ["L0", "L1h", "L2h", "realL1", "realL2"]:
        x = c.loc[m, "med"]; y = src.loc[m, ycol]
        if lo: e = [[y - src.loc[m, lo]], [src.loc[m, hi] - y]]
        else: s = src.loc[m, "se"] * 1.96; e = [[s], [s]]
        ax.errorbar(x, y, yerr=e, xerr=[[x - c.loc[m, "q25"]], [c.loc[m, "q75"] - x]], fmt="o", ms=8, color=COL[m], mec=SURF, mew=1.5, elinewidth=1.6, capsize=0, zorder=3)
        o = off[m]; xa = c.loc[m, "q75"] if o[2]=="left" else (c.loc[m, "q25"] if o[2]=="right" else x)
        ax.annotate(LAB[m], (xa, y), xytext=o[:2], textcoords="offset points", ha=o[2], va="center", fontsize=9, color=COL[m])
    ax.set_xlabel("Compute cost per simulated run (seconds, log scale; median over maps, IQR)", color=INK2, fontsize=8.5)
    ax.set_ylabel(ylab, color=INK2, fontsize=8.5); ax.set_xlim(150, 60000); ax.grid(axis="y", color=GRID, lw=.8, zorder=0)
axes[0].set_ylim(.45, .95); axes[1].set_ylim(-260, 40)
fig.text(.012, .01, "All 39 stimuli, 50 runs/map (*real L2: 1395/1400). Vertical bars: 95% participant-bootstrap interval (left); ±1.96 paired SE, quick fits (right). "
         "Times from the cluster, one cold-cache cell per run.\nCaveat: the A* search guide currently collapses to an uninformed search on 26/28 maps, so absolute times are inflated for every model; to be regenerated after the fix.", color=INK2, fontsize=7.4)
fig.tight_layout(rect=(0, .09, 1, 1))
fig.savefig("results/n50_compute_vs_fit.png", dpi=200, facecolor=SURF)
