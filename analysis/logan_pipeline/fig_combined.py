"""Combined n=50 figure: top = fit to universalizability judgments (Pearson r, Δ vs L1 shallow);
bottom = universalization slope in Logan's model (ΔELPD vs L1 shallow). All 39 vs restricted 20."""
import matplotlib.pyplot as plt, pandas as pd
from n50_style import *
UNIV = "results/n50_univ_r.csv"; LOGAN = "../../../logan-line-cutting/output/n50_quick_summary.csv"  # run from analysis/logan_pipeline
u = pd.read_csv(UNIV); g = pd.read_csv(LOGAN)
U = {"L0": "L0", "L1 heuristic": "L1h", "L2 heuristic": "L2h", "real L1": "realL1"}
u["model"] = u["model"].map(U)
ORDER = [("L0", "L0"), ("L1h", "L1 shallow"), ("L2h", "L2 shallow"), ("realL1", "L1 rollout")]
SUBS = ["all", "dopt"]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), facecolor=SURF, sharey=True)
def f3(x): return f"{x:+.3f}".replace("0.", ".")
for j, sub in enumerate(SUBS):
    for i, (m, lab) in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        r = u[(u.subset == sub) & (u.model == m)].iloc[0]
        point(axes[0, j], r.r, y, r.r_lo, r.r_hi, COL[m], sub)
        txt = f"{r.r:.2f}" + ("   (ref.)" if m == "L1h" else f"   (Δ {f3(r.d_vs_L1h)} [{f3(r.d_lo)}, {f3(r.d_hi)}])")
        axes[0, j].annotate(txt, (r.r_hi, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=8.5)
        s = g[(g.subset == sub) & (g.model == m)].iloc[0]
        point(axes[1, j], s.slope, y, s.slope_lo, s.slope_hi, COL[m], sub)
        txt = f"{s.slope:.2f}" + ("   (ref.)" if m == "L1h" else f"   (ΔELPD {s.d_elpd_vs_L1h:+.0f} ± {s.se:.0f})")
        axes[1, j].annotate(txt, (s.slope_hi, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=8.5)
    axes[0, j].set_title(SETS[sub][0], color=INK, fontsize=11.5, loc="left", pad=8)
for ax in axes[0]: ax.set_xlim(.45, 1.12); ax.set_xticks([.5, .6, .7, .8, .9, 1.0])
for ax in axes[1]: ax.set_xlim(0, .78)
for ax in axes.ravel():
    clean(ax); ax.set_ylim(-.6, len(ORDER) - .4); ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([l for _, l in ORDER][::-1])
    for t, (m, _) in zip(ax.get_yticklabels(), ORDER[::-1]): t.set_color(COL[m])
for j in range(2):
    axes[0, j].set_xlabel("Fit to universalizability judgments (US, Pearson r)", color=INK2, fontsize=8.5)
    axes[1, j].set_xlabel("Universalization slope in moral judgments (20 countries; logits per SD)", color=INK2, fontsize=8.5)
fig.text(.012, .015,
    "Top: correlation of each model's universalization score with mean universalizability judgments; 95% participant-bootstrap intervals; "
    "Δ = paired difference vs L1 shallow.\nBottom: Logan's hierarchical ordinal model, slope averaged over countries; 95% posterior intervals; "
    "ΔELPD = held-out predictive accuracy vs L1 shallow (± paired SE).\n50 simulation runs per map. Compare within a panel, not across stimulus sets. "
    "Logan's model: quick fits (2 chains × 500 draws).", color=INK2, fontsize=7.6)
fig.tight_layout(rect=(0, .1, 1, 1), h_pad=2.5)
fig.savefig("results/n50_combined.png", dpi=200, facecolor=SURF)
