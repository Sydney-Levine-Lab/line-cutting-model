import matplotlib.pyplot as plt, pandas as pd
from n50_style import *
d = pd.read_csv("../../../logan-line-cutting/output/n50_quick_summary.csv")  # run from analysis/logan_pipeline
ORDER = [("L0", "L0"), ("L1h", "L1 shallow"), ("L2h", "L2 shallow"), ("realL1", "L1 rollout")]  # L2 rollout left out: incomplete runs
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURF, sharey=True)
for ax, sub in zip(axes, ["all", "dopt"]):
    clean(ax)
    for i, (m, lab) in enumerate(ORDER):
        r = d[(d.subset == sub) & (d.model == m)].iloc[0]; y = len(ORDER) - 1 - i
        point(ax, r.slope, y, r.slope_lo, r.slope_hi, COL[m], sub)
        de = "ref." if m == "L1h" else f"{r.d_elpd_vs_L1h:+.0f} ± {r.se:.0f}"
        ax.annotate(f"{r.slope:.2f}   (ΔELPD {de})", (r.slope_hi, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=8.5)
    ax.set_xlim(0, .75); ax.set_ylim(-.6, len(ORDER) - .4)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([l for _, l in ORDER][::-1], color=INK)
    for t, (m, _) in zip(ax.get_yticklabels(), ORDER[::-1]): t.set_color(COL[m])
    ax.set_title(SETS[sub][0], color=INK, fontsize=10.5, loc="left")
    ax.set_xlabel("Universalization slope (logits per SD, mean over 20 countries)", color=INK2, fontsize=8.5)
fig.text(.012, .02, "Logan's model, moral judgments, 20 countries; quick fits; 95% posterior intervals; ΔELPD vs L1 shallow in brackets. "
         "Compare slopes within a panel only\n(each score is standardized over its own stimulus set). exp(slope) = odds ratio per SD, e.g. 0.31 → 1.36.",
         color=INK2, fontsize=7.6)
fig.tight_layout(rect=(0, .1, 1, 1))
fig.savefig("results/n50_logan_slopes.png", dpi=200, facecolor=SURF)
