import matplotlib.pyplot as plt
from n50_style import *
D = {("univ", "L0"):   {"all": (.275, .239, .314), "dopt": (.143, .106, .184)},
     ("univ", "roll"): {"all": (.003, -.012, .022), "dopt": (.026, .007, .045)},
     ("elpd", "L0"):   {"all": (200.9, 23.1), "dopt": (114.3, 16.6)},
     ("elpd", "roll"): {"all": (-104.7, 15.3), "dopt": (-11.1, 11.0)}}
for k, v in D.items():
    if k[0] == "elpd": D[k] = {s: (e, e - 1.96 * se, e + 1.96 * se) for s, (e, se) in v.items()}
ROWS = {"univ": ("Universalizability judgments\n(US, Δ Pearson r)", "{:+.3f}", (-.08, .36)),
        "elpd": ("Moral judgments, 20 countries\n(Logan's model, Δ ELPD)", "{:+.0f}", (-150, 260))}
COLS = {"L0": ("L1 shallow − L0\n(does modelling others' next move help?)", COL["L1h"]),
        "roll": ("L1 rollout − L1 shallow\n(does planning further ahead help?)", COL["realL1"])}
fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), facecolor=SURF)
for i, r in enumerate(ROWS):
    for j, c in enumerate(COLS):
        ax = axes[i, j]; clean(ax)
        for k, s in enumerate(["all", "dopt"]):
            e, lo, hi = D[(r, c)][s]; y = 1 - k
            point(ax, e, y, lo, hi, COLS[c][1], s)
            ax.annotate(ROWS[r][1].format(e), (hi, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        ax.set_xlim(*ROWS[r][2]); ax.set_ylim(-.6, 1.6)
        ax.set_yticks([1, 0]); ax.set_yticklabels([SETS[s][0] for s in ["all", "dopt"]] if j == 0 else [], color=INK)
        if i == 0: ax.set_title(COLS[c][0], color=INK, fontsize=10.5, loc="left", pad=10)
        if j == 0: ax.text(-.75, .5, ROWS[r][0], transform=ax.transAxes, ha="center", va="center", rotation=90, color=INK2, fontsize=9)
fig.text(.012, .015, "Positive = first model fits humans better. 50 simulation runs per map. Filled circle = all 39 stimuli; open square = restricted 20.\n"
         "Bars: 95% intervals (participant bootstrap; ±1.96 paired SE for ELPD). Logan's model: quick fits (2 chains × 500 draws).",
         color=INK2, fontsize=7.8)
fig.tight_layout(rect=(0.07, .07, 1, 1))
fig.savefig("results/n50_all_vs_restricted.png", dpi=200, facecolor=SURF)
