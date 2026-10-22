# Shared style, from the notebook's STYLE cell (n50_stimuli-subset_robustness.ipynb)
import matplotlib.pyplot as plt
from matplotlib import font_manager
fam = "Avenir Next" if any("Avenir Next" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 10})
SURF, INK, INK2, GRID = "#FFFFFF", "#202124", "#5F6368", "#D8D4CC"
COL = {"L0": "#D95F4C", "L1h": "#2F6DB2", "L2h": "#8EADD2", "realL1": "#7A5AA6", "realL2": "#B6A4C9"}
# same hue for both stimulus sets; filled circle = all 39, open square = restricted 20
SETS = {"all": ("All 39 stimuli", "o", True), "dopt": ("Restricted 20 (new wave)", "s", False)}
def point(ax, x, y, lo, hi, color, subset):
    _, mk, filled = SETS[subset]
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(x, y, marker=mk, ms=8, mfc=color if filled else SURF, mec=color, mew=2, zorder=3)
def clean(ax):
    ax.set_facecolor(SURF); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
    ax.grid(axis="x", color=GRID, lw=.8, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
