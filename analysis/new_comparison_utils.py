"""
sim_experiment_comparison.py

Compare simulation runs against experimental participant data.

Core workflow (in notebook):
    1. datasets = load_experiments(...)
    2. sims = load_simulations(runs)
    3. out_df = load_outcome_metrics()
    4. design_mats = build_all_design_matrices(sims, out_df, dataset)
    5. eval_df = evaluate_fits(design_mats, dataset)
    6. corr, styled = correlation_matrix(design_mats, datasets_list)
    7. split_half_all(datasets)
    8. plot_grouped_bars(eval_df, ...)

Naming conventions (all merges go through scenario_lookup.csv):
    - "stimulus"        = preferred_label (e.g., "maybe_4_1cut")
    - "map"             = simulation map name (e.g., "maybe_4")
    - "context"         = Joe's CogSci 2023 label (e.g., "Maybe 1")
    - "cogsci 26"       = Logan's cross-cultural label (= preferred_label)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_PATHS = {
    "sim_root":         Path("../data/simulations"),
    "scenario_dir":     Path("../data/scenarios"),
    "experimental_dir": Path("../data/experimental"),
    "lookup_csv":       Path("../data/scenarios/scenario_lookup.csv"),
    "outcome_csv":      Path("../data/scenarios/outcome_metrics.csv"),
    "figures_dir":      Path("figures"),
}

DEFAULT_BASELINE_COLS = ("aggregate_welfare", "ordinal_harm_blind", "inequality")
UAW_COL = "univ_aggregate_welfare"


# =====================================================================
# 1. LOADING
# =====================================================================

class ExperimentalDataset:
    """
    Container for one experimental dataset.

    Attributes
    ----------
    name : str            Display name (e.g., "Joe moral", "Logan moral (pooled)")
    agg_df : DataFrame    Stimulus-level means. Must have 'stimulus' column + dv_col.
    dv_col : str          Column name for the DV in agg_df.
    long_df : DataFrame   Optional participant-level data (for split-halves).
    n_stimuli : int       Number of stimuli (28 or 39).
    """

    def __init__(self, name, agg_df, dv_col, long_df=None,
                 participant_col="subject.code", judgment_col="answer",
                 stimulus_col="stimulus", description=""):
        self.name = name
        self.agg_df = agg_df
        self.dv_col = dv_col
        self.long_df = long_df
        self.participant_col = participant_col
        self.judgment_col = judgment_col
        self.stimulus_col = stimulus_col
        self.description = description
        self.n_stimuli = agg_df["stimulus"].nunique()

    def __repr__(self):
        n_part = (self.long_df[self.participant_col].nunique()
                  if self.long_df is not None else "?")
        return (f"ExperimentalDataset('{self.name}', "
                f"{self.n_stimuli} stimuli, {n_part} participants)")


# ---------- Logan ----------

def load_logan_pooled(xp_file="country_agg_judgment.csv",
                      experimental_dir=None, lookup_csv=None):
    """Logan cross-cultural moral judgments, pooled across all countries."""
    experimental_dir = Path(experimental_dir or DEFAULT_PATHS["experimental_dir"])
    lookup_csv = Path(lookup_csv or DEFAULT_PATHS["lookup_csv"])

    xp = pd.read_csv(experimental_dir / xp_file)
    lookup = pd.read_csv(lookup_csv)

    # Logan's stimulus names already match preferred_label
    # but apply the map anyway for safety
    label_map = dict(zip(lookup["cogsci 26"], lookup["preferred_label"]))
    xp["stimulus"] = xp["stimulus"].map(label_map).fillna(xp["stimulus"])

    agg = (xp.groupby("stimulus", as_index=False)
             .agg(judgment_mean=("judgment_mean", "mean")))

    return ExperimentalDataset(
        name="Logan moral (pooled)", agg_df=agg, dv_col="judgment_mean",
        description="Cross-cultural moral judgments pooled, 20 countries, N≈2059")


def load_logan_country(country, xp_file="country_agg_judgment.csv",
                       experimental_dir=None, lookup_csv=None):
    """Logan moral judgments for a single country."""
    experimental_dir = Path(experimental_dir or DEFAULT_PATHS["experimental_dir"])
    lookup_csv = Path(lookup_csv or DEFAULT_PATHS["lookup_csv"])

    xp = pd.read_csv(experimental_dir / xp_file)
    lookup = pd.read_csv(lookup_csv)

    label_map = dict(zip(lookup["cogsci 26"], lookup["preferred_label"]))
    xp["stimulus"] = xp["stimulus"].map(label_map).fillna(xp["stimulus"])

    sub = xp[xp["country"] == country].copy()
    if len(sub) == 0:
        available = sorted(xp["country"].unique())
        raise ValueError(f"Country '{country}' not found. Available: {available}")

    agg = sub[["stimulus", "judgment_mean"]].copy()
    n = sub["sample_size"].iloc[0]

    return ExperimentalDataset(
        name=f"Logan moral ({country})", agg_df=agg, dv_col="judgment_mean",
        description=f"Moral judgments from {country} (N≈{n})")


# ---------- Joe ----------

def load_joe(agg_file, long_file=None, dv_name="moral",
             experimental_dir=None, lookup_csv=None):
    """
    Load Joe's data (exported from his R script).

    agg_file:  39 rows, columns include 'context' and 'mean'.
    long_file: ~1500+ rows, columns include 'subject.code', 'answer', 'context'.
    dv_name:   label ("moral", "univ", "outcome").
    """
    experimental_dir = Path(experimental_dir or DEFAULT_PATHS["experimental_dir"])
    lookup_csv = Path(lookup_csv or DEFAULT_PATHS["lookup_csv"])
    lookup = pd.read_csv(lookup_csv)

    context_to_label = dict(zip(lookup["cogsci 23"], lookup["preferred_label"]))

    # --- Aggregated ---
    agg = pd.read_csv(experimental_dir / agg_file)
    agg["stimulus"] = agg["context"].map(context_to_label)
    unmapped = agg["stimulus"].isna().sum()
    if unmapped:
        missing = agg.loc[agg["stimulus"].isna(), "context"].tolist()
        print(f"  Warning: {unmapped} Joe contexts unmapped: {missing}")

    dv_col = f"joe_{dv_name}_mean"
    if "mean" in agg.columns:
        agg = agg.rename(columns={"mean": dv_col})

    # --- Long (optional) ---
    long_df = None
    if long_file is not None:
        long_df = pd.read_csv(experimental_dir / long_file)
        long_df["stimulus"] = long_df["context"].map(context_to_label)

    return ExperimentalDataset(
        name=f"Joe {dv_name}", agg_df=agg, dv_col=dv_col,
        long_df=long_df,
        participant_col="subject.code", judgment_col="answer",
        stimulus_col="stimulus",
        description=f"Joe {dv_name} judgments (CogSci 2023, 39 stimuli)")


# ---------- Convenience loader ----------

def load_experiments(
    joe_moral_agg="joe_moral_agg.csv",   joe_moral_long="joe_moral_long.csv",
    joe_univ_agg="joe_univ_agg.csv",     joe_univ_long="joe_univ_long.csv",
    joe_outcome_agg=None,                joe_outcome_long=None,
    logan_countries=None,
    experimental_dir=None, lookup_csv=None,
):
    """
    Load all experimental datasets into an OrderedDict.

    Parameters
    ----------
    logan_countries : list[str] or None
        Individual countries to load (e.g., ["United States", "Japan"]).
    """
    datasets = {}
    kw = dict(experimental_dir=experimental_dir, lookup_csv=lookup_csv)

    # Logan pooled
    try:
        datasets["logan_moral"] = load_logan_pooled(**kw)
    except Exception as e:
        print(f"  Could not load Logan pooled: {e}")

    # Logan per-country
    for c in (logan_countries or []):
        key = f"logan_{c.lower().replace(' ', '_')}"
        try:
            datasets[key] = load_logan_country(c, **kw)
        except Exception as e:
            print(f"  Could not load Logan {c}: {e}")

    # Joe
    for label, af, lf in [
        ("moral",   joe_moral_agg,   joe_moral_long),
        ("univ",    joe_univ_agg,    joe_univ_long),
        ("outcome", joe_outcome_agg, joe_outcome_long),
    ]:
        if af is None:
            continue
        try:
            datasets[f"joe_{label}"] = load_joe(af, lf, label, **kw)
        except Exception as e:
            print(f"  Could not load Joe {label}: {e}")

    return datasets


# ---------- Simulations ----------

def load_simulations(runs, sim_root=None, recompute=False):
    """
    Load universalization summary DataFrames for a dict of runs.

    runs : dict[tag, run_label]   e.g. {"blind_L0": "blind_L0_random_02-06"}
    """
    from build_utils import get_universalization_summary
    sim_root = str(sim_root or DEFAULT_PATHS["sim_root"])
    sims = {}
    for tag, label in runs.items():
        try:
            df = get_universalization_summary(label, sim_root=sim_root,
                                             recompute=recompute)
            sims[tag] = df
            print(f"  {tag:20s}  {len(df)} maps  "
                  f"mean U_AW={df['univ_aggregate_welfare'].mean():.3f}")
        except Exception as e:
            print(f"  {tag:20s}  FAILED — {e}")
    return sims


def load_outcome_metrics(outcome_csv=None, recompute=False):
    """Load scenario outcome metrics (one row per stimulus)."""
    from build_utils import get_outcome_metrics
    return get_outcome_metrics(
        outcome_csv=str(outcome_csv or DEFAULT_PATHS["outcome_csv"]),
        recompute=recompute)


# =====================================================================
# 2. DESIGN MATRICES
# =====================================================================

def build_design_matrix(univ_df, out_df, experiment):
    """
    Merge simulation U_AW + outcome metrics + experimental DV.

    The merge key is 'stimulus'. For Joe's 39-stimulus data the _bad/_badder
    variants share a map (and thus the same U_AW) with the original stimulus.
    """
    scenario_univ = out_df.merge(univ_df, on="map", how="left")
    agg = experiment.agg_df[["stimulus", experiment.dv_col]].copy()
    design = agg.merge(scenario_univ, on="stimulus", how="inner")
    return design


def build_all_design_matrices(sims, out_df, experiment):
    """Build design matrices for all simulation runs × one experiment."""
    mats = {}
    for tag, univ_df in sims.items():
        try:
            mats[tag] = build_design_matrix(univ_df, out_df, experiment)
        except Exception as e:
            print(f"  {tag}: {e}")
    return mats


# =====================================================================
# 3. EVALUATION
# =====================================================================

def evaluate_single_run(design_df, dv_col,
                        baseline_cols=DEFAULT_BASELINE_COLS,
                        uaw_col=UAW_COL):
    """
    Baseline (outcome vars only) vs. full (+ U_AW) regression.

    Returns dict with R², ΔR², β, SE, CI, p, r, etc.
    """
    bl = [c for c in baseline_cols if c in design_df.columns]
    df = design_df.dropna(subset=[uaw_col, dv_col]).copy()
    if len(df) < len(bl) + 3:
        return None

    y = df[dv_col]

    X_base = sm.add_constant(df[list(bl)])
    base = sm.OLS(y, X_base).fit()

    X_full = sm.add_constant(df[list(bl) + [uaw_col]])
    full = sm.OLS(y, X_full).fit()

    r_uaw, _ = stats.pearsonr(df[uaw_col].values, y.values)
    ci = full.conf_int().loc[uaw_col]

    return {
        "n":            int(full.nobs),
        "R2_baseline":  base.rsquared,
        "R2_full":      full.rsquared,
        "delta_R2":     full.rsquared - base.rsquared,
        "BIC_baseline": base.bic,
        "BIC_full":     full.bic,
        "delta_BIC":    full.bic - base.bic,
        "beta_U_AW":    full.params.get(uaw_col, np.nan),
        "se_U_AW":      full.bse.get(uaw_col, np.nan),
        "p_U_AW":       full.pvalues.get(uaw_col, np.nan),
        "ci_lo_U_AW":   ci[0],
        "ci_hi_U_AW":   ci[1],
        "r_U_AW":       r_uaw,
    }


def evaluate_fits(design_mats, experiment,
                  baseline_cols=DEFAULT_BASELINE_COLS,
                  uaw_col=UAW_COL, save_path=None):
    """
    Evaluate all runs against one experiment.

    Returns sorted DataFrame (by delta_R2 desc), one row per run.
    """
    rows = []
    for tag, dm in design_mats.items():
        res = evaluate_single_run(dm, experiment.dv_col, baseline_cols, uaw_col)
        if res is not None:
            res["tag"] = tag
            res["experiment"] = experiment.name
            rows.append(res)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("delta_R2", ascending=False).reset_index(drop=True)

    lead = ["tag", "experiment", "n", "r_U_AW",
            "beta_U_AW", "se_U_AW", "p_U_AW",
            "ci_lo_U_AW", "ci_hi_U_AW",
            "delta_R2", "delta_BIC"]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]

    if save_path:
        df.round(4).to_csv(save_path, index=False)
        print(f"  Saved {save_path}")
    return df


def evaluate_fits_multi(design_mats_per_exp, experiments,
                        baseline_cols=DEFAULT_BASELINE_COLS,
                        uaw_col=UAW_COL, save_path=None):
    """
    Evaluate all runs × all experiments.

    Parameters
    ----------
    design_mats_per_exp : dict[exp_key, dict[tag, DataFrame]]
    experiments : dict[exp_key, ExperimentalDataset]

    Returns DataFrame with columns: tag, experiment, r_U_AW, delta_R2, ...
    """
    frames = []
    for exp_key, mats in design_mats_per_exp.items():
        exp = experiments[exp_key]
        ev = evaluate_fits(mats, exp, baseline_cols, uaw_col)
        frames.append(ev)

    df = pd.concat(frames, ignore_index=True)
    if save_path:
        df.round(4).to_csv(save_path, index=False)
        print(f"  Saved {save_path}")
    return df


# =====================================================================
# 4. CORRELATION MATRIX
# =====================================================================

def correlation_matrix(
    design_mats_per_exp,
    experiments,
    uaw_col=UAW_COL,
    run_order=None,
    exp_order=None,
    group_separators=None,
    save_path=None,
):
    """
    Build an experiments × simulations correlation matrix.

    Rows = experimental DVs, Columns = simulation runs.
    Cell = Pearson r between U_AW from that run and the DV for that experiment,
    computed over the stimuli they share.

    Parameters
    ----------
    design_mats_per_exp : dict[exp_key, dict[tag, DataFrame]]
    experiments         : dict[exp_key, ExperimentalDataset]
    run_order           : list[str] — column order (simulation tags)
    exp_order           : list[str] — row order (experiment keys)
    group_separators    : list[int] — column indices after which to draw separator
                          (e.g. [2] draws a line after the 3rd column)

    Returns (corr_df, styled)
    """
    exp_keys = exp_order or list(design_mats_per_exp.keys())
    all_tags = set()
    for mats in design_mats_per_exp.values():
        all_tags.update(mats.keys())
    tags = run_order or sorted(all_tags)

    rows = []
    for ek in exp_keys:
        exp = experiments[ek]
        mats = design_mats_per_exp.get(ek, {})
        row = {"experiment": exp.name}
        for tag in tags:
            if tag in mats:
                dm = mats[tag]
                sub = dm.dropna(subset=[uaw_col, exp.dv_col])
                if len(sub) >= 3:
                    r, _ = stats.pearsonr(sub[uaw_col], sub[exp.dv_col])
                    row[tag] = r
                else:
                    row[tag] = np.nan
            else:
                row[tag] = np.nan
        rows.append(row)

    corr = pd.DataFrame(rows).set_index("experiment")

    # Style
    styled = (corr.round(3).style
              .background_gradient(cmap="RdYlGn", vmin=0.0, vmax=1.0, axis=None)
              .format("{:.3f}", na_rep="—"))

    if group_separators:
        css = []
        for si in group_separators:
            col_idx = si + 2   # +1 for 0-index, +1 for row header
            css.append({
                "selector": f"td:nth-child({col_idx}), th:nth-child({col_idx})",
                "props": [("border-right", "3px solid #333")]
            })
        styled = styled.set_table_styles(css, overwrite=False)

    if save_path:
        corr.round(4).to_csv(save_path)
        print(f"  Saved {save_path}")

    return corr, styled

def full_correlation_matrix(
    design_mats_per_exp,
    experiments,
    uaw_col=UAW_COL,
    run_order=None,
    exp_order=None,
    group_separators=None,
    save_path=None,
):
    """
    Full cross-correlation matrix: experiments + simulations on BOTH axes.

    Shows experiment-experiment, experiment-simulation, and simulation-simulation
    correlations in a single square matrix.

    Parameters
    ----------
    group_separators : list[int]
        Positions after which to draw separator lines (0-indexed into display order).
        E.g., if you have 4 DVs then 3 baselines then 3 v1, use [3, 5, 8].
    """
    first_exp_key = list(design_mats_per_exp.keys())[0]
    first_tag = list(design_mats_per_exp[first_exp_key].keys())[0]
    base = design_mats_per_exp[first_exp_key][first_tag][["stimulus"]].drop_duplicates()
    combined = base.copy()

    exp_keys = exp_order or list(experiments.keys())
    col_labels = {}

    for ek in exp_keys:
        exp = experiments[ek]
        col = f"__exp__{ek}"
        col_labels[col] = exp.name
        sub = exp.agg_df[["stimulus", exp.dv_col]].rename(columns={exp.dv_col: col})
        combined = combined.merge(sub, on="stimulus", how="left")

    all_tags = set()
    for mats in design_mats_per_exp.values():
        all_tags.update(mats.keys())
    tags = run_order or sorted(all_tags)

    any_mats = list(design_mats_per_exp.values())[0]
    for tag in tags:
        if tag in any_mats:
            col = f"__sim__{tag}"
            col_labels[col] = tag
            sub = any_mats[tag][["stimulus", uaw_col]].rename(columns={uaw_col: col})
            combined = combined.merge(sub, on="stimulus", how="inner")

    all_cols = list(col_labels.keys())
    all_labels = list(col_labels.values())

    corr = combined[all_cols].corr()
    corr.columns = all_labels
    corr.index = all_labels

    styled = (corr.round(3).style
              .background_gradient(cmap="RdYlGn", vmin=0.0, vmax=1.0, axis=None)
              .format("{:.3f}", na_rep="—"))

    if group_separators:
        css = []
        for si in group_separators:
            col_idx = si + 2
            css.append({
                "selector": f"th:nth-child({col_idx}), td:nth-child({col_idx})",
                "props": [("border-right", "3px solid #333")]
            })
            css.append({
                "selector": f"tr:nth-child({si + 1}) td, tr:nth-child({si + 1}) th",
                "props": [("border-bottom", "3px solid #333")]
            })
        styled = styled.set_table_styles(css, overwrite=False)

    if save_path:
        corr.round(4).to_csv(save_path)
        print(f"  Saved {save_path}")

    return corr, styled


# =====================================================================
# 5. SPLIT-HALF RELIABILITY
# =====================================================================

def split_half(experiment, n_splits=1000, seed=42, by_stimulus="stimulus"):
    """
    Split-half reliability (participant split) for one ExperimentalDataset.

    Returns dict with r_half, spearman_brown, noise_ceiling, CIs.
    """
    if experiment.long_df is None:
        raise ValueError(f"No long data for '{experiment.name}'.")

    df = experiment.long_df.copy()
    pcol = experiment.participant_col
    jcol = experiment.judgment_col

    rng = np.random.RandomState(seed)
    parts = df[pcol].unique()
    n = len(parts)

    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(parts)
        ha, hb = set(perm[:n // 2]), set(perm[n // 2:])
        ma = df[df[pcol].isin(ha)].groupby(by_stimulus)[jcol].mean()
        mb = df[df[pcol].isin(hb)].groupby(by_stimulus)[jcol].mean()
        common = ma.index.intersection(mb.index)
        if len(common) >= 3:
            rs.append(np.corrcoef(ma[common], mb[common])[0, 1])

    rs = np.array(rs)
    r = np.mean(rs)
    sb = 2 * r / (1 + r)

    return {
        "name":            experiment.name,
        "n_participants":  n,
        "n_stimuli":       df[by_stimulus].nunique(),
        "r_half":          round(r, 4),
        "ci_lo":           round(np.percentile(rs, 2.5), 4),
        "ci_hi":           round(np.percentile(rs, 97.5), 4),
        "spearman_brown":  round(sb, 4),
        "noise_ceiling":   round(np.sqrt(sb), 4),
    }


def split_half_all(experiments, n_splits=1000, seed=42):
    """Split-half for every experiment that has long data."""
    rows = []
    for key, exp in experiments.items():
        if exp.long_df is not None:
            try:
                rows.append(split_half(exp, n_splits, seed))
            except Exception as e:
                print(f"  {key}: {e}")
    return pd.DataFrame(rows)


# =====================================================================
# 6. PLOTTING
# =====================================================================

def plot_grouped_bars(
    eval_df,
    metric="delta_R2",
    categories=None,
    ylabel=None,
    title=None,
    figsize=(12, 5),
    save_path=None,
):
    """
    Grouped bar chart of a metric across simulation runs, colored by category.

    Parameters
    ----------
    eval_df : DataFrame from evaluate_fits()
    categories : list of (cat_name, [tags], color).
        Default groups: Baselines, v1 blind, v2 obs.model, Other.
    """
    if categories is None:
        categories = [
            ("Baselines",        ["joe", "random_full", "blind_L0", "blind_L1"], "#78909C"),
            ("v1 (blind)",       ["depth0", "depth1", "depth2"],                 "#42A5F5"),
            ("v2 (obs. model)",  ["depth0_v2", "depth1_v2", "depth2_v2"],        "#FF8A65"),
        ]

    if ylabel is None:
        ylabel = {"delta_R2": "ΔR²", "beta_U_AW": "β (U_AW)",
                  "r_U_AW": "r"}.get(metric, metric)
    if title is None:
        title = f"{ylabel} by model"

    fig, ax = plt.subplots(figsize=figsize)
    x, xt, xl = 0, [], []
    handles = []

    for cat_name, cat_tags, color in categories:
        present = [t for t in cat_tags if t in eval_df["tag"].values]
        if not present:
            continue
        for tag in present:
            row = eval_df[eval_df["tag"] == tag].iloc[0]
            val = row[metric]
            ax.bar(x, val, color=color, alpha=0.85, edgecolor="white", width=0.8)

            if metric == "beta_U_AW":
                lo = row.get("ci_lo_U_AW", np.nan)
                hi = row.get("ci_hi_U_AW", np.nan)
                if pd.notna(lo) and pd.notna(hi):
                    ax.errorbar(x, val, yerr=[[val - lo], [hi - val]],
                                fmt="none", color="black", capsize=3, lw=1)

            p = row.get("p_U_AW", 1.0)
            if pd.notna(p) and p < 0.05:
                ax.text(x, max(val, 0) + 0.003, "*",
                        ha="center", fontsize=12, fontweight="bold")
            xt.append(x); xl.append(tag); x += 1
        x += 0.7
        handles.append(mpatches.Patch(color=color, alpha=0.85, label=cat_name))

    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        fig.savefig(str(save_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()
    return fig


def plot_eval_multi(
    eval_multi_df,
    metric="delta_R2",
    run_order=None,
    figsize=None,
    save_path=None,
):
    """
    Grouped bar chart: one group per simulation run, bars = experiments.

    Parameters
    ----------
    eval_multi_df : DataFrame from evaluate_fits_multi()
        Must have columns: tag, experiment, <metric>
    """
    experiments = eval_multi_df["experiment"].unique()
    tags = run_order or eval_multi_df["tag"].unique()
    n_exp = len(experiments)
    n_tags = len(tags)

    if figsize is None:
        figsize = (max(10, n_tags * 1.2), 5)

    colors = plt.cm.Set2(np.linspace(0, 1, n_exp))

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.8 / n_exp
    x_base = np.arange(n_tags)

    for i, exp_name in enumerate(experiments):
        sub = eval_multi_df[eval_multi_df["experiment"] == exp_name]
        vals = []
        for tag in tags:
            row = sub[sub["tag"] == tag]
            vals.append(row[metric].iloc[0] if len(row) else 0)
        ax.bar(x_base + i * bar_width, vals, bar_width,
               label=exp_name, color=colors[i], alpha=0.85)

    ax.set_xticks(x_base + bar_width * (n_exp - 1) / 2)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=9)
    ylabel = {"delta_R2": "ΔR²", "beta_U_AW": "β", "r_U_AW": "r"}.get(metric, metric)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by model × experiment")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        fig.savefig(str(save_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()
    return fig


def plot_scatter(design_df, x_col, y_col, title=None,
                 xlabel=None, ylabel=None, save_path=None):
    """Quick scatter of two columns in a design matrix."""
    df = design_df.dropna(subset=[x_col, y_col])
    r = df[x_col].corr(df[y_col])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x_col], df[y_col], s=40, alpha=0.7)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"r = {r:.3f}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig

def plot_simple_regression(
    design_df,
    x_col="univ_aggregate_welfare",
    y_col=None,
    title=None,
    xlabel=None,
    ylabel=None,
    display_condition=False,
    save_path=None,
):
    """
    Scatter plot of simulation metric vs experimental DV (or sim vs sim),
    colored by map type (no_line / yes_line / maybe).
    
    Works with any design matrix from build_design_matrix().
    If y_col is None, auto-detects the DV column.
    """
    MAP_COLORS = {"no_line": "tab:blue", "yes_line": "tab:red", "maybe": "tab:green"}
    MAP_MARKERS = {
        "no_line_number": "o", "no_line_letter": "x",
        "yes_line_number": "o", "yes_line_letter": "x",
        "esque": "s", "maybe": "o", "new_maybe": "x",
    }
    SUBTYPE_ORDER = [
        "no_line_number", "no_line_letter",
        "yes_line_number", "yes_line_letter",
        "esque", "maybe", "new_maybe",
    ]

    def _map_type(name):
        if name.startswith("no_line"): return "no_line"
        elif name.startswith("yes_line"): return "yes_line"
        else: return "maybe"

    def _map_subtype(name):
        if name.startswith("no_line"):
            s = name.split("no_line_")[1]
            return "no_line_number" if s[-1].isdigit() else "no_line_letter"
        elif name.startswith("yes_line_"):
            s = name.split("yes_line_")[1]
            return "yes_line_number" if s[-1].isdigit() else "yes_line_letter"
        elif name.startswith("new_maybe"): return "new_maybe"
        elif name.startswith("maybe"): return "maybe"
        elif "esque" in name: return "esque"
        return "maybe"

    # Auto-detect DV
    if y_col is None:
        for c in ["judgment_mean", "joe_moral_mean", "joe_univ_mean", "joe_outcome_mean"]:
            if c in design_df.columns:
                y_col = c
                break

    df = design_df.dropna(subset=[x_col, y_col]).copy()
    df["_map_type"] = df["map"].apply(_map_type)
    df["_map_subtype"] = df["map"].apply(_map_subtype)

    r, p = stats.pearsonr(df[x_col], df[y_col])
    slope, intercept, _, _, _ = stats.linregress(df[x_col], df[y_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(right=0.75)

    for st in SUBTYPE_ORDER:
        sub = df[df["_map_subtype"] == st]
        if sub.empty: continue
        macro = sub["_map_type"].iloc[0]
        ax.scatter(sub[x_col], sub[y_col],
                   label=st, color=MAP_COLORS.get(macro, "gray"),
                   marker=MAP_MARKERS.get(st, "o"), s=70, alpha=0.9)

    sns.regplot(data=df, x=x_col, y=y_col, scatter=False,
                color="black", line_kws={"linewidth": 1.5, "alpha": 0.8},
                ci=95, ax=ax)

    # Condition labels (b/B for bad/badder)
    if display_condition and "condition" in df.columns:
        dx = 0.01 * (df[x_col].max() - df[x_col].min())
        dy = 0.01 * (df[y_col].max() - df[y_col].min())
        for _, row in df.iterrows():
            if row["condition"] == "1cut_bad":
                ax.text(row[x_col]+dx, row[y_col]+dy, "b", fontsize=8, alpha=0.8)
            elif row["condition"] == "1cut_badder":
                ax.text(row[x_col]+dx, row[y_col]+dy, "B", fontsize=8, alpha=0.8)

    stats_text = f"r = {r:.2f}, p = {p:.3g}\nslope = {slope:.2f}\nintercept = {intercept:.2f}"
    ax.text(1.02, 0.95, stats_text, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            clip_on=False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Map", loc="upper left",
              bbox_to_anchor=(1.02, 0.55), frameon=True, fontsize=8)

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"{y_col} vs {x_col}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax