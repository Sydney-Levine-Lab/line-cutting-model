"""
Tools for stat analysis - DEC 11: copy-pasted source-Comparison,
and added useful functions to get Kwon2023 pipeline.

To be cleaned

"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
import itertools


# ---------------------------------------------------------------------
# Global plotting constants
# ---------------------------------------------------------------------

# Distinguish between map types on plots
MAP_MACRO_COLORS = {
    "no_line": "tab:blue",
    "yes_line": "tab:red",
    "maybe": "tab:green",
}
MAP_MARKERS = {
    "no_line_number": "o",
    "no_line_letter": "x",
    "yes_line_number": "o",
    "yes_line_letter": "x",
    "esque": "s",
    "maybe": "o",
    "new_maybe": "x",
}
MAP_SUBTYPE_ORDER = [
    "no_line_number",
    "no_line_letter",
    "yes_line_number",
    "yes_line_letter",
    "esque",
    "maybe",
    "new_maybe",
]

# Effect-size variants when comparing alt_source vs ref_source
EFFECT_KIND_MAP = {
    # public key:   (column_template,                           human-readable y-label template)
    "delta":        ("delta_{metric}_{alt}_vs_{ref}",           "Δ{metric} ({alt} − {ref})"),
    "difference":   ("delta_{metric}_{alt}_vs_{ref}",           "Δ{metric} ({alt} − {ref})"),    
    "d":            ("d_{metric}_{alt}_vs_{ref}",               "Cohen's d ({alt} − {ref})"),
    "cohen_d":      ("d_{metric}_{alt}_vs_{ref}",               "Cohen's d ({alt} − {ref})"),
    "range":        ("q_{metric}_{alt}_vs_{ref}",               "Δ{metric}/range ({alt} − {ref})"),
    "q":            ("q_{metric}_{alt}_vs_{ref}",               "Δ{metric}/range ({alt} − {ref})"),
    "relative":     ("relative_{metric}_{alt}_vs_{ref}",        "Relative Δ{metric} ({alt} vs {ref})"),
    "rel":          ("relative_{metric}_{alt}_vs_{ref}",        "Relative Δ{metric} ({alt} vs {ref})"),
}

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _map_type(name):
    """Divide maps into three macro types"""
    if name.startswith("no_line"):
        return "no_line"
        
    elif name.startswith("yes_line_"):
        return "yes_line"

    else:
        return "maybe"


def _map_subtype(name):
    """Divide maps into seven subtypes"""
    if name.startswith("no_line"):
        suffix = name.split("no_line_")[1]
        if suffix and suffix[-1].isdigit():
            return "no_line_number"
        elif suffix and suffix[-1].isalpha():
            return "no_line_letter"
        
    elif name.startswith("yes_line_"):
        suffix = name.split("yes_line_")[1]
        if suffix and suffix[-1].isdigit():
            return "yes_line_number"
        elif suffix and suffix[-1].isalpha():
            return "yes_line_letter"

    elif name.startswith("maybe"):
        return "maybe"
    elif name.startswith("new_maybe"):
        return "new_maybe"
    elif "esque" in name:
        return "esque"
    
def _add_map_type_columns(df):
    """Add map type and subtype to dataframe"""
    df = df.copy()

    if "map" in df.columns:
        map_col = "map"
    elif "map_name" in df.columns:
        map_col = "map_name"
        # keep an explicit 'map' alias for downstream code that expects it
        #df["map"] = df["map_name"]
    else:
        raise ValueError(
            "Need a 'map' or 'map_name' column to compute map_type/map_subtype."
        )

    df["map_type"] = df[map_col].apply(_map_type)
    df["map_subtype"] = df[map_col].apply(_map_subtype)
    return df

def _pivot_metric(df, metric):
    """Returns wide table (useful for statistics) from summary table"""
    return df.pivot(index="map", columns="source", values=metric)

SHORT_METRICS = {
    "aggregate_welfare": "AW",
    "inequality": "Ineq",
    "cardinal_harm": "CH",
    "ordinal_harm_blind": "OH",
    "gini": "Gini",
}

def _short_metrics(metrics):
    """
    Produce small labels for outcome and/or universalization metrics.

    metrics: list of column names, e.g. ["aggregate_welfare", "univ_aggregate_welfare"]
    returns: list of short labels, e.g. ["AW", "U_AW"]
    """
    short_parts = []
    for m in metrics:
        if m.startswith("univ_"):
            base = m.replace("univ_", "")
            base_short = SHORT_METRICS.get(base, base)
            short_parts.append("U_" + base_short)
        else:
            short_parts.append(SHORT_METRICS.get(m, m))
    return short_parts


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def regression_xy(
    df,
    x_col,
    y_col,
    map_col="map",
):
    """
    Summarize how y relates to x across maps.

    Fits y ~ x (simple linear regression).

    Returns:
    - summary_df : table of correlation, slope, intercept, errors, tests
    - per_map_df : original rows with added diff, abs_diff
    """
    # keep only the columns we need and drop missing
    use_cols = [map_col, x_col, y_col]
    df_use = df[use_cols].dropna().copy()
    df_use = df_use.rename(columns={map_col: "map"})

    x = df_use[x_col].values
    y = df_use[y_col].values
    n = len(x)

    # correlation + regression
    r, p_r = stats.pearsonr(x, y)
    slope, intercept, r_lin, p_lin, stderr = stats.linregress(x, y)

    # Significance test for intercept (vs 0), same as before
    residuals = y - (slope * x + intercept)
    rse = np.sqrt(np.sum(residuals**2) / (n - 2))
    se_intercept = rse * np.sqrt(1/n + np.mean(x)**2 / np.sum((x - np.mean(x))**2))
    t_intercept = intercept / se_intercept
    p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), n - 2))

    # Significance test for slope vs 1 (note: not vs 0)
    t_slope_vs_1 = (slope - 1) / stderr
    p_slope_vs_1 = 2 * (1 - stats.t.cdf(np.abs(t_slope_vs_1), n - 2))

    diff = y - x
    mae  = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_abs = np.max(np.abs(diff))
    value_range = x.max() - x.min()

    summary_data = {
        "Parameter": [
            "Correlation (r)",
            "Slope",
            "Intercept",
            "Mean Abs Error",
            "RMSE",
            "Max Abs Diff",
            "Range (x)",
            "N maps",
        ],
        "Value": [
            f"{r:.4f}",
            f"{slope:.4f}",
            f"{intercept:.4f}",
            f"{mae:.4f}",
            f"{rmse:.4f}",
            f"{max_abs:.4f}",
            f"{value_range:.4f}",
            f"{n}",
        ],
        "Std Error": [
            "—",
            f"{stderr:.4f}",
            f"{se_intercept:.4f}",
            "—",
            "—",
            "—",
            "—",
            "—",
        ],
        "t-stat": [
            "—",
            f"{t_slope_vs_1:.3f}",
            f"{t_intercept:.3f}",
            "—",
            "—",
            "—",
            "—",
            "—",
        ],
        "p-value": [
            "—",
            f"{p_slope_vs_1:.4g}",
            f"{p_intercept:.4g}",
            "—",
            "—",
            "—",
            "—",
            "—",
        ],
        "Sig": [
            "—",
            "***" if p_slope_vs_1  < 0.001 else "**" if p_slope_vs_1  < 0.01 else "*" if p_slope_vs_1 < 0.05 else "ns",
            "***" if p_intercept < 0.001 else "**" if p_intercept < 0.01 else "*" if p_intercept < 0.05 else "ns",
            "—",
            "—",
            "—",
            "—",
            "—",
        ],
    }

    summary_df = pd.DataFrame(summary_data)

    df_use["diff"] = diff
    df_use["abs_diff"] = np.abs(diff)

    return summary_df, df_use

def add_effect_sizes_by_source(
    df,
    metric,
    ref_source,
    alt_source
    ):
    """
    Add effect-size columns comparing alt_source to ref_source for a given metric, 
    using a summary table (df).

    For each map, computes:
        delta      = metric(alt_source) - metric(ref_source)
        d          = delta / SD(ref_source)       
        q          = delta / range(ref_source)       
        relative   = delta / metric(ref_source)         

    and adds columns such as delta_{metric}_{alt_source}_vs_{ref_source}.
    New values are filled only on rows where df['source'] == alt_source.
    """
    df = df.copy()
    wide = _pivot_metric(df, metric)

    # Across-map SD and range for the reference source
    ref_series = wide[ref_source]
    sd_across_ref = ref_series.std(ddof=1)
    range_ref = ref_series.max() - ref_series.min()

    #eps = 1e-9 # avoid division by zero
    mask_alt = df["source"] == alt_source # add effect sizes to alt_source lines

    # Raw difference
    delta = wide[alt_source] - ref_series
    delta_name = f"delta_{metric}_{alt_source}_vs_{ref_source}"
    df[delta_name] = pd.NA
    delta_dict = delta.to_dict()
    df.loc[mask_alt, delta_name] = df.loc[mask_alt, "map"].map(delta_dict)

    # Cohen's d
    d = delta / sd_across_ref
    d_name = f"d_{metric}_{alt_source}_vs_{ref_source}"
    df[d_name] = pd.NA
    d_dict = d.to_dict()
    df.loc[mask_alt, d_name] = df.loc[mask_alt, "map"].map(d_dict)

    # Range-normalized difference (q)
    if range_ref > 0:
        q = delta / range_ref
        q_name = f"q_{metric}_{alt_source}_vs_{ref_source}"
        df[q_name] = pd.NA
        q_dict = q.to_dict()
        df.loc[mask_alt, q_name] = df.loc[mask_alt, "map"].map(q_dict)

    # Relative difference
    rel = delta / ref_series.abs() 
    rel_name = f"relative_{metric}_{alt_source}_vs_{ref_source}"
    df[rel_name] = pd.NA
    rel_dict = rel.to_dict()
    df.loc[mask_alt, rel_name] = df.loc[mask_alt, "map"].map(rel_dict)

    return df


def add_zscores_by_source(
    df,
    metric,
    ref_source,
    alt_source,
    se_from="both" # "alt", "ref", or "both"
):
    """
    Add per-map z-statistics, comparing alt_source to ref_source for a given metric, 
    using a summary table (df).

    For each map, computes:
        delta      = metric(alt_source) - metric(ref_source)
        z          = delta / SE(map)

    SE is computed based on both sources by default.
    Other options: use alt_source, treating ref_source as the true distribution (or the other way around).

    Adds columns such as z_{metric}_{alt_source}_vs_{ref_source}.
    New values are filled only on rows where df['source'] == alt_source.
    """
    df = df.copy()
    wide_mean = _pivot_metric(df, metric)
    sd_col = metric + "_sd"
    wide_sd = _pivot_metric(df, sd_col)
    wide_n = _pivot_metric(df, "n_runs")

    mask_alt = df["source"] == alt_source # add z-scores to alt_source lines

    # Raw difference
    delta = wide_mean[alt_source] - wide_mean[ref_source]
    delta_name = f"delta_{metric}_{alt_source}_vs_{ref_source}"
    df[delta_name] = pd.NA
    delta_dict = delta.to_dict()
    df.loc[mask_alt, delta_name] = df.loc[mask_alt, "map"].map(delta_dict)

    # Standard error (3 options)
    sd_alt = wide_sd[alt_source]
    n_alt = wide_n[alt_source]
    sd_ref = wide_sd[ref_source]
    n_ref = wide_n[ref_source]
    if se_from=="alt":
        se = sd_alt / np.sqrt(n_alt)
    elif se_from=="ref":
        se = sd_ref / np.sqrt(n_ref)
    elif se_from=="both":
        se = np.sqrt(
            (sd_ref ** 2) / n_ref + (sd_alt ** 2) / n_alt)
    else:
        raise ValueError("sd_from must be one of: 'ref', 'alt', 'both'")
    
    # Z-score
    z = delta / se
    
    z_name = f"z_{metric}_{alt_source}_vs_{ref_source}"
    df[z_name] = pd.NA
    z_dict = z.to_dict()
    df.loc[mask_alt, z_name] = df.loc[mask_alt, "map"].map(z_dict)

    return df

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


def plot_simple_regression(
    df,
    x_col,
    y_col,
    map_col="map_name",
    title=None,
    xlabel=None,
    ylabel=None,
    save_path=None,
    display_line_cutting_condition=False,
    fit_regression=True,
    identity_line=False
):
    """
    Scatterplot comparing y_col vs x_col across scenarios.

    - x-axis: df[x_col]
    - y-axis: df[y_col]
    - one point per scenario (map x line-cutting condition)
    - color = map macro type (no_line / yes_line / maybe)
    - marker = map subtype
    - optional: distinguish worse line-cutting conditions with "b" or "B"

    Overlays:
    - regression line (y ~ x) with 95% CI
    - stats box (r, p, RMSE, max|Δ|, slope, intercept) on the right
    - scenario markers on the right
    """
    # Normalize map column name and add map_type / map_subtype
    df = df.copy().rename(columns={map_col: "map"})
    df = _add_map_type_columns(df)

    # We assume df has a "condition" column already
    use_cols = ["map", x_col, y_col, "map_type", "map_subtype", "condition"]
    sub = df[use_cols].dropna().copy()

    x = sub[x_col].values
    y = sub[y_col].values

    # Basic stats for overlay box
    r, p = stats.pearsonr(x, y)
    slope, intercept, r_lin, p_lin, stderr = stats.linregress(x, y)
    diff = y - x
    rmse = np.sqrt(np.mean(diff**2))
    max_abs = np.max(np.abs(diff))

    # Make extra room on the right for stats + legends
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(right=0.75)  # reserve right side for text/legends

    # Scatter by subtype, with color from macro type
    for fine_cat in MAP_SUBTYPE_ORDER:
        sub_cat = sub[sub["map_subtype"] == fine_cat]
        if sub_cat.empty:
            continue
        macro = sub_cat["map_type"].iloc[0]
        color = MAP_MACRO_COLORS.get(macro, "tab:gray")
        marker = MAP_MARKERS.get(fine_cat, "o")

        ax.scatter(
            sub_cat[x_col],
            sub_cat[y_col],
            label=fine_cat,
            color=color,
            marker=marker,
            s=70,
            alpha=0.9,
        )

    # Regression line
    sns.regplot(
        data=sub,
        x=x_col,
        y=y_col,
        scatter=False,
        color="black",
        line_kws={"linewidth": 1.5, "alpha": 0.8},
        ci=95,
        ax=ax,
    )
    if identity_line:
        ax.axline((0, 0), slope=1, linestyle="--", linewidth=1, alpha=0.5)

    # Titles and labels
    if title is None:
        title = f"{y_col} vs {x_col}"
    ax.set_title(title)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)

    # Stats box – place it just outside the right edge of the axes
    if fit_regression:
        stats_text = (
            f"r = {r:.2f}, p = {p:.3g}\n"
            f"RMSE = {rmse:.2f}\n"
            f"max|Δ| = {max_abs:.2f}\n"
            f"slope = {slope:.2f}\n"
            f"intercept = {intercept:.2f}"
        )
    else:
        stats_text = (
            f"r = {r:.2f}, p = {p:.3g}"
        )

    ax.text(
        1.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        clip_on=False,
    )

    # Optional: tiny labels to distinguish 1cut_bad and 1cut_badder scenarios
    if display_line_cutting_condition:
        dx = 0.01 * (sub[x_col].max() - sub[x_col].min())
        dy = 0.01 * (sub[y_col].max() - sub[y_col].min())

        for _, row in sub.iterrows():
            cond = row["condition"]
            if cond == "1cut_bad":
                label = "b"
            elif cond == "1cut_badder":
                label = "B"
            else:
                label = None

            if label is not None:
                ax.text(
                    row[x_col] + dx,
                    row[y_col] + dy,
                    label,
                    fontsize=8,
                    color="black",
                    alpha=0.8,
                )

    # ----- Legend 1: map subtype (7 categories) -----
    handles, labels = ax.get_legend_handles_labels()
    subtype_handles = []
    subtype_labels = []
    for h, lab in zip(handles, labels):
        if lab in MAP_SUBTYPE_ORDER:
            subtype_handles.append(h)
            subtype_labels.append(lab)

    leg1 = ax.legend(
        subtype_handles,
        subtype_labels,
        title="Map",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),  # to the right of the axes
        borderaxespad=0.0,
        frameon=True,
    )
    ax.add_artist(leg1)

    # ----- Legend 2: line-cutting condition (b/B) -----
    if display_line_cutting_condition:
        cond_handles = [
            mpatches.Patch(color="white", alpha=0.0, label="b = bad"),
            mpatches.Patch(color="white", alpha=0.0, label="B = badder"),
        ]
        cond_labels = [h.get_label() for h in cond_handles]

        leg2 = ax.legend(
            cond_handles,
            cond_labels,
            title="Line-cutting condition",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.15),  # slightly lower on the right
            borderaxespad=0.0,
            frameon=True,
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax







def plot_effect_vs_sd_by_source(
    df,
    metric,
    ref_source,
    alt_source,
    effect_kind,    # "delta", "d"/"cohen_d", "range"/"q", "relative"/"rel"
    title=None,
    xlabel=None,
    ylabel=None,
    label_threshold=0.25,   # label maps with |effect| > this
    save_path=None,
):
    """
    Scatter plot of effect size vs. noisiness of the map.

        x-axis:   SD of metric for ref_source (per map)
        y-axis:   effect size comparing alt_source vs ref_source

    Assumes df has already been augmented with effect-size columns by 
    add_effect_sizes_by_source().
    """
    df = _add_map_type_columns(df)

    # Normalize and look up effect_kind
    key = effect_kind.lower()
    if key not in EFFECT_KIND_MAP:
        valid = ", ".join(sorted(EFFECT_KIND_MAP.keys()))
        raise ValueError(f"effect_kind must be one of: {valid}")

    col_template, y_label_template = EFFECT_KIND_MAP[key]

    eff_col = col_template.format(
        metric=metric,
        alt=alt_source,
        ref=ref_source,
    )

    if eff_col not in df.columns:
        raise ValueError(
            f"{eff_col} not found in df; call add_effect_sizes_by_source() first for "
            f"metric={metric}, ref_source={ref_source}, alt_source={alt_source}."
        )

    # SD column name in the summary table
    sd_col = f"{metric}_sd"

    # --- get SD for the ref_source per map ---
    df_ref = df[df["source"] == ref_source].copy()
    if df_ref.empty:
        raise ValueError(f"No rows for ref_source={ref_source} in df")

    sd_ref_by_map = df_ref.set_index("map")[sd_col]

    # --- work on rows for the alt_source ---
    df_alt = df[df["source"] == alt_source].copy()
    if df_alt.empty:
        raise ValueError(f"No rows for alt_source={alt_source} in df")

    # Attach SD(ref_source) to alt rows
    ref_sd_col = f"{metric}_sd_{ref_source}"
    df_alt[ref_sd_col] = df_alt["map"].map(sd_ref_by_map)

    # Drop rows without SD or effect
    df_alt = df_alt.dropna(subset=[ref_sd_col, eff_col])

    fig, ax = plt.subplots(figsize=(7, 6))

    for fine_cat in MAP_SUBTYPE_ORDER:
        sub = df_alt[df_alt["map_subtype"] == fine_cat]
        if sub.empty:
            continue

        macro = sub["map_type"].iloc[0]
        color = MAP_MACRO_COLORS.get(macro, "tab:gray")
        marker = MAP_MARKERS.get(fine_cat, "o")

        ax.scatter(
            sub[ref_sd_col],      # SD of ref_source on x-axis
            sub[eff_col],         # effect size on y-axis
            label=fine_cat,
            color=color,
            marker=marker,
            s=70,
            alpha=0.9,
        )

    # Horizontal line at 0 effect
    ax.axhline(0, linestyle="--", linewidth=1, alpha=0.5)

    # Label “big” effects
    if label_threshold is not None:
        for _, row in df_alt.iterrows():
            if abs(row[eff_col]) > label_threshold:
                ax.text(
                    row[ref_sd_col],
                    row[eff_col],
                    row["map"],
                    fontsize=8,
                    ha="left",
                    va="center",
                )

    # Titles and labels
    if title is None:
        title = f"{metric}: effect vs SD ({alt_source} vs {ref_source})"
    ax.set_title(title)

    ax.set_xlabel(xlabel or f"{metric} SD ({ref_source})")

    # Use provided ylabel or default from mapping
    auto_ylabel = y_label_template.format(
        metric=metric,
        alt=alt_source,
        ref=ref_source,
    )
    ax.set_ylabel(ylabel or auto_ylabel)

    # Legend by subtype
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax

def plot_zscores_by_source(
    df,
    metric,
    ref_source,
    alt_source,
    threshold,
    title=None,
    xlabel=None,
    save_path=None,
):
    """
    Horizontal bar plot of z-statistics by map for one metric,
    comparing alt_source to ref_source.

    Assumes df has already been augmented with z-statistics columns by 
    add_zstats_by_source().
    """
    df = _add_map_type_columns(df)

    z_col = f"z_{metric}_{alt_source}_vs_{ref_source}"
    sub = df[(df["source"] == alt_source) & df[z_col].notna()].copy()
    sub = sub.sort_values(z_col, ascending=True) # order maps by z

    colors = sub["map_type"].map(MAP_MACRO_COLORS).fillna("tab:gray") # color bars by map macro type

    fig, ax = plt.subplots(figsize=(8, 7))

    y_pos = range(len(sub))
    ax.barh(y_pos, sub[z_col].values, color=colors)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sub["map"].values)
    ax.set_xlabel(xlabel or f"z-score for {metric}\n({alt_source} vs {ref_source})")

    if title is None:
        title = f"Z-scores: {metric}, {alt_source} vs {ref_source}"
    ax.set_title(title)

    # zero line
    ax.axvline(0, color="black", linewidth=1)

    # threshold lines (e.g. ±2)
    if threshold is not None:
        ax.axvline(threshold, color="gray", linestyle="--", linewidth=1)
        ax.axvline(-threshold, color="gray", linestyle="--", linewidth=1)

    # legend for map_type colors
    handles = []
    for mt, col in MAP_MACRO_COLORS.items():
        if mt in sub["map_type"].values:
            handles.append(mpatches.Patch(color=col, label=mt))
    if handles:
        ax.legend(handles=handles, title="map_type", loc="lower right")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax





# ---------------------------------------------------------------------
# New (Dec 11): for multi-level reg / model selection / kwon2023 pipeline
# ---------------------------------------------------------------------

def annotate_maps(design_df, map_col="map_name"):
    """
    Ensure the dataframe has columns:
       - map   (for utils)
       - map_type
       - map_subtype

    Returns a copy with these columns added.
    """
    df = design_df.copy()
    # normalize name for utils
    df = df.rename(columns={map_col: "map"})
    df = _add_map_type_columns(df)
    # keep both names if you want
    if "map_name" not in df.columns and map_col != "map_name":
        df["map_name"] = df["map"]
    return df


def subset_design(
    design_df,
    map_types=None,        # e.g. ["yes_line", "no_line"]
    map_subtypes=None,     # e.g. ["yes_line_number"]
    maps=None,             # explicit list of map names
    conditions=None,       # e.g. ["1cut_bad", "1cut_badder"]
):
    """
    Filter a design dataframe by map type / subtype / map name / condition.

    All filters are combined with AND:
        - if a filter is None, it's ignored
        - if non-None, you must satisfy it to be kept.
    """
    df = _add_map_type_columns(design_df)  # adds map_type, map_subtype

    mask = pd.Series(True, index=df.index)

    if map_types is not None:
        mask &= df["map_type"].isin(map_types)

    if map_subtypes is not None:
        mask &= df["map_subtype"].isin(map_subtypes)

    if maps is not None:
        mask &= df["map_name"].isin(maps)  # or df["map_name"].isin(maps)

    if conditions is not None:
        mask &= df["condition"].isin(conditions)

    return df[mask].copy()


def fit_ols(design_df, predictors, dv="rating_mean", run_type=""):
    """
    Fit OLS: dv ~ predictors.
    
    Returns:
        summary_row: dict with model info and fit indices
        model: statsmodels fitted OLS object
    """
    # Filter out predictors that might be missing in this df
    predictors = [p for p in predictors if p in design_df.columns]
    if not predictors:
        raise ValueError("No predictors present in design_df for fit_ols.")

    n_predictors = len(predictors)
    X = design_df[predictors].copy()
    X = sm.add_constant(X)
    y = design_df[dv]

    m = sm.OLS(y, X).fit()

    short_labels = _short_metrics(predictors)


    summary_row = {
        "n_predictors": n_predictors,
        "predictors": "+".join(short_labels),
        "run_type": run_type,
        "R2": m.rsquared,
        "adj_R2": m.rsquared_adj,
        "AIC": m.aic,
        "BIC": m.bic,
        "n": len(design_df),
        "predictors_full_name": "+".join(predictors),
    }
    return summary_row, m

def compare_models(design_df, predictors, run_type="", dv="rating_mean"):
    """    
    Fits all non-empty subsets of `predictors`.

    Returns:
        summary_df: one row per model with R2, adj_R2, AIC, BIC
        models: dict mapping tuple(predictors) -> fitted model
    """
    rows = []

    # make sure we only include predictors that actually exist
    predictors_present = [p for p in predictors if p in design_df.columns]

    for k in range(1, len(predictors_present) + 1):
        for combo in itertools.combinations(predictors_present, k):
            combo = list(combo)

            row, _ = fit_ols(
                design_df=design_df,
                predictors=combo,
                dv=dv,
                run_type=run_type,
            )
            rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("BIC")
    summary_df
    return summary_df