"""
Tools for simple regression and plotting of simulation metrics
against experimental judgments.
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

def fit_ols(design_df, predictors, dv="rating_mean", run_label=""):
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
        "run_label": run_label,
        "R2": m.rsquared,
        "adj_R2": m.rsquared_adj,
        "AIC": m.aic,
        "BIC": m.bic,
        "n": len(design_df),
        "predictors_full_name": "+".join(predictors),
    }
    return summary_row, m

def compare_models(design_df, predictors, run_label="", dv="rating_mean"):
    """    
    Fits all non-empty subsets of `predictors`.

    Returns:
        summary_df: one row per model with R2, adj_R2, AIC, BIC
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
                run_label=run_label,
            )
            rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("BIC")
    summary_df
    return summary_df

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






