"""
Comparison utilities for evaluating simulation runs against participant data.

Provides functions to:
1. Evaluate a set of simulation runs against participant judgments
2. Compute simulation noise (SD across runs per map)
3. Plot results (delta-R², betas, correlations) with error bars
4. Compare runs on different metrics

Usage in notebook:
    from comparison_utils import evaluate_runs, plot_info_sweep, compute_simulation_noise
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =====================================================================
# 1. EVALUATION: run regression comparisons
# =====================================================================

def find_dv_col(design_df):
    """Auto-detect the DV column in a design matrix."""
    candidates = ["judgment_mean", "rating_mean", "moral_judgment_mean", "univ_judgment_mean"]
    for c in candidates:
        if c in design_df.columns:
            return c
    raise KeyError(
        f"No DV column found. Expected one of {candidates}. "
        f"Available columns: {list(design_df.columns)}"
    )


def evaluate_single_run(
    design_df,
    baseline_cols=("aggregate_welfare", "ordinal_harm_blind", "inequality"),
    uaw_col="univ_aggregate_welfare",
    dv_col=None,
):
    """
    For a single simulation run's design matrix, compute:
    - baseline model: dv ~ baseline_cols
    - full model: dv ~ baseline_cols + U_AW
    
    If dv_col is None, auto-detects from the DataFrame columns.
    
    Returns dict with R2, delta_R2, beta, SE, p, CI, etc.
    """
    if dv_col is None:
        dv_col = find_dv_col(design_df)
    
    df = design_df.dropna(subset=[uaw_col, dv_col]).copy()
    
    if len(df) < len(baseline_cols) + 3:
        return None
    
    y = df[dv_col]
    
    # Baseline model
    X_base = sm.add_constant(df[list(baseline_cols)])
    base_model = sm.OLS(y, X_base).fit()
    
    # Full model (baseline + U_AW)
    X_full = sm.add_constant(df[list(baseline_cols) + [uaw_col]])
    full_model = sm.OLS(y, X_full).fit()
    
    # Correlation with participant judgments
    r_uaw, p_corr = stats.pearsonr(
        df[uaw_col].values, df[dv_col].values
    )
    
    # Beta CI
    ci = full_model.conf_int().loc[uaw_col]
    
    return {
        "n": int(full_model.nobs),
        "R2_baseline": base_model.rsquared,
        "R2_full": full_model.rsquared,
        "delta_R2": full_model.rsquared - base_model.rsquared,
        "BIC_baseline": base_model.bic,
        "BIC_full": full_model.bic,
        "delta_BIC": full_model.bic - base_model.bic,
        "beta_U_AW": full_model.params.get(uaw_col, np.nan),
        "se_U_AW": full_model.bse.get(uaw_col, np.nan),
        "p_U_AW": full_model.pvalues.get(uaw_col, np.nan),
        "ci_lo_U_AW": ci[0],
        "ci_hi_U_AW": ci[1],
        "r_U_AW": r_uaw,
        "p_corr_U_AW": p_corr,
    }


def evaluate_runs(
    design_mats,
    baseline_cols=("aggregate_welfare", "ordinal_harm_blind", "inequality"),
    uaw_col="univ_aggregate_welfare",
    dv_col=None,
):
    """
    Evaluate multiple simulation runs.
    
    Parameters
    ----------
    design_mats : dict[str, pd.DataFrame]
        tag -> design matrix (from build_design_matrix)
    
    Returns
    -------
    pd.DataFrame with one row per run, sorted by delta_R2 descending.
    """
    rows = []
    for tag, dm in design_mats.items():
        result = evaluate_single_run(
            dm, baseline_cols, uaw_col, dv_col
        )
        if result is not None:
            result["tag"] = tag
            rows.append(result)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("delta_R2", ascending=False).reset_index(drop=True)
    
    # Reorder columns for readability
    lead_cols = ["tag", "n", "r_U_AW", "beta_U_AW", "se_U_AW", "p_U_AW",
                 "ci_lo_U_AW", "ci_hi_U_AW", "delta_R2", "delta_BIC",
                 "R2_baseline", "R2_full"]
    other_cols = [c for c in df.columns if c not in lead_cols]
    df = df[lead_cols + other_cols]
    
    return df


# =====================================================================
# 2. SIMULATION NOISE
# =====================================================================

def compute_simulation_noise(univ_summaries):
    """
    Compute simulation noise from universalization summary DataFrames.
    
    Parameters
    ----------
    univ_summaries : dict[str, pd.DataFrame]
        tag -> universalization summary (from get_universalization_summary).
        Each should have columns: map, univ_aggregate_welfare,
        univ_aggregate_welfare_sd, mean, mean_sd (or similar _sd columns).
    
    Returns
    -------
    pd.DataFrame with noise summary per run.
    """
    rows = []
    for tag, univ in univ_summaries.items():
        row = {"tag": tag, "n_maps": len(univ)}
        
        # Average SD across maps for key metrics
        for metric in ["univ_aggregate_welfare", "mean", "last", "first"]:
            sd_col = f"{metric}_sd"
            if sd_col in univ.columns and metric in univ.columns:
                row[f"{metric}_avg_sd"] = univ[sd_col].mean()
                row[f"{metric}_grand_mean"] = univ[metric].mean()
                mean_val = univ[metric].mean()
                if mean_val != 0:
                    row[f"{metric}_cv"] = univ[sd_col].mean() / abs(mean_val)
        
        # Proportion finished
        if "prop_finished" in univ.columns:
            row["prop_finished_mean"] = univ["prop_finished"].mean()
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# =====================================================================
# 3. PLOTTING
# =====================================================================

# Default reference line styles
DEFAULT_REFS = {
    "random_blind": {
        "label": "Blind, no prediction",
        "color": "#9E9E9E", "linestyle": ":", "linewidth": 2.0, "primary": True,
    },
    "random_blind_L1": {
        "label": "Blind, 1-step prediction",
        "color": "#2E7D32", "linestyle": "--", "linewidth": 2.0, "primary": True,
    },
    "joe": {
        "label": "joe (fixed order, full info)",
        "color": "#C62828", "linestyle": "-.", "linewidth": 0.8, "primary": False,
    },
    "random": {
        "label": "Full information (random order)",
        "color": "#E65100", "linestyle": "-.", "linewidth": 0.8, "primary": False,
    },
}


def _plot_panel(
    ax, eval_df, metric="beta", x_col="info_prob",
    reference_runs=None, ylabel="", title="",
    show_xlabel=True, show_legend=True, logscale=False,
    alpha_threshold=0.05,
):
    """
    Core plotting function for a single panel.
    
    Parameters
    ----------
    ax : matplotlib Axes
    eval_df : pd.DataFrame
        Output of evaluate_runs() with an x_col column for sweep runs.
        Reference runs should also be in this df.
    metric : str
        "beta" (plots beta_U_AW with CI error bars) or "dr2" (plots delta_R2).
    reference_runs : dict or None
        tag -> {label, color, linestyle, linewidth, primary}.
        Primary refs get labeled on the figure; secondary go in legend only.
        If None, uses DEFAULT_REFS.
    logscale : bool
        If True, use log x-axis (pi=0 is placed at 0.0005 and labeled "0*").
    """
    if reference_runs is None:
        reference_runs = DEFAULT_REFS

    # ── Sweep points (rows with non-null x_col) ──
    sweep = eval_df[eval_df[x_col].notna()].copy()
    sweep = sweep.sort_values(x_col)

    if len(sweep) > 0:
        pi_vals = sweep[x_col].values
        ps = sweep["p_U_AW"].values

        if metric == "beta":
            vals = sweep["beta_U_AW"].values
            ses = sweep["se_U_AW"].values
            yerr = [1.96 * ses, 1.96 * ses]
            show_errorbars = True
        else:
            vals = sweep["delta_R2"].values
            show_errorbars = False

        # For log scale, shift 0 to small value
        if logscale:
            x_vals = np.array([max(p, 0.0003) for p in pi_vals])
            ax.set_xscale("log")
        else:
            x_vals = pi_vals

        # Colors by significance
        face_colors = ["#1976D2" if p < alpha_threshold else "#BBDEFB" for p in ps]
        edge_colors = ["#0D47A1" if p < alpha_threshold else "#90CAF9" for p in ps]

        if show_errorbars:
            ax.errorbar(x_vals, vals, yerr=yerr,
                        fmt="none", ecolor="#BDBDBD", elinewidth=1, capsize=2.5, zorder=3)
        ax.scatter(x_vals, vals, s=55, c=face_colors, edgecolors=edge_colors,
                   linewidths=0.8, zorder=5)
        ax.plot(x_vals, vals, color="#1976D2", alpha=0.25, linewidth=0.8, zorder=2)

    # ── Reference lines ──
    val_col = "beta_U_AW" if metric == "beta" else "delta_R2"
    all_vals = []

    for tag, style in reference_runs.items():
        row = eval_df[eval_df["tag"] == tag]
        if len(row) == 0:
            continue
        ref_val = row[val_col].iloc[0]
        all_vals.append(ref_val)
        is_primary = style.get("primary", False)

        ax.axhline(y=ref_val,
                   color=style.get("color", "gray"),
                   linestyle=style.get("linestyle", "--"),
                   linewidth=style.get("linewidth", 1.3),
                   alpha=0.8 if is_primary else 0.45, zorder=1)

        # Label primary refs on the right edge using axes transform
        if is_primary and len(sweep) > 0:
            # Place label at right edge of plot area
            ax.text(1.01, ref_val, style.get("label", tag),
                    transform=ax.get_yaxis_transform(),
                    fontsize=7, color=style.get("color", "gray"),
                    fontweight="bold", va="center", ha="left")

    # Zero line
    ax.axhline(y=0, color="black", linewidth=0.3, alpha=0.3)

    # ── Formatting ──
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
    if show_xlabel:
        ax.set_xlabel("Information probability (π)", fontsize=9, labelpad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.5)

    if logscale:
        ax.set_xticks([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
        ax.set_xticklabels([".001", ".005", ".01", ".05", ".1", ".5", "1"], fontsize=7)
        ax.set_xlim(0.0002, 1.5)
        # Annotate that leftmost point is truly π=0
        if 0.0 in pi_vals:
            idx_0 = list(pi_vals).index(0.0)
            ax.annotate("π=0", xy=(x_vals[idx_0], vals[idx_0]),
                        xytext=(x_vals[idx_0] * 1.5, vals[idx_0]),
                        fontsize=6, color="#757575",
                        arrowprops=dict(arrowstyle="-", color="#BDBDBD", lw=0.5))
    else:
        ax.set_xticks([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".01", ".05", ".1", ".2", ".3", ".5", ".7", "1.0"], fontsize=7)

    # ── Legend (secondary refs + significance markers) ──
    if show_legend:
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1976D2",
                       markeredgecolor="#0D47A1", markersize=6,
                       label="β > 0 (p < .05)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#BBDEFB",
                       markeredgecolor="#90CAF9", markersize=6,
                       label="β n.s. (p ≥ .05)"),
        ]
        for tag, style in reference_runs.items():
            if not style.get("primary", False):
                legend_elements.append(
                    plt.Line2D([0], [0],
                               color=style.get("color", "gray"),
                               linestyle=style.get("linestyle", "--"),
                               linewidth=style.get("linewidth", 0.8),
                               label=style.get("label", tag))
                )
        ax.legend(handles=legend_elements, loc="center right", fontsize=6,
                  framealpha=0.95, edgecolor="#E0E0E0", fancybox=True)


def plot_two_panel(
    eval_df_left, eval_df_right,
    metric="beta",
    title_left="Cross-cultural moral judgments (N=2,059, 20 countries)",
    title_right="Universalizability judgments (US, N=43)",
    suptitle="Effect of partial information on universalization",
    x_col="info_prob",
    reference_runs=None,
    logscale=False,
    save_path=None,
    figsize=(15, 4.5),
):
    """
    Two-panel figure: left panel for one DV, right panel for another.
    
    Parameters
    ----------
    eval_df_left, eval_df_right : pd.DataFrame
        Output of evaluate_runs() for each DV. Must include x_col.
    metric : str
        "beta" or "dr2"
    """
    ylabel = "β (U_AW)" if metric == "beta" else "ΔR²"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(right=0.85)  # make room for right-side labels

    _plot_panel(ax1, eval_df_left, metric=metric, x_col=x_col,
                reference_runs=reference_runs,
                ylabel=ylabel, title=title_left,
                show_legend=True, logscale=logscale)

    _plot_panel(ax2, eval_df_right, metric=metric, x_col=x_col,
                reference_runs=reference_runs,
                ylabel=ylabel, title=title_right,
                show_legend=False, logscale=logscale)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        # Also save PDF
        fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_single_panel(
    eval_df,
    metric="beta",
    title="",
    x_col="info_prob",
    reference_runs=None,
    logscale=False,
    save_path=None,
    figsize=(7, 4.5),
):
    """Single-panel version for individual plots."""
    ylabel = "β (U_AW)" if metric == "beta" else "ΔR²"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_panel(ax, eval_df, metric=metric, x_col=x_col,
                reference_runs=reference_runs,
                ylabel=ylabel, title=title,
                show_legend=True, logscale=logscale)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    return fig, ax


def plot_comparison_table(
    eval_df,
    cols=None,
    title="Simulation run comparison",
    save_path=None,
):
    """Print a nicely formatted comparison table."""
    if cols is None:
        cols = ["tag", "r_U_AW", "beta_U_AW", "se_U_AW", "p_U_AW",
                "delta_R2", "delta_BIC"]

    display_df = eval_df[[c for c in cols if c in eval_df.columns]].copy()

    for c in display_df.columns:
        if display_df[c].dtype in [np.float64, np.float32]:
            if "p_" in c:
                display_df[c] = display_df[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            else:
                display_df[c] = display_df[c].round(4)

    if save_path:
        display_df.to_csv(save_path, index=False)

    return display_df