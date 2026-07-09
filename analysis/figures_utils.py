"""
Paper figures for LEVEL×DEPTH simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =====================================================================
# Default grid mapping: tag → (row=LEVEL, col=DEPTH)
# =====================================================================

DEFAULT_GRID_MAP = {
    'L0':                    (0, 0),
    'L1 heuristic':          (0, 1),
    'real L1':               (0, 2),
    'L2 heuristic':          (1, 1),
    'real L2 (1 run only)':  (1, 2),
    'L3 heuristic':          (2, 1),
}

DEFAULT_COLORS = {
    'L0': '#E24B4A',
    'L1 heuristic': '#1D9E75', 'L2 heuristic': '#1D9E75', 'L3 heuristic': '#1D9E75',
    'real L1': '#EF9F27', 'real L2 (1 run only)': '#378ADD',
    'depth2_v1': '#EF9F27', 'depth2_v2': '#EF9F27',
    'blind L0': '#888780', 'blind L1': '#888780', 'joe': '#888780',
}


# =====================================================================
# Figure 1: LEVEL × DEPTH heatmap (pulls values from eval_df)
# =====================================================================

def fig_level_depth_grid(eval_df, metric="delta_R2", grid_map=None,
                         vmin=None, vmax=None,
                         title_suffix=None, save_path=None):
    """
    Heatmap of model fit across the LEVEL × DEPTH grid.
    Auto-scales color range from data if vmin/vmax not provided.
    """
    if grid_map is None:
        grid_map = DEFAULT_GRID_MAP

    # Metric config
    if metric == "r_U_AW":
        fmt = '.2f'
        cbar_label = 'Pearson $r$'
        title_metric = 'Correlation ($r$)'
    elif metric == "delta_R2":
        fmt = '.3f'
        cbar_label = '$\\Delta R^2$'
        title_metric = '$\\Delta R^2$ (additional variance explained)'
    elif metric == "delta_BIC":
        fmt = '.1f'
        cbar_label = '$\\Delta$BIC'
        title_metric = '$\\Delta$BIC (lower = better)'
    else:
        fmt = '.3f'
        cbar_label = metric
        title_metric = metric

    # Build 3×3 grid from eval_df
    grid = np.full((3, 3), np.nan)
    cell_labels = np.full((3, 3), '', dtype=object)

    tag_vals = dict(zip(eval_df['tag'], eval_df[metric]))
    for tag, (row, col) in grid_map.items():
        if tag in tag_vals:
            grid[row, col] = tag_vals[tag]
            cell_labels[row, col] = tag

    na_cells = {(1, 0), (2, 0)}  # L>0, D=0: ToM unused
    nr_cells = set()
    for i in range(3):
        for j in range(3):
            if (i, j) not in na_cells and np.isnan(grid[i, j]) and (i, j) != (0, 0):
                nr_cells.add((i, j))

    # Auto-scale from data
    valid_vals = grid[~np.isnan(grid)]
    if vmin is None:
        if 'R2' in metric or 'BIC' not in metric:
            vmin = 0.0
        else:
            vmin = np.floor(valid_vals.min() / 5) * 5
        if metric == "r_U_AW":
            vmin = max(0, np.floor(valid_vals.min() * 10) / 10 - 0.1)
    if vmax is None:
        vmax = np.ceil(valid_vals.max() * 10) / 10 + 0.05

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    cmap = LinearSegmentedColormap.from_list('ryg',
        ['#E24B4A', '#FAC775', '#1D9E75'], N=256)
    if metric == "delta_BIC":
        cmap = cmap.reversed()
    cmap.set_bad('#F0EFEC')

    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        'D = 0\nno prediction',
        'D = 1\nheuristic',
        'D = 2\nforward simulation'], fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([
        'L = 0\nothers = walls',
        'L = 1\nothers = L0',
        'L = 2\nothers = L1'], fontsize=9)

    ax.set_xlabel('Planning horizon (depth)', fontsize=11, labelpad=10)
    ax.set_ylabel('Theory of mind (nesting)', fontsize=11, labelpad=10)

    for i in range(3):
        for j in range(3):
            val = grid[i, j]
            label = cell_labels[i, j]
            if (i, j) in na_cells:
                ax.text(j, i - 0.05, '= L0,D0', ha='center', va='center',
                        fontsize=10, color='#999', style='italic')
                ax.text(j, i + 0.25, 'ToM unused', ha='center', va='center',
                        fontsize=8, color='#BBB')
            elif (i, j) in nr_cells:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=14, color='#999')
                ax.text(j, i + 0.25, 'not run', ha='center', va='center',
                        fontsize=8, color='#BBB')
            elif not np.isnan(val):
                thresh = vmin + 0.6 * (vmax - vmin)
                color = 'white' if val > thresh else 'black'
                ax.text(j, i - 0.08, f'{val:{fmt}}', ha='center', va='center',
                        fontsize=16, color=color)
                if label:
                    ax.text(j, i + 0.25, label, ha='center', va='center',
                            fontsize=8, color=color, alpha=0.6)

    title = f'{title_metric}'
    if title_suffix:
        title += f'\n{title_suffix}'
    ax.set_title(title, fontsize=11, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label(cbar_label, fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
  
# =====================================================================
# Figure 2: Bootstrap CIs (works for r or ΔR²)
# =====================================================================

def fig_bootstrap_ci(boot_values, model_order=None, ceiling=None, colors=None,
                     xlabel=None, title=None, xlim=None, save_path=None):
    """
    Horizontal bar chart with bootstrap 95% CIs.

    Parameters
    ----------
    boot_values : dict[tag, list[float]]
        Bootstrap samples for each model.
    model_order : list[str] or None
        Display order (top = best). If None, sorted by mean descending.
    colors : dict[str, str] or None
        Falls back to DEFAULT_COLORS.
    """
    if colors is None:
        colors = DEFAULT_COLORS

    if model_order is None:
        means = {tag: np.nanmean(rs) for tag, rs in boot_values.items()}
        model_order = sorted(means, key=lambda t: means[t], reverse=True)

    names, m_vals, ci_los, ci_his, bar_colors = [], [], [], [], []
    for tag in reversed(model_order):
        if tag not in boot_values:
            continue
        rs = np.array(boot_values[tag])
        rs = rs[~np.isnan(rs)]
        if len(rs) == 0:
            continue
        names.append(tag)
        m_vals.append(np.mean(rs))
        ci_los.append(np.percentile(rs, 2.5))
        ci_his.append(np.percentile(rs, 97.5))
        bar_colors.append(colors.get(tag, '#888780'))

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.5))

    ax.barh(y, m_vals, height=0.5,
            color=[c + '55' for c in bar_colors],
            edgecolor=bar_colors, linewidth=1.2)

    for i in range(len(m_vals)):
        ax.errorbar(m_vals[i], y[i],
                    xerr=[[m_vals[i] - ci_los[i]], [ci_his[i] - m_vals[i]]],
                    fmt='none', capsize=4, capthick=1.5, elinewidth=1.5,
                    ecolor=bar_colors[i])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)

    if xlabel is None:
        xlabel = 'Pearson $r$'
    ax.set_xlabel(xlabel, fontsize=11)

    if xlim is not None:
        ax.set_xlim(xlim)

    for i, (m, hi) in enumerate(zip(m_vals, ci_his)):
        ax.text(hi + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015,
                i, f'{m:.3f}', va='center', fontsize=9, color='#666')

    if title is None:
        title = 'Bootstrap 95% CIs\n(resampling participants)'
    ax.set_title(title, fontsize=11, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ceiling is not None:
    	ax.axvline(ceiling, color='#999', linestyle='--', linewidth=1)
    	ax.text(ceiling - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
            ax.get_ylim()[1] * 0.05, 'noise ceiling',
            ha='right', va='top', fontsize=9, color='#999')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


# =====================================================================
# Figure 3: Multi-experiment grouped bars
# =====================================================================

def fig_multi_experiment_bars(eval_dfs, metric="delta_R2",
                              model_order=None, colors=None, save_path=None):
    """
    Grouped bar chart: models on x-axis, one bar per experiment.
    
    eval_dfs : dict[exp_name, DataFrame]
        Each DataFrame from evaluate_fits, with 'tag' and metric columns.
    """
    if model_order is None:
        model_order = ['L0', 'L1 heuristic', 'real L1',
                       'L2 heuristic', 'real L2 (1 run only)', 'L3 heuristic']

    exp_names = list(eval_dfs.keys())
    n_models = len(model_order)
    n_exps = len(exp_names)
    exp_colors = ['#444441', '#EF9F27', '#378ADD', '#1D9E75', '#E24B4A'][:n_exps]

    data = {}
    for exp_name, df in eval_dfs.items():
        vals = dict(zip(df['tag'], df[metric]))
        data[exp_name] = vals

    x = np.arange(n_models)
    width = 0.8 / n_exps

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, exp in enumerate(exp_names):
        vals = [data[exp].get(m, 0) for m in model_order]
        offset = (i - n_exps / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=exp,
               color=exp_colors[i] + '88', edgecolor=exp_colors[i], linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=9, rotation=15, ha='right')

    if metric == "delta_R2":
        ax.set_ylabel('$\\Delta R^2$', fontsize=11)
    elif metric == "r_U_AW":
        ax.set_ylabel('Pearson $r$', fontsize=11)
    else:
        ax.set_ylabel(metric, fontsize=11)

    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Model comparison across experiments', fontsize=11, pad=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    
    
    
    
# =====================================================================
# Figure 4: Comparing between models
# =====================================================================


def fig_model_scatter(*args, **kwargs):
    """Deprecated. Use fig_model_vs_model with show_variance=False for the
    same look (or omit to get error bars)."""
    raise NotImplementedError(
        "fig_model_scatter has been removed. "
        "Use fig_model_vs_model(sims, tag_x, tag_y, show_variance=False) "
        "for the legacy look, or fig_model_vs_model(sims, tag_x, tag_y) "
        "for the new variance-aware version."
    )



###### ---
 #NEW FIGURES
###### ---
def fig_compute_time(timing_detailed, model_order=None, colors=None, save_path=None):
    if colors is None:
        colors = DEFAULT_COLORS
    if model_order is None:
        means = {t: np.mean(v) for t, v in timing_detailed.items()}
        model_order = sorted(means, key=lambda t: means[t])

    tags = [t for t in reversed(model_order) if t in timing_detailed]
    means = [np.mean(timing_detailed[t]) for t in tags]
    ci_los = [np.percentile(timing_detailed[t], 2.5) for t in tags]
    ci_his = [np.percentile(timing_detailed[t], 97.5) for t in tags]
    bar_colors = [colors.get(t, '#888780') for t in tags]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(tags) + 1.5))
    y = np.arange(len(tags))

    ax.barh(y, means, height=0.5,
            color=[c + '55' for c in bar_colors],
            edgecolor=bar_colors, linewidth=1.2)

    for i in range(len(means)):
        ax.errorbar(means[i], y[i],
                    xerr=[[means[i] - ci_los[i]], [ci_his[i] - means[i]]],
                    fmt='none', capsize=4, capthick=1.5, elinewidth=1.5,
                    ecolor=bar_colors[i])

    ax.set_yticks(y)
    ax.set_yticklabels(tags, fontsize=10)
    ax.set_xscale('log')

    baseline = np.mean(timing_detailed['L0'])
    for i, (m, hi) in enumerate(zip(means, ci_his)):
        tag = tags[i]
    # Consistent labels: always show most readable unit
        if tag == 'L0' or m <= baseline:
            if m >= 60:
                label = f'{m/60:.1f}h'
            else:
                label = f'{m:.0f}min'
        else:
            if m >= 60:
                label = f'{m/60:.1f}h ({m/baseline:.0f}x)'
            elif m >= 1:
                label = f'{m:.0f}min ({m/baseline:.0f}x)'
            else:
                label = f'{m*60:.0f}s'
        ax.text(hi * 1.15, i, label, va='center', fontsize=9, color='#666')

    ax.set_xlabel('Compute time per run (minutes, log scale)', fontsize=11)
    ax.set_title('Computational cost by model', fontsize=11, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def fig_model_vs_humans(dm_dict, dataset, tags, tag_labels=None,
                        colors=None, metric='univ_aggregate_welfare', 
                        xlim=None, ylim=None, label_threshold=None,
                        detail_level='macro', save_path=None,
                        sims=None, show_variance=True,
                        show_regression_ci=True):
    """
    Multi-panel scatter: each model's U_AW vs human judgments.
    
    detail_level: 'macro' (3 categories) or 'sub' (7 subtypes with markers)
    label_threshold: label maps where U_AW < this value (e.g., -10)
    sims: optional dict of summary DataFrames (output of load_simulations).
          Required if show_variance=True (default).
    show_variance: if True (default), draw horizontal ±SD error bars on each
                   point using the `<metric>_sd` column from sims[tag]. Falls
                   back silently to no error bars if sims is None.
    show_regression_ci: if True (default), draw a 95% CI band around the
                       regression line (Joe-style). Set False for a plain
                       dashed line.
    """
    dv = dataset.dv_col
    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    
    # Macro level: 3 categories
    macro_colors = {
        'no_line': 'tab:blue',
        'yes_line': 'tab:red',
        'maybe/other': 'tab:green',
    }
    
    # Sub level: 7 subtypes
    sub_colors = {
        'no_line_number': 'tab:blue', 'no_line_letter': 'tab:blue',
        'yes_line_number': 'tab:red', 'yes_line_letter': 'tab:red',
        'esque': 'tab:green',
        'maybe': 'tab:green', 'new_maybe': 'tab:green',
    }
    sub_markers = {
        'no_line_number': 'o', 'no_line_letter': 'x',
        'yes_line_number': 'o', 'yes_line_letter': 'x',
        'esque': 's',
        'maybe': 'o', 'new_maybe': 'x',
    }
    sub_order = ['no_line_number', 'no_line_letter', 'yes_line_number', 
                 'yes_line_letter', 'esque', 'maybe', 'new_maybe']
        
    def get_macro(name):
        name = str(name)
        if 'noline' in name or 'new_no' in name: return 'no_line'
        if 'yesline' in name or 'new_yes' in name: return 'yes_line'
        return 'maybe/other'
    
    def get_subtype(name):
        name = str(name)
        if 'new_no' in name: return 'no_line_number'
        if 'noline' in name: return 'no_line_letter'
        if name.startswith('new_yes_') and 'esque' in name: return 'esque'
        if name.startswith('new_yes_') and 'mod' in name: return 'esque'
        if 'new_yes' in name: return 'yes_line_number'
        if 'yesline' in name: return 'yes_line_letter'
        if 'new_maybe' in name: return 'new_maybe'
        if 'maybe' in name: return 'maybe'
        return 'maybe'
    
    def get_severity(name):
        name = str(name)
        if '_badder' in name: return 'badder'
        if '_bad' in name: return 'bad'
        if '_1cut' in name: return '1cut'
        return 'unknown'
        
    for ax, tag in zip(axes, tags):
        dm = dm_dict[tag]
        df = dm.dropna(subset=[metric, dv]).copy()

        # Merge in per-map SD if requested and available
        sd_col = f'{metric}_sd'
        has_sd = False
        if show_variance and sims is not None and tag in sims:
            sim_df = sims[tag]
            if sd_col in sim_df.columns and 'map' in df.columns:
                # Drop existing sd_col from df to avoid suffix clash
                if sd_col in df.columns:
                    df = df.drop(columns=[sd_col])
                df = df.merge(sim_df[['map', sd_col]], on='map', how='left')
                # Confirm the column is actually present after the merge
                has_sd = sd_col in df.columns

        if detail_level == 'sub':
            df['type'] = df['stimulus'].apply(get_subtype)
            type_list = sub_order
            type_colors = sub_colors
            type_markers = sub_markers
        else:
            df['type'] = df['stimulus'].apply(get_macro)
            type_list = ['no_line', 'yes_line', 'maybe/other']
            type_colors = macro_colors
            type_markers = {k: 'o' for k in macro_colors}
        
        r = np.corrcoef(df[metric], df[dv])[0, 1]
        
        for t in type_list:
            sub = df[df['type'] == t]
            if len(sub) > 0:
                mkr = type_markers[t]
                if has_sd:
                    ax.errorbar(sub[metric], sub[dv],
                                xerr=sub[sd_col].fillna(0),
                                fmt=mkr, color=type_colors[t],
                                ecolor=type_colors[t],
                                elinewidth=1, capsize=2, alpha=0.7,
                                markersize=6, label=t, zorder=2)
                else:
                    ax.scatter(sub[metric], sub[dv], 
                              c=type_colors[t], marker=mkr,
                              s=40, alpha=0.7, linewidth=1.5,
                              label=t, zorder=2)   

        # Regression line — with optional 95% CI band (Joe-style)
        z = np.polyfit(df[metric], df[dv], 1)
        p = np.poly1d(z)
        if show_regression_ci:
            try:
                import seaborn as sns
                sns.regplot(data=df, x=metric, y=dv, scatter=False,
                            color='#666', ci=95, ax=ax,
                            line_kws={'linewidth': 1.2, 'linestyle': '-'})
            except ImportError:
                # Fall back to plain line if seaborn unavailable
                xr = np.linspace(df[metric].min(), df[metric].max(), 100)
                ax.plot(xr, p(xr), '--', color='#999', linewidth=1)
        else:
            xr = np.linspace(df[metric].min(), df[metric].max(), 100)
            ax.plot(xr, p(xr), '--', color='#999', linewidth=1)
        
        # Label maps below threshold
        if label_threshold is not None:
            low = df[df[metric] < label_threshold]
            for _, row in low.iterrows():
                ax.annotate(row['stimulus'], (row[metric], row[dv]),
                           fontsize=6, color='#666', xytext=(4, 4), textcoords='offset points')
        else:
            # Default: label top 3 outliers from regression
            df['resid'] = np.abs(df[dv] - p(df[metric]))
            outliers = df.nlargest(3, 'resid')
            for _, row in outliers.iterrows():
                ax.annotate(row['stimulus'], (row[metric], row[dv]),
                           fontsize=6, color='#666', xytext=(4, 4), textcoords='offset points')
        
        label = tag_labels.get(tag, tag) if tag_labels else tag
        bar_color = colors.get(tag, '#888780') if colors else '#2266CC'
        ax.set_title(f'{label}\nr = {r:.2f}', fontsize=12, color=bar_color, fontweight='bold')
        ax.set_xlabel('Universalization score (U_AW)', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('Human judgment', fontsize=10)
            ax.legend(fontsize=7, frameon=False, loc='best')
        
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def fig_model_vs_model(sims, tag_x, tag_y, metric='univ_aggregate_welfare',
                       show_variance=True, error_kind='sd',
                       colors=None, xlim=None, ylim=None,
                       equal_axes=True,
                       label_threshold=None, label_top_diffs=6,
                       detail_level='macro',
                       title=None, save_path=None):
    """
    Scatter plot comparing two models' U_AW per map, with diagonal reference
    and optional run-to-run error bars.

    Args (key new ones):
        equal_axes: if True (default), x and y use the same range — diagonal
                    is meaningful at 45°. If False, each axis fits its own
                    data; useful when models live in very different ranges
                    (e.g., L0 vs L1 heuristic — L0 has extreme outliers).

    Reads ±SD from the `<metric>_sd` column already produced by
    build_universalization_summary (in build_utils.py). For SEM, also needs
    `n_valid` column (number of non-stuck runs).

    Args:
        sims: dict[tag] -> DataFrame with columns ['map', metric, f'{metric}_sd', 'n_valid']
        show_variance: draw error bars (default True). Set False for legacy look.
        error_kind: 'sd' or 'sem'. Only used if show_variance=True.
        detail_level: 'macro' or 'sub' (subtype markers).
    """
    macro_colors = {
        'no_line': 'tab:blue', 'yes_line': 'tab:red', 'maybe/other': 'tab:green',
    }
    sub_colors = {
        'no_line_number': 'tab:blue', 'no_line_letter': 'tab:blue',
        'yes_line_number': 'tab:red', 'yes_line_letter': 'tab:red',
        'esque': 'tab:green', 'maybe': 'tab:green', 'new_maybe': 'tab:green',
    }
    sub_markers = {
        'no_line_number': 'o', 'no_line_letter': 'x',
        'yes_line_number': 'o', 'yes_line_letter': 'x',
        'esque': 's', 'maybe': 'o', 'new_maybe': 'x',
    }
    sub_order = ['no_line_number', 'no_line_letter', 'yes_line_number',
                 'yes_line_letter', 'esque', 'maybe', 'new_maybe']

    def get_macro(name):
        name = str(name)
        if 'no_line' in name: return 'no_line'
        if 'yes_line' in name: return 'yes_line'
        return 'maybe/other'

    def get_subtype(name):
        name = str(name)
        if 'no_line' in name:
            return 'no_line_letter' if name.replace('no_line_', '').isalpha() else 'no_line_number'
        if 'esque' in name: return 'esque'
        if 'yes_line' in name:
            return 'yes_line_letter' if name.replace('yes_line_', '').isalpha() else 'yes_line_number'
        if 'new_maybe' in name: return 'new_maybe'
        if 'maybe' in name: return 'maybe'
        return 'maybe'

    sd_col = f'{metric}_sd'
    cols_x = ['map', metric] + ([sd_col] if sd_col in sims[tag_x].columns else [])
    cols_y = ['map', metric] + ([sd_col] if sd_col in sims[tag_y].columns else [])
    if 'n_valid' in sims[tag_x].columns: cols_x.append('n_valid')
    if 'n_valid' in sims[tag_y].columns: cols_y.append('n_valid')

    df_x = sims[tag_x][cols_x].rename(
        columns={metric: 'x_mean', sd_col: 'x_sd', 'n_valid': 'x_n'})
    df_y = sims[tag_y][cols_y].rename(
        columns={metric: 'y_mean', sd_col: 'y_sd', 'n_valid': 'y_n'})
    df = df_x.merge(df_y, on='map', how='inner')

    if detail_level == 'sub':
        df['type'] = df['map'].apply(get_subtype)
        type_list = sub_order
        type_colors = sub_colors
        type_markers = sub_markers
    else:
        df['type'] = df['map'].apply(get_macro)
        type_list = ['no_line', 'yes_line', 'maybe/other']
        type_colors = macro_colors
        type_markers = {k: 'o' for k in type_colors}

    # Compute error bars
    if show_variance and 'x_sd' in df.columns and 'y_sd' in df.columns:
        if error_kind == 'sem' and 'x_n' in df.columns and 'y_n' in df.columns:
            df['x_err'] = df['x_sd'] / np.sqrt(df['x_n'].clip(lower=1))
            df['y_err'] = df['y_sd'] / np.sqrt(df['y_n'].clip(lower=1))
            err_label = 'SEM'
        else:
            df['x_err'] = df['x_sd'].fillna(0)
            df['y_err'] = df['y_sd'].fillna(0)
            err_label = 'SD'
        draw_errorbars = True
    else:
        df['x_err'] = 0
        df['y_err'] = 0
        draw_errorbars = False

    fig, ax = plt.subplots(figsize=(6, 6))
    x_lo, x_hi = df['x_mean'].min() - 1, df['x_mean'].max() + 1
    y_lo, y_hi = df['y_mean'].min() - 1, df['y_mean'].max() + 1
    if equal_axes:
        lo = min(x_lo, y_lo)
        hi = max(x_hi, y_hi)
        x_lo, x_hi = lo, hi
        y_lo, y_hi = lo, hi
    # Diagonal reference (y = x); always drawn, but visually only at 45°
    # when equal_axes=True. Otherwise it's tilted but still represents y=x.
    diag_lo = min(x_lo, y_lo)
    diag_hi = max(x_hi, y_hi)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], '--',
            color='#CCC', linewidth=1, zorder=0)

    for t in type_list:
        sub = df[df['type'] == t]
        if len(sub) == 0:
            continue
        mkr = type_markers[t]
        if draw_errorbars:
            ax.errorbar(sub['x_mean'], sub['y_mean'],
                        xerr=sub['x_err'], yerr=sub['y_err'],
                        fmt=mkr, color=type_colors[t], ecolor=type_colors[t],
                        elinewidth=1, capsize=2, alpha=0.6, markersize=6,
                        label=t, zorder=2)
        else:
            ax.scatter(sub['x_mean'], sub['y_mean'],
                       c=type_colors[t], marker=mkr,
                       s=40, alpha=0.7, linewidth=1.5, label=t, zorder=2)

    # Label outliers (largest absolute difference between x and y means)
    df['abs_diff'] = (df['y_mean'] - df['x_mean']).abs()
    outliers = df.nlargest(label_top_diffs, 'abs_diff')
    for _, row in outliers.iterrows():
        ax.annotate(row['map'], (row['x_mean'], row['y_mean']),
                    fontsize=7, color='#666',
                    xytext=(5, 5), textcoords='offset points')

    if label_threshold is not None:
        for _, row in df.iterrows():
            if (row['x_mean'] < label_threshold or row['y_mean'] < label_threshold) \
               and row['map'] not in outliers['map'].values:
                ax.annotate(row['map'], (row['x_mean'], row['y_mean']),
                            fontsize=7, color='#666',
                            xytext=(5, 5), textcoords='offset points')

    # Label by metric: U_AW for univ_aggregate_welfare, otherwise just metric
    metric_label_map = {
        'univ_aggregate_welfare': 'U_AW',
        'mean': 'mean fill time',
        'last': 'last fill time',
        'first': 'first fill time',
        'prop_finished': 'prop finished',
    }
    metric_label = metric_label_map.get(metric, metric)
    r = np.corrcoef(df['x_mean'], df['y_mean'])[0, 1]
    ax.set_xlabel(f'{tag_x} ({metric_label})', fontsize=11)
    ax.set_ylabel(f'{tag_y} ({metric_label})', fontsize=11)
    if xlim: ax.set_xlim(xlim)
    else: ax.set_xlim(x_lo, x_hi)
    if ylim: ax.set_ylim(ylim)
    else: ax.set_ylim(y_lo, y_hi)
    if equal_axes:
        ax.set_aspect('equal')
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title is None:
        suffix = f' (±{err_label})' if draw_errorbars else ''
        title = f'{tag_x} vs {tag_y} (r = {r:.2f}){suffix}'
    ax.set_title(title, fontsize=11, pad=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return df


# =====================================================================
# Figure 5: Run-to-run variance per map (one bar per model)
# =====================================================================

def fig_run_variance_per_map(sims, model_order, colors=None,
                              metric='univ_aggregate_welfare',
                              sort_by=None, top_n=None,
                              save_path=None, figsize=(13, 5)):
    """
    Grouped bar chart: x = map, y = SD across runs, one bar per model.
    Reads `<metric>_sd` column already in summary CSV.

    Args:
        sims: dict[tag] -> DataFrame with `<metric>_sd` column
        model_order: list of tags to plot (in order)
        sort_by: tag whose SD orders the maps (descending). None = alphabetical.
        top_n: keep only the N maps with highest SD according to sort_by.
    """
    if colors is None:
        colors = DEFAULT_COLORS

    sd_col = f'{metric}_sd'
    rows = []
    for tag in model_order:
        if tag not in sims:
            continue
        if sd_col not in sims[tag].columns:
            print(f"  [warn] {tag}: no '{sd_col}' column, skipping")
            continue
        for _, r in sims[tag].iterrows():
            rows.append({'map': r['map'], 'tag': tag, 'sd': r[sd_col]})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("No SD data found.")
        return None

    if sort_by and sort_by in sims and sd_col in sims[sort_by].columns:
        order = (sims[sort_by].sort_values(sd_col, ascending=False)['map']
                 .tolist())
    else:
        order = sorted(plot_df['map'].unique())

    if top_n:
        order = order[:top_n]

    pivot = plot_df.pivot(index='map', columns='tag', values='sd').reindex(order)
    pivot = pivot[[t for t in model_order if t in pivot.columns]]

    fig, ax = plt.subplots(figsize=figsize)
    bar_colors = [colors.get(t, '#888780') for t in pivot.columns]
    pivot.plot(kind='bar', ax=ax, width=0.85, color=bar_colors,
               edgecolor='white', linewidth=0.3)

    ax.set_xlabel('Map', fontsize=10)
    ax.set_ylabel(f'SD across runs (U_AW)', fontsize=10)
    sort_str = f' — sorted by {sort_by}' if sort_by else ''
    ax.set_title(f'Run-to-run variance per map{sort_str}', fontsize=11)
    ax.legend(fontsize=9, frameon=False, ncol=len(pivot.columns))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=70, ha='right', fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return pivot


# =====================================================================
# Figure 6: Model variance summary (one bar per model, mean SD across maps)
# =====================================================================

def fig_model_variance_summary(sims, model_order, colors=None,
                                metric='univ_aggregate_welfare',
                                save_path=None, figsize=(7, 4),
                                annotate_max=True):
    """
    For each model, mean across-map SD. Shows which models are noisiest overall.
    """
    if colors is None:
        colors = DEFAULT_COLORS

    sd_col = f'{metric}_sd'
    rows = []
    for tag in model_order:
        if tag not in sims or sd_col not in sims[tag].columns:
            continue
        sds = sims[tag][sd_col].dropna()
        if len(sds) == 0:
            continue
        max_idx = sims[tag][sd_col].idxmax() if not sims[tag][sd_col].dropna().empty else None
        max_map = sims[tag].loc[max_idx, 'map'] if max_idx is not None else '?'
        rows.append({
            'tag': tag,
            'mean_sd': sds.mean(),
            'median_sd': sds.median(),
            'max_sd': sds.max(),
            'max_sd_map': max_map,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No SD data found.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    bar_colors = [colors.get(t, '#888780') for t in df['tag']]
    bars = ax.bar(df['tag'], df['mean_sd'], color=bar_colors, edgecolor='white')

    if annotate_max:
        for bar, mx, mxmap in zip(bars, df['max_sd'], df['max_sd_map']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f'max={mx:.1f}\n({mxmap})',
                    ha='center', fontsize=7, color='#666')

    ax.set_ylabel('Mean SD across runs (U_AW)', fontsize=10)
    ax.set_title('Run-to-run variance, averaged across maps', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=20, ha='right', fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return df


# =====================================================================
# Helper: variance table (text)
# =====================================================================

def variance_table(sims, model_order, metric='univ_aggregate_welfare',
                   sort_by=None, top_n=None):
    """Wide table: rows = maps, cols = models, values = SD across runs."""
    sd_col = f'{metric}_sd'
    rows = []
    for tag in model_order:
        if tag not in sims or sd_col not in sims[tag].columns:
            continue
        for _, r in sims[tag].iterrows():
            rows.append({'map': r['map'], 'tag': tag, 'sd': r[sd_col]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pivot = df.pivot(index='map', columns='tag', values='sd')
    pivot = pivot[[t for t in model_order if t in pivot.columns]]

    if sort_by and sort_by in pivot.columns:
        pivot = pivot.sort_values(sort_by, ascending=False)
    if top_n:
        pivot = pivot.head(top_n)

    return pivot.round(2)

def fig_per_run_dots_by_map(sims_or_runs, model_order, colors=None,
                             metric='univ_aggregate_welfare',
                             maps=None, sort_by=None, sort_kind='sd',
                             reference_tag=None,
                             top_n=None,
                             sim_root='../data/simulations',
                             RUNS=None,
                             ncols=4, save_path=None,
                             show_mean=True, show_sd=True,
                             jitter=0.05):
    """
    Multi-panel dot plot: every run's U_AW for each model, one panel per map.

    Args:
        model_order: list of tags shown left-to-right within each panel
        sort_kind: how to order map panels
            'sd'           — descending SD of `sort_by` model (default; noise focus)
            'mean'         — ascending mean of `sort_by` (worst-performing first)
            'gap'          — largest |mean(sort_by) - mean(reference_tag)| first
            'alphabetical' — A→Z map names
        sort_by: tag whose values drive the sort (required unless alphabetical)
        reference_tag: only for sort_kind='gap'. Compare sort_by to this model.
        top_n: how many maps to plot (default 8, or 28 = all)
        maps: explicit list overrides sort_kind/sort_by entirely

    Other args:
        sims_or_runs: sims dict (needed for sort logic)
        RUNS: dict[tag, run_label] (needed to find raw CSVs)
        sim_root: parent dir of run dirs
        ncols: panels per row
        show_mean / show_sd / jitter: visual options
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if RUNS is None:
        raise ValueError("Must pass RUNS dict to locate raw CSVs.")
    if colors is None:
        colors = DEFAULT_COLORS

    # Resolve which maps to plot
    if maps is None:
        if sort_by is None and sort_kind != 'alphabetical':
            raise ValueError("Pass either `maps=[...]`, `sort_by='tag'`, "
                             "or `sort_kind='alphabetical'`")
        if not isinstance(sims_or_runs, dict):
            raise ValueError("Need sims dict to pick maps")

        n = top_n or 8

        if sort_kind == 'alphabetical':
            # Use any model's map list, sorted alphabetically
            any_tag = sort_by or next(iter(sims_or_runs))
            maps = sorted(sims_or_runs[any_tag]['map'].tolist())[:n]
        elif sort_kind == 'sd':
            if sort_by not in sims_or_runs:
                raise ValueError(f"sims['{sort_by}'] not found")
            sd_col = f'{metric}_sd'
            maps = (sims_or_runs[sort_by]
                    .sort_values(sd_col, ascending=False)['map']
                    .head(n).tolist())
        elif sort_kind == 'mean':
            # Lowest mean (worst U_AW) first — these are "where the model fails"
            if sort_by not in sims_or_runs:
                raise ValueError(f"sims['{sort_by}'] not found")
            maps = (sims_or_runs[sort_by]
                    .sort_values(metric, ascending=True)['map']
                    .head(n).tolist())
        elif sort_kind == 'gap':
            # Largest gap |sort_by - reference| first — "where models disagree most"
            if sort_by not in sims_or_runs:
                raise ValueError(f"sims['{sort_by}'] not found")
            if reference_tag is None or reference_tag not in sims_or_runs:
                raise ValueError("sort_kind='gap' requires reference_tag in sims")
            sub = sims_or_runs[sort_by][['map', metric]].rename(
                columns={metric: 'sort_val'}).merge(
                sims_or_runs[reference_tag][['map', metric]].rename(
                    columns={metric: 'ref_val'}), on='map')
            sub['gap'] = (sub['sort_val'] - sub['ref_val']).abs()
            maps = sub.sort_values('gap', ascending=False)['map'].head(n).tolist()
        else:
            raise ValueError(f"Unknown sort_kind={sort_kind!r}; "
                             "use 'sd', 'mean', 'gap', or 'alphabetical'")

    # In summary CSVs, metrics have 'univ_' prefix. In per-run processed CSVs,
    # they don't. Strip the prefix when looking up.
    metric_in_processed = metric[5:] if metric.startswith('univ_') else metric

    # Pull per-run values from processed CSVs for each (model, map)
    rows = []
    for tag in model_order:
        if tag not in RUNS:
            continue
        label = RUNS[tag]
        for map_name in maps:
            csv_path = os.path.join(sim_root, label, 'processed', f'{map_name}.csv')
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            valid = df[~df['stuck']] if 'stuck' in df.columns else df
            if metric_in_processed not in valid.columns:
                continue
            for _, r in valid.iterrows():
                rows.append({
                    'map': map_name, 'tag': tag,
                    'run': int(r.get('run', 0)),
                    'value': r[metric_in_processed],
                })
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("No data found.")
        return plot_df

    # Layout
    n_maps = len(maps)
    nrows = (n_maps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.0 * ncols, 2.6 * nrows),
                              sharey=False, squeeze=False)
    rng = np.random.default_rng(0)  # for reproducible jitter

    for ax_idx, map_name in enumerate(maps):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        sub = plot_df[plot_df['map'] == map_name]

        for x, tag in enumerate(model_order):
            tag_sub = sub[sub['tag'] == tag]
            if len(tag_sub) == 0:
                continue
            color = colors.get(tag, '#888')
            xs = x + rng.uniform(-jitter, jitter, size=len(tag_sub))
            ax.scatter(xs, tag_sub['value'], color=color, alpha=0.7,
                       s=35, edgecolor='white', linewidth=0.4, zorder=3)
            if show_sd and len(tag_sub) >= 2:
                m = tag_sub['value'].mean()
                s = tag_sub['value'].std(ddof=1)
                ax.errorbar(x, m, yerr=s, color=color, capsize=4,
                            elinewidth=1.2, alpha=0.85, zorder=2)
            if show_mean and len(tag_sub) >= 1:
                m = tag_sub['value'].mean()
                ax.plot([x - 0.18, x + 0.18], [m, m], color=color,
                        linewidth=1.5, zorder=4)

        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=30, ha='right', fontsize=8)
        ax.set_title(map_name, fontsize=10)
        ax.set_ylabel(metric, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2, linewidth=0.5)

    # Hide unused panels
    for k in range(n_maps, nrows * ncols):
        axes[k // ncols, k % ncols].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return plot_df
