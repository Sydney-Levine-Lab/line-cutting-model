"""
Paper figures for LEVEL×DEPTH simulation results.
"""

import numpy as np
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
    
    
def fig_model_scatter(sims, tag_x, tag_y, metric='univ_aggregate_welfare',
                      title=None, save_path=None):
    """
    Scatter plot comparing two models' U_AW per map.
    Points above diagonal = tag_y is better (more negative = worse universalization).
    """
    df_x = sims[tag_x][['map', metric]].rename(columns={metric: tag_x})
    df_y = sims[tag_y][['map', metric]].rename(columns={metric: tag_y})
    df = df_x.merge(df_y, on='map', how='inner')

    # Color by map type
    def map_type(name):
        if 'no_line' in name: return 'no_line'
        if 'yes_line' in name: return 'yes_line'
        if 'esque' in name: return 'esque'
        if 'maybe' in name or 'new_maybe' in name: return 'maybe'
        return 'other'

    df['type'] = df['map'].apply(map_type)
    type_colors = {
        'no_line': '#1D9E75',
        'yes_line': '#E24B4A',
        'maybe': '#EF9F27',
        'esque': '#378ADD',
        'other': '#888780',
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    # Diagonal reference
    all_vals = np.concatenate([df[tag_x].values, df[tag_y].values])
    lo, hi = all_vals.min() - 1, all_vals.max() + 1
    ax.plot([lo, hi], [lo, hi], '--', color='#CCC', linewidth=1, zorder=0)

    # Plot each type
    for t, color in type_colors.items():
        sub = df[df['type'] == t]
        if len(sub) == 0:
            continue
        ax.scatter(sub[tag_x], sub[tag_y], c=color, s=40, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label=t, zorder=2)

    # Label outliers (5 maps with biggest absolute difference)
    df['diff'] = df[tag_y] - df[tag_x]
    df['abs_diff'] = df['diff'].abs()
    outliers = df.nlargest(5, 'abs_diff')
    for _, row in outliers.iterrows():
        ax.annotate(row['map'], (row[tag_x], row[tag_y]),
                    fontsize=7, color='#666', 
                    xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel(f'{tag_x} (U_AW)', fontsize=11)
    ax.set_ylabel(f'{tag_y} (U_AW)', fontsize=11)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title is None:
        title = f'{tag_x} vs {tag_y}'
    ax.set_title(title, fontsize=11, pad=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()



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
                        detail_level='macro', save_path=None):
    """
    Multi-panel scatter: each model's U_AW vs human judgments.
    
    detail_level: 'macro' (3 categories) or 'sub' (7 subtypes with markers)
    label_threshold: label maps where U_AW < this value (e.g., -10)
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
                ax.scatter(sub[metric], sub[dv], 
                          c=type_colors[t], marker=mkr,
                          s=40, alpha=0.7, linewidth=1.5,
                          label=t, zorder=2)   

        # Regression line
        z = np.polyfit(df[metric], df[dv], 1)
        p = np.poly1d(z)
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
                       colors=None, xlim=None, ylim=None, 
                       label_threshold=None, detail_level='macro',
                       title=None, save_path=None):
    """
    Scatter plot comparing two models' U_AW per map, with diagonal reference.
    Uses same map type coloring as fig_model_vs_humans.
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
        
    df_x = sims[tag_x][['map', metric]].rename(columns={metric: tag_x})
    df_y = sims[tag_y][['map', metric]].rename(columns={metric: tag_y})
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
        type_markers = {k: 'o' for k in macro_colors}

    fig, ax = plt.subplots(figsize=(6, 6))

    # Diagonal reference
    all_vals = np.concatenate([df[tag_x].values, df[tag_y].values])
    lo, hi = all_vals.min() - 1, all_vals.max() + 1
    ax.plot([lo, hi], [lo, hi], '--', color='#CCC', linewidth=1, zorder=0)

    for t in type_list:
        sub = df[df['type'] == t]
        if len(sub) > 0:
            mkr = type_markers[t]
            ax.scatter(sub[tag_x], sub[tag_y], c=type_colors[t], marker=mkr,
                       s=40, alpha=0.7, linewidth=1.5, label=t, zorder=2)

    # Label outliers
    df['abs_diff'] = (df[tag_y] - df[tag_x]).abs()
    outliers = df.nlargest(6, 'abs_diff')
    for _, row in outliers.iterrows():
        ax.annotate(row['map'], (row[tag_x], row[tag_y]),
                    fontsize=7, color='#666',
                    xytext=(5, 5), textcoords='offset points')

    if label_threshold is not None:
        for col in [tag_x, tag_y]:
            low = df[df[col] < label_threshold]
            for _, row in low.iterrows():
                if row['map'] not in outliers['map'].values:
                    ax.annotate(row['map'], (row[tag_x], row[tag_y]),
                                fontsize=7, color='#666',
                                xytext=(5, 5), textcoords='offset points')

    r = np.corrcoef(df[tag_x], df[tag_y])[0, 1]

    ax.set_xlabel(f'{tag_x} (U_AW)', fontsize=11)
    ax.set_ylabel(f'{tag_y} (U_AW)', fontsize=11)
    if xlim: ax.set_xlim(xlim)
    else: ax.set_xlim(lo, hi)
    if ylim: ax.set_ylim(ylim)
    else: ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title is None:
        title = f'{tag_x} vs {tag_y} (r = {r:.2f})'
    ax.set_title(title, fontsize=11, pad=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()

