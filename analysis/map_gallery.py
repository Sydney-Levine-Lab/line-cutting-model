"""
map_gallery.py
==============

Render all maps in a grid, color-coded/labeled by their real_L1 vs L1h gap.
For visually scanning for structural patterns that predict divergence.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from map_analysis import load_map, render_image


def render_map_gallery(full_table, map_dir, save_path=None, ncols=4,
                       sort_by='gap'):
    """Plot all maps in a grid, sorted by `sort_by`. Title each subplot with
    the gap and key features.
    """
    df = full_table.sort_values(sort_by, ascending=True).reset_index(drop=True)
    n_maps = len(df)
    nrows = (n_maps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.0 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        map_path = f"{map_dir}/{row['map']}.pddl"
        if not os.path.exists(map_path):
            ax.text(0.5, 0.5, f"NOT FOUND:\n{row['map']}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        m = load_map(map_path)
        # Color the title by gap severity
        gap = row.get('gap', np.nan)
        if np.isnan(gap):
            color = 'black'
        elif gap < -3:
            color = '#cc4444'
        elif gap < -1:
            color = '#cc8800'
        else:
            color = '#226622'
        fb = row.get('pct_fallback', np.nan)
        title = f"{row['map']}\ngap={gap:.1f}  fb={fb:.0f}%"
        render_image(m, ax=ax, title=title)
        ax.title.set_color(color)

    for i in range(n_maps, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    return fig
