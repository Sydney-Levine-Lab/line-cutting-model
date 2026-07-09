"""
trajectory_snapshots.py
=======================

Generate a static PNG showing N timesteps side-by-side for two trajectories.
Use for slides (more reliable than embedding the HTML viewer).

Usage:
    from trajectory_snapshots import snapshots
    snapshots(map_path, log_real, log_heur,
              timesteps=[0, 5, 10, 15, 20],
              output='figures/yes_line_8_snapshots.png')
"""

import matplotlib.pyplot as plt
import numpy as np
from map_analysis import load_map
from trajectory_visualizer import parse_decisions, reconstruct_states


def _draw_state(ax, map_obj, state, diverged_set, title=None,
                 agent_to_position=None):
    """Draw one snapshot. diverged_set: set of agent indices (1-based) that
    differ from the other side. agent_to_position: dict mapping agent_id -> label
    (e.g., {7: '1', 2: '2', ...} for order-based labels). If None, use agent_id."""
    n_rows, n_cols = map_obj.walls.shape
    # walls
    ax.imshow(map_obj.walls, cmap='Greys', vmin=0, vmax=1.5,
               extent=(0.5, n_cols + 0.5, n_rows + 0.5, 0.5),
               aspect='equal', interpolation='none')
    # wells
    for x, y in map_obj.wells:
        ax.scatter(x, y, c='#4477AA', s=40, marker='s', edgecolor='white', linewidth=0.3)
    # tanks
    for x, y in map_obj.tanks:
        ax.scatter(x, y, c='#229944', s=40, marker='s', edgecolor='white', linewidth=0.3)
    # agents
    for i, (x, y) in enumerate(state['positions'], start=1):
        color = '#FFCC00' if i in diverged_set else '#CC4444'
        ax.scatter(x, y, c=color, s=130, edgecolor='black', linewidth=0.6, zorder=3)
        label = str(agent_to_position[i]) if (agent_to_position and i in agent_to_position) else str(i)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='black', zorder=4)
    ax.set_xlim(0.5, n_cols + 0.5)
    ax.set_ylim(n_rows + 0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def snapshots(map_path, log_real, log_heur, output,
              timesteps=(0, 5, 10, 15, 20),
              labels=('real L1', 'L1 heuristic'),
              order=None):
    """Static side-by-side PNG with rows=timesteps, columns=models.

    order: optional list like [7, 2, 5, 6, 3, 8, 1, 4] (agent 7 goes first).
           If given, agents are labeled by play-order position rather than id.
    """
    m = load_map(map_path)
    decs_r = parse_decisions(log_real)
    decs_h = parse_decisions(log_heur)
    states_r = reconstruct_states(decs_r, m.agents, max_t=max(timesteps))
    states_h = reconstruct_states(decs_h, m.agents, max_t=max(timesteps))
    # Pad
    while len(states_r) <= max(timesteps): states_r.append(states_r[-1])
    while len(states_h) <= max(timesteps): states_h.append(states_h[-1])

    agent_to_position = None
    if order is not None:
        agent_to_position = {agent_id: pos + 1 for pos, agent_id in enumerate(order)}

    n_rows = len(timesteps)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for row_i, t in enumerate(timesteps):
        sr = next((s for s in states_r if s['t'] == t), states_r[-1])
        sh = next((s for s in states_h if s['t'] == t), states_h[-1])
        diverged = set()
        for i, (pr, ph) in enumerate(zip(sr['positions'], sh['positions']), start=1):
            if pr != ph:
                diverged.add(i)
        title_l = f"{labels[0]} (t={t})" if row_i == 0 else f"t={t}"
        title_r = f"{labels[1]} (t={t})" if row_i == 0 else f"t={t}"
        _draw_state(axes[row_i, 0], m, sr, diverged, title=title_l,
                     agent_to_position=agent_to_position)
        _draw_state(axes[row_i, 1], m, sh, diverged, title=title_r,
                     agent_to_position=agent_to_position)

    plt.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches='tight')
    print(f"Saved {output}")
    plt.close(fig)
