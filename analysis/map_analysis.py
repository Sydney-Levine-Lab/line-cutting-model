"""
map_analysis.py
===============

Parse PDDL map files, render them, compute structural features.

Use:
    from map_analysis import load_map, render_ascii, structural_features

    m = load_map('maps/yes_line_8.pddl')
    print(render_ascii(m))
    print(structural_features(m))
"""

import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MapInfo:
    name: str
    walls: np.ndarray             # (n_rows, n_cols), 1 = wall
    agents: List[Tuple[int, int]]  # [(x, y)] for agent1..N (1-indexed)
    tanks:  List[Tuple[int, int]]
    wells:  List[Tuple[int, int]]

    @property
    def n_rows(self):
        return self.walls.shape[0]

    @property
    def n_cols(self):
        return self.walls.shape[1]


def _split_init_and_goal(text):
    """Return only the :init body; drops :goal where agents have repeated xloc/yloc."""
    init_match = re.search(r'\(:init(.+?)(?=\(:goal)', text, re.DOTALL)
    if init_match:
        return init_match.group(1)
    return text


def _parse_walls(init_text):
    # Find the walls block start; then iteratively collect bit-vec entries until walls block closes
    walls_start = init_text.find('(= (walls)')
    if walls_start == -1:
        # try alternate formatting
        walls_start = init_text.find('(=(walls)')
    # Scan forward until we find the matching outer ')'
    pos = walls_start + len('(= (walls)')
    # Now we have an unbalanced 1 open paren; walk to find when we balance back to 0
    depth = 1
    end = pos
    while depth > 0 and end < len(init_text):
        ch = init_text[end]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        end += 1
    body = init_text[pos:end]
    rows = re.findall(r'\(bit-vec\s+([0-9\s]+?)\)', body)
    matrix = []
    for r in rows:
        bits = [int(b) for b in r.split()]
        matrix.append(bits)
    return np.array(matrix, dtype=int)


def _parse_positions(init_text, prefix='agent'):
    """Return list indexed by agent number (1..N) of (x, y)."""
    pattern = re.compile(
        rf'\(=\s*\(xloc\s+({prefix}\d+)\)\s+(\d+)\)\s*'
        rf'\(=\s*\(yloc\s+\1\)\s+(\d+)\)')
    rows = {}
    for m in pattern.finditer(init_text):
        name = m.group(1)
        idx = int(re.search(r'\d+', name).group())
        x = int(m.group(2)); y = int(m.group(3))
        rows[idx] = (x, y)
    # Return ordered list by index
    return [rows[i] for i in sorted(rows.keys())]


def load_map(path):
    """Parse a PDDL map file into MapInfo."""
    path = Path(path)
    text = path.read_text()
    init = _split_init_and_goal(text)
    walls = _parse_walls(init)
    agents = _parse_positions(init, 'agent')
    tanks  = _parse_positions(init, 'tank')
    wells  = _parse_positions(init, 'well')
    return MapInfo(
        name=path.stem,
        walls=walls,
        agents=agents,
        tanks=tanks,
        wells=wells,
    )


def render_ascii(m: MapInfo, agent_size=1):
    """ASCII render: # for wall, T tank, W well, digit for agent."""
    grid = np.full((m.n_rows, m.n_cols), '.', dtype='<U1')
    grid[m.walls.astype(bool)] = '#'
    for (x, y) in m.tanks:
        if 1 <= y <= m.n_rows and 1 <= x <= m.n_cols:
            grid[y-1, x-1] = 'T'
    for (x, y) in m.wells:
        if 1 <= y <= m.n_rows and 1 <= x <= m.n_cols:
            grid[y-1, x-1] = 'W'
    for i, (x, y) in enumerate(m.agents, start=1):
        if 1 <= y <= m.n_rows and 1 <= x <= m.n_cols:
            grid[y-1, x-1] = str(i) if i < 10 else 'A'
    return '\n'.join(''.join(row) for row in grid)


def render_image(m: MapInfo, ax=None, agent_color='tab:red',
                  well_color='tab:blue', tank_color='tab:green',
                  show_labels=True, title=None):
    """Render with matplotlib. Returns axes."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.4*m.n_cols+1, 0.4*m.n_rows+1))

    # Walls as black squares
    ax.imshow(m.walls, cmap='Greys', vmin=0, vmax=1.5,
               extent=(0.5, m.n_cols+0.5, m.n_rows+0.5, 0.5),
               aspect='equal', interpolation='none')

    for x, y in m.wells:
        ax.scatter(x, y, c=well_color, s=80, marker='s',
                   edgecolor='white', linewidth=0.5)
    for x, y in m.tanks:
        ax.scatter(x, y, c=tank_color, s=80, marker='s',
                   edgecolor='white', linewidth=0.5)
    for i, (x, y) in enumerate(m.agents, start=1):
        ax.scatter(x, y, c=agent_color, s=120, edgecolor='white', linewidth=0.5,
                   zorder=3)
        if show_labels:
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white', zorder=4)

    ax.set_xlim(0.5, m.n_cols + 0.5)
    ax.set_ylim(m.n_rows + 0.5, 0.5)  # y inverted
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

def _floodfill_distances(walls, sources):
    """BFS distances from any of `sources` (list of (x,y)) to every free cell."""
    n_rows, n_cols = walls.shape
    dist = np.full((n_rows, n_cols), -1, dtype=int)
    from collections import deque
    q = deque()
    for x, y in sources:
        if 1 <= y <= n_rows and 1 <= x <= n_cols and not walls[y-1, x-1]:
            dist[y-1, x-1] = 0
            q.append((y-1, x-1))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols and not walls[nr, nc] \
               and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _passage_widths(walls):
    """For each free cell, count free-cell neighbors (up/down/left/right).
    A 'narrow passage' is a free cell with only 2 free neighbors, both along
    one axis (corridor cell). Returns array of (n_free_neighbors, in_corridor).
    """
    n_rows, n_cols = walls.shape
    n_free_neighbors = np.zeros_like(walls, dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            if walls[r, c]:
                continue
            count = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols and not walls[nr, nc]:
                    count += 1
            n_free_neighbors[r, c] = count
    return n_free_neighbors


def structural_features(m: MapInfo):
    """Compute structural features that might predict model divergence."""
    walls = m.walls
    n_rows, n_cols = walls.shape
    n_cells = n_rows * n_cols
    n_walls = int(walls.sum())
    n_free = n_cells - n_walls

    # Distance from each agent to the nearest well
    dist = _floodfill_distances(walls, m.wells)
    agent_dists = []
    for x, y in m.agents:
        if 1 <= y <= n_rows and 1 <= x <= n_cols:
            d = dist[y-1, x-1]
            agent_dists.append(d if d >= 0 else np.inf)

    # Passage widths
    n_free_neighbors = _passage_widths(walls)
    # Count corridor cells (only 2 free neighbors) AMONG cells on paths between agents and wells
    # Simpler: just total corridor cells in the free region
    free_mask = ~walls.astype(bool)
    corridor_mask = (n_free_neighbors == 2) & free_mask
    pinch_mask = (n_free_neighbors == 1) & free_mask  # dead ends
    n_corridor = int(corridor_mask.sum())
    n_pinch = int(pinch_mask.sum())

    # 'Has a line' heuristic: count cells where agents are clearly tightly packed
    # = adjacency of agent positions
    agent_positions = set(m.agents)
    n_adj_agents = 0
    for x, y in m.agents:
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            if (x+dx, y+dy) in agent_positions:
                n_adj_agents += 1
    n_adj_agents //= 2  # each pair counted twice

    # Density: fraction of cells that are walls, in the convex region the agents inhabit
    # Simpler: just the raw wall density
    wall_density = n_walls / n_cells

    return {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'wall_density': round(wall_density, 3),
        'n_corridors': n_corridor,
        'n_deadends': n_pinch,
        'mean_dist_to_water': round(float(np.mean(agent_dists)), 1),
        'max_dist_to_water': max(agent_dists) if agent_dists else None,
        'n_adj_agent_pairs': n_adj_agents,
    }
