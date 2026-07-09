"""
trajectory_visualizer.py
========================

Generate a side-by-side HTML visualization of two trajectory logs (e.g.
real_L1 vs L1 heuristic) on the same map. Opens in any browser. Has
play/step controls and highlights divergences.

Usage:
    from trajectory_visualizer import build_visualization
    build_visualization(
        map_path='maps/yes_line_8.pddl',
        log_real='trajectory_yes_line_8_run1.log',   # real_L1
        log_heur='trajectory_yes_line_8_run1.log',   # L1 heuristic
        output='visualization.html',
        labels=('real L1', 'L1 heuristic'),
    )
"""

import re
import json
from pathlib import Path
from map_analysis import load_map


DECISION_RE = re.compile(
    r'\[decision\]\s+t=(\d+)\s+idx=(\d+)\s+k=(\d+)\s+'
    r'pos=\(([0-9-]+),([0-9-]+)\)\s+'
    r'chose=([\w-]+)\(agent(\d+)\)\s+tag=(\w+)'
)

# Action name → (dx, dy)
ACTION_DELTAS = {
    'up':    (0, -1),
    'down':  (0, +1),
    'left':  (-1, 0),
    'right': (+1, 0),
    'wait':  (0, 0),
    'fill':  (0, 0),       # in-place
    'start-back': (0, 0),  # state change but not move (we don't track this either)
    'done':  (0, 0),
}


def parse_decisions(log_path):
    """Parse trajectory log, return list of {t, idx, k, pos, action, agent, tag}."""
    rows = []
    with open(log_path) as f:
        for line in f:
            m = DECISION_RE.match(line)
            if not m:
                continue
            rows.append({
                't': int(m.group(1)),
                'idx': int(m.group(2)),
                'k': int(m.group(3)),
                'x': int(m.group(4)),
                'y': int(m.group(5)),
                'action': m.group(6),
                'agent': int(m.group(7)),
                'tag': m.group(8),
            })
    return rows


def reconstruct_states(decisions, initial_positions, max_t=None):
    """Build agent positions at each timestep from the trajectory log.

    Strategy: trust the `pos=(x,y)` in each [decision] line as ground truth
    (this is the agent's actual position at the moment of decision).
    Between timesteps, an agent's position at end of timestep T is its
    position at the start of timestep T+1's earliest decision for that agent.

    Returns: list of {t: int, positions: list[(x,y)], last_actions: list}
    """
    n_agents = len(initial_positions)
    
    # Snapshot at t=0 from initial positions
    states = [{'t': 0,
                'positions': [tuple(p) for p in initial_positions],
                'last_actions': [None] * n_agents}]
    
    # Group decisions by t
    by_t = {}
    for d in decisions:
        by_t.setdefault(d['t'], []).append(d)
    
    ts_sorted = sorted(by_t.keys())
    if max_t is not None:
        ts_sorted = [t for t in ts_sorted if t <= max_t]
    
    # Track agent positions across time. Start from initial.
    current_pos = [tuple(p) for p in initial_positions]
    
    for t in ts_sorted:
        decs = sorted(by_t[t], key=lambda d: d['idx'])
        last_actions = [None] * n_agents
        for d in decs:
            agent = d['agent']
            if agent < 1 or agent > n_agents:
                continue
            # Use the logged position as ground truth for this agent at this t
            current_pos[agent-1] = (d['x'], d['y'])
            # Then apply the action to predict next position
            dx, dy = ACTION_DELTAS.get(d['action'], (0, 0))
            current_pos[agent-1] = (d['x'] + dx, d['y'] + dy)
            last_actions[agent-1] = d['action']
        states.append({
            't': t,
            'positions': [tuple(p) for p in current_pos],
            'last_actions': last_actions,
        })
    return states


def build_visualization(map_path, log_real, log_heur, output,
                        labels=('Real L1', 'L1 heuristic'),
                        max_t=None, order=None):
    """Generate a self-contained HTML file with side-by-side trajectory animation.

    order: optional list like [7, 2, 5, 6, 3, 8, 1, 4] meaning agent 7 goes first,
           agent 2 second, etc. If given, agents are LABELED by their position
           in the order (1 = first mover, 8 = last mover).
    """
    map_obj = load_map(map_path)
    initial_positions = map_obj.agents

    real_decs = parse_decisions(log_real)
    heur_decs = parse_decisions(log_heur)

    states_real = reconstruct_states(real_decs, initial_positions, max_t)
    states_heur = reconstruct_states(heur_decs, initial_positions, max_t)

    # Pad shorter sequence with last state repeated
    max_len = max(len(states_real), len(states_heur))
    while len(states_real) < max_len:
        states_real.append({**states_real[-1], 't': states_real[-1]['t']})
    while len(states_heur) < max_len:
        states_heur.append({**states_heur[-1], 't': states_heur[-1]['t']})

    # Mark agents whose position differs between sides at each timestep
    for sr, sh in zip(states_real, states_heur):
        diverged = []
        for i, (pr, ph) in enumerate(zip(sr['positions'], sh['positions'])):
            if pr != ph:
                diverged.append(i + 1)
        sr['diverged'] = diverged
        sh['diverged'] = diverged

    # Walls as list of (x, y) cells
    wall_cells = []
    for r in range(map_obj.n_rows):
        for c in range(map_obj.n_cols):
            if map_obj.walls[r, c]:
                wall_cells.append([c + 1, r + 1])

    map_data = {
        'name': map_obj.name,
        'n_rows': map_obj.n_rows,
        'n_cols': map_obj.n_cols,
        'walls': wall_cells,
        'tanks': [list(t) for t in map_obj.tanks],
        'wells': [list(w) for w in map_obj.wells],
        'n_agents': len(initial_positions),
    }

    # Build agent_id -> play-order-position map if order is given
    agent_to_position = None
    if order is not None:
        agent_to_position = {agent_id: str(pos + 1)
                              for pos, agent_id in enumerate(order)}

    sim_data = {
        'map': map_data,
        'states_a': states_real,
        'states_b': states_heur,
        'label_a': labels[0],
        'label_b': labels[1],
        'agent_to_position': agent_to_position,
    }

    html = _build_html(sim_data)
    Path(output).write_text(html)
    print(f"Saved {output}")
    print(f"Open in browser. {len(states_real)} timesteps loaded.")


def _build_html(sim_data):
    data_json = json.dumps(sim_data)
    return r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Trajectory comparison</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #f8f8f8; }
h1 { font-size: 18px; margin: 0 0 10px 0; }
.panels { display: flex; gap: 30px; }
.panel { background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.panel h2 { font-size: 14px; margin: 0 0 10px 0; }
.controls { margin: 10px 0; }
button { padding: 6px 14px; margin-right: 6px; font-size: 13px; cursor: pointer;
         background: #4477AA; color: white; border: none; border-radius: 4px; }
button:hover { background: #2D5A8A; }
button:disabled { background: #ccc; cursor: not-allowed; }
.status { font-family: monospace; font-size: 12px; margin: 8px 0; color: #444; }
canvas { background: white; border: 1px solid #ddd; }
.divergence { color: #cc4444; font-weight: bold; }
.legend { font-size: 11px; margin-top: 10px; color: #666; }
.legend span { display: inline-block; width: 12px; height: 12px; margin-right: 4px;
               vertical-align: middle; border-radius: 2px; }
</style>
</head>
<body>

<h1 id="title">Trajectory comparison</h1>
<div class="controls">
  <button id="playBtn">Play</button>
  <button id="stepBtn">Step</button>
  <button id="backBtn">Back</button>
  <button id="resetBtn">Reset</button>
  <label style="margin-left:20px; font-size:12px;">
    Speed:
    <input type="range" id="speedSlider" min="100" max="2000" value="500" step="100" style="vertical-align:middle;">
    <span id="speedVal" style="font-family:monospace;">500ms</span>
  </label>
</div>
<div class="status" id="status">t=0</div>

<div class="panels">
  <div class="panel">
    <h2 id="labelA"></h2>
    <canvas id="canvasA" width="500" height="500"></canvas>
    <div class="legend">
      <span style="background:#222"></span>walls
      <span style="background:#4477AA; margin-left:10px;"></span>wells
      <span style="background:#229944; margin-left:10px;"></span>tanks
      <span style="background:#CC4444; margin-left:10px;"></span>agents
      <span style="background:#FFCC00; margin-left:10px;"></span>diverged from other side
    </div>
  </div>

  <div class="panel">
    <h2 id="labelB"></h2>
    <canvas id="canvasB" width="500" height="500"></canvas>
  </div>
</div>

<script>
const data = __DATA__;
const map = data.map;
const statesA = data.states_a;
const statesB = data.states_b;

document.getElementById('labelA').textContent = data.label_a;
document.getElementById('labelB').textContent = data.label_b;
document.getElementById('title').textContent =
    'Map: ' + map.name + ' — ' + statesA.length + ' steps';

// Drawing setup
const canvasA = document.getElementById('canvasA');
const canvasB = document.getElementById('canvasB');
const cellSize = Math.floor(Math.min(canvasA.width / map.n_cols,
                                       canvasA.height / map.n_rows));
const offX = (canvasA.width - cellSize * map.n_cols) / 2;
const offY = (canvasA.height - cellSize * map.n_rows) / 2;

function cellRect(x, y) {
    return { x: offX + (x - 1) * cellSize, y: offY + (y - 1) * cellSize, w: cellSize, h: cellSize };
}

function drawMap(ctx, state) {
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, canvasA.width, canvasA.height);

    // grid
    ctx.strokeStyle = '#eee';
    ctx.lineWidth = 1;
    for (let c = 0; c <= map.n_cols; c++) {
        ctx.beginPath();
        ctx.moveTo(offX + c * cellSize, offY);
        ctx.lineTo(offX + c * cellSize, offY + map.n_rows * cellSize);
        ctx.stroke();
    }
    for (let r = 0; r <= map.n_rows; r++) {
        ctx.beginPath();
        ctx.moveTo(offX, offY + r * cellSize);
        ctx.lineTo(offX + map.n_cols * cellSize, offY + r * cellSize);
        ctx.stroke();
    }

    // walls
    ctx.fillStyle = '#222';
    for (const [x, y] of map.walls) {
        const r = cellRect(x, y);
        ctx.fillRect(r.x, r.y, r.w, r.h);
    }
    // wells
    ctx.fillStyle = '#4477AA';
    for (const [x, y] of map.wells) {
        const r = cellRect(x, y);
        ctx.fillRect(r.x + 2, r.y + 2, r.w - 4, r.h - 4);
    }
    // tanks
    ctx.fillStyle = '#229944';
    for (const [x, y] of map.tanks) {
        const r = cellRect(x, y);
        ctx.fillRect(r.x + 2, r.y + 2, r.w - 4, r.h - 4);
    }
    // agents
    state.positions.forEach((pos, i) => {
        const [x, y] = pos;
        const r = cellRect(x, y);
        const isDiverged = state.diverged.includes(i + 1);
        ctx.fillStyle = isDiverged ? '#FFCC00' : '#CC4444';
        ctx.beginPath();
        ctx.arc(r.x + r.w/2, r.y + r.h/2, r.w * 0.4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.font = 'bold ' + Math.floor(r.w * 0.5) + 'px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        // Label by play-order position if provided; otherwise by agent id.
        const label = (data.agent_to_position && data.agent_to_position[i + 1])
                       ? data.agent_to_position[i + 1] : String(i + 1);
        ctx.fillText(label, r.x + r.w/2, r.y + r.h/2);
    });
}

let step = 0;
let playing = false;
let playInterval = null;
let speed = 500;

function render() {
    const ctxA = canvasA.getContext('2d');
    const ctxB = canvasB.getContext('2d');
    drawMap(ctxA, statesA[step]);
    drawMap(ctxB, statesB[step]);
    const sA = statesA[step];
    const nDiverged = sA.diverged.length;
    let html = 't=' + sA.t + ' (step ' + step + '/' + (statesA.length - 1) + ')';
    if (nDiverged > 0) {
        html += ' — <span class="divergence">' + nDiverged +
                ' agent(s) diverged: ' + sA.diverged.join(', ') + '</span>';
    }
    document.getElementById('status').innerHTML = html;
}

function stepForward() {
    if (step < statesA.length - 1) {
        step++;
        render();
    } else {
        pause();
    }
}

function stepBack() {
    if (step > 0) {
        step--;
        render();
    }
}

function play() {
    if (playing) return;
    playing = true;
    document.getElementById('playBtn').textContent = 'Pause';
    playInterval = setInterval(() => {
        if (step < statesA.length - 1) {
            step++;
            render();
        } else {
            pause();
        }
    }, speed);
}

function pause() {
    playing = false;
    document.getElementById('playBtn').textContent = 'Play';
    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
    }
}

document.getElementById('playBtn').onclick = () => { playing ? pause() : play(); };
document.getElementById('stepBtn').onclick = () => { pause(); stepForward(); };
document.getElementById('backBtn').onclick = () => { pause(); stepBack(); };
document.getElementById('resetBtn').onclick = () => { pause(); step = 0; render(); };
document.getElementById('speedSlider').oninput = (e) => {
    speed = parseInt(e.target.value);
    document.getElementById('speedVal').textContent = speed + 'ms';
    if (playing) { pause(); play(); }
};

render();
</script>

</body>
</html>
""".replace('__DATA__', data_json)
