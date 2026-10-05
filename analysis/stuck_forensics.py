"""
stuck_forensics.py
==================

Answers, from trajectory logs alone: when a depth-2 run gets "stuck",
what is actually happening?

  A. TRUE DEADLOCK  — stranded agents frozen on one cell, run killed by
                      the 3-identical-states detector ("(stuck)" marker,
                      ends early).
  B. OSCILLATION    — agents ping-pong between 2 cells. The detector
                      never sees three identical states, so the run
                      grinds to TIME_MAX. (Julien's hypothesis 3: the
                      detector's blind spot.)
  C. LAG            — agents keep moving/making progress but too slowly;
                      would maybe finish with more time.

And for the mutual-hesitation hypothesis: in the final window, are the
stranded agents choosing `wait` with tag=OK (confident hesitation — the
rollout says advancing is -Inf) or falling back (tag=FALLBACK)?

Usage
-----
    # single run (auto-finds the CSV next to the trajectories dir):
    python analysis/stuck_forensics.py \
        data/simulations/nl2_ghost_*/raw/trajectories/trajectory_no_line_2_run1.log

    # every log in a run directory:
    python analysis/stuck_forensics.py "data/simulations/y8_new_*/raw/trajectories/*.log"

    # wider final window:
    python analysis/stuck_forensics.py --window 25 <logs...>

Works on both old-script (ghost, random order) and new-script (zap,
fixed order) summary logs. Old logs keep emitting decisions for filled
"ghost" agents; the script uses the CSV fill times, when found, to tell
finished agents from stranded ones.
"""

import argparse
import csv
import glob
import re
import sys
from collections import Counter
from pathlib import Path

DECISION_RE = re.compile(
    r'\[decision\]\s+t=(\d+)\s+idx=(\d+)\s+k=(\d+)\s+'
    r'pos=\(([0-9-]+),([0-9-]+)\)\s+'
    r'chose=([\w-]+)\(agent(\d+)\)\s+tag=(\w+)'
)
ZAP_RE = re.compile(r'\[zap\]\s+t=(\d+)\s+removed=\[([0-9,\s]*)\]')
HEADER_RE = re.compile(r'#\s*map=(\S+)\s+run=(\d+)')
STUCK_MARK_RE = re.compile(r'#\s*t=(\d+)\s*\(stuck\)')
FINAL_MARK_RE = re.compile(r'#\s*t=(\d+)\s*\(final\)')


def parse_log(path):
    decisions, zaps = [], {}
    meta = {"map": None, "run": None, "stuck_marker_t": None, "final_marker_t": None}
    with open(path) as f:
        for line in f:
            m = DECISION_RE.match(line)
            if m:
                decisions.append({
                    "t": int(m.group(1)), "idx": int(m.group(2)),
                    "k": int(m.group(3)),
                    "x": int(m.group(4)), "y": int(m.group(5)),
                    "action": m.group(6), "agent": int(m.group(7)),
                    "tag": m.group(8),
                })
                continue
            m = ZAP_RE.match(line)
            if m:
                agents = [int(a) for a in m.group(2).replace(" ", "").split(",") if a]
                zaps[int(m.group(1))] = agents
                continue
            m = HEADER_RE.match(line)
            if m:
                meta["map"] = m.group(1).replace(".pddl", "")
                meta["run"] = int(m.group(2))
                continue
            m = STUCK_MARK_RE.match(line)
            if m:
                meta["stuck_marker_t"] = int(m.group(1))
                continue
            m = FINAL_MARK_RE.match(line)
            if m:
                meta["final_marker_t"] = int(m.group(1))
    return decisions, zaps, meta


def find_csv_row(log_path, meta):
    """Find the raw CSV next to the trajectories dir; return the row for this run."""
    p = Path(log_path)
    raw_dir = p.parent.parent  # .../raw/trajectories/ -> .../raw/
    if meta["map"] is None:
        return None, None
    csv_path = raw_dir / f'{meta["map"]}.csv'
    if not csv_path.exists():
        return None, None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if meta["run"] is None or int(row.get("run", -999)) == meta["run"]:
                fills = {i: int(row[f"agent_{i}"]) for i in range(1, 9)
                         if f"agent_{i}" in row}
                time_max = int(row["time_max"]) if "time_max" in row else None
                return fills, time_max
    return None, None


def classify_positions(positions):
    """positions: ordered list of (x,y) over the final window, one per timestep."""
    uniq = list(dict.fromkeys(positions))
    if len(uniq) == 1:
        return "FROZEN"
    if len(uniq) == 2:
        # strict A-B-A-B alternation?
        expect = [uniq[i % 2] for i in range(len(positions))]
        expect2 = [uniq[(i + 1) % 2] for i in range(len(positions))]
        if positions == expect or positions == expect2:
            return "OSCILLATING(2)"
        return "CONFINED(2)"
    if len(uniq) <= 3:
        return f"CONFINED({len(uniq)})"
    return "MOVING"


def analyse(path, window=15):
    decisions, zaps, meta = parse_log(path)
    if not decisions:
        print(f"{path}: no [decision] lines (TRAJECTORY_LEVEL=none?)")
        return None
    fills, time_max = find_csv_row(path, meta)

    last_t = max(d["t"] for d in decisions)
    zapped = {a for agents in zaps.values() for a in agents}

    # Which agents never finished?
    if fills is not None:
        stranded = sorted(a for a, ft in fills.items() if ft <= 0)
        finished = sorted(a for a, ft in fills.items() if ft > 0)
    else:
        # No CSV: for zap logs, unfinished = never zapped. For ghost logs
        # (no zaps at all), we can't tell finished ghosts from stranded
        # agents — report unknown.
        if zaps:
            all_agents = sorted({d["agent"] for d in decisions})
            stranded = sorted(set(all_agents) - zapped)
            finished = sorted(zapped)
        else:
            stranded, finished = None, None

    # End reason
    if meta["stuck_marker_t"] is not None:
        end = f"DETECTOR ('(stuck)' at t={meta['stuck_marker_t']})"
    elif time_max is not None and last_t >= time_max - 1:
        end = f"TIME_MAX (t={last_t} of {time_max}) — detector never fired"
    elif stranded == []:
        end = f"COMPLETED (t={last_t})"
    else:
        end = f"log ends at t={last_t}"

    # Final-window per-agent behaviour
    lo = max(1, last_t - window + 1)
    by_agent = {}
    for d in decisions:
        if lo <= d["t"] <= last_t:
            by_agent.setdefault(d["agent"], []).append(d)

    print(f"\n=== {path}")
    print(f"map={meta['map']} run={meta['run']}  last_t={last_t}  "
          f"decisions={len(decisions)}  zaps at t={sorted(zaps) if zaps else 'none (ghost log)'}")
    if fills:
        print(f"fill times: {fills}")
    if stranded is not None:
        print(f"finished: {finished}   STRANDED: {stranded if stranded else 'none'}")
    print(f"end: {end}")

    fallback_total = sum(1 for d in decisions if d["tag"] == "FALLBACK")
    print(f"fallback overall: {fallback_total}/{len(decisions)} "
          f"({100*fallback_total/len(decisions):.1f}%)")

    focus = stranded if stranded else sorted(by_agent)
    print(f"final window (t={lo}..{last_t}):")
    verdicts = []
    for a in focus:
        ds = sorted(by_agent.get(a, []), key=lambda d: d["t"])
        if not ds:
            print(f"  agent {a}: no decisions in window (zapped earlier)")
            continue
        pos = [(d["x"], d["y"]) for d in ds]
        cls = classify_positions(pos)
        acts = Counter(d["action"] for d in ds)
        tags = Counter(d["tag"] for d in ds)
        verdicts.append(cls)
        star = "  <-- STRANDED" if (stranded and a in stranded) else ""
        print(f"  agent {a}: {cls:14s} acts={dict(acts)} tags={dict(tags)}{star}")

    # Headline
    if stranded:
        vc = Counter(verdicts)
        if vc.get("FROZEN", 0) == len(stranded):
            head = "TRUE DEADLOCK — stranded agents frozen"
        elif any(v.startswith("OSCILLATING") for v in verdicts):
            head = "OSCILLATION — detector blind spot (period-2), run burned to the end"
        elif all(v.startswith(("CONFINED", "FROZEN", "OSCILLATING")) for v in verdicts):
            head = "CONFINED — trapped in a small pocket, not strictly frozen"
        else:
            head = "LAG — stranded agents still moving; more time might finish them"
        print(f"VERDICT: {head}")
        # mutual-hesitation check
        wait_ok = sum(1 for a in stranded for d in by_agent.get(a, [])
                      if d["action"] == "wait" and d["tag"] == "OK")
        n_str = sum(len(by_agent.get(a, [])) for a in stranded)
        if n_str:
            print(f"mutual-hesitation check: stranded agents chose wait+OK on "
                  f"{wait_ok}/{n_str} final-window decisions "
                  f"({100*wait_ok/n_str:.0f}%)"
                  + ("  <-- confident hesitation" if wait_ok > n_str/2 else ""))
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+",
                   help="trajectory log paths or globs (quote globs!)")
    p.add_argument("--window", type=int, default=15,
                   help="final-window size in timesteps (default 15)")
    args = p.parse_args(argv)

    paths = []
    for pat in args.logs:
        hits = glob.glob(pat)
        paths.extend(hits if hits else [pat])

    if not paths:
        print("No logs found.")
        return 1
    for path in sorted(paths):
        try:
            analyse(path, window=args.window)
        except FileNotFoundError:
            print(f"{path}: not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
