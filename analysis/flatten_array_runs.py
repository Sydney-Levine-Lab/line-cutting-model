"""
flatten_array_runs.py
=====================

The cluster job array (cluster/run_array.sbatch) writes ONE CELL PER TASK:

    data/simulations/{TAG}/{map}_run{N}/raw/{map}.csv     (1 data row each)

The analysis pipeline (build_utils.py, new_comparison_utils.py) expects the
older single-job layout:

    data/simulations/{run_label}/raw/{map}.csv            (N data rows)

This script reshapes the former into the latter, so that everything
downstream — get_universalization_summary(), load_simulations(),
build_design_matrix(), cross_map_analysis — runs unchanged.

Usage
-----
From the repo root:

    python analysis/flatten_array_runs.py realL1_fixedorder_zap
    python analysis/flatten_array_runs.py L0_fixedorder_zap L1h_fixedorder_zap

    # write to a different name than the source tag
    python analysis/flatten_array_runs.py realL1_fixedorder_zap --out realL1_flat

    # check completeness without writing anything
    python analysis/flatten_array_runs.py realL1_fixedorder_zap --dry-run

Then, in the notebook, exactly as before:

    sims = load_simulations({
        "L0":           "L0_fixedorder_zap_flat",
        "L1 heuristic": "L1h_fixedorder_zap_flat",
        "real L1":      "realL1_fixedorder_zap_flat",
    }, recompute=True)

Notes
-----
- Cells are sorted by the `run` column, which main.jl sets to
  RUN_OFFSET + 1. A cell with RUN_OFFSET=2 therefore has run=3, matching
  what run 3 of a flat 5-run job would have produced. Seeds match too
  (see seed_for() in src/main.jl).
- Missing cells (preempted, not yet finished) are reported, not silently
  dropped. A map with 3/5 runs still gets a flat CSV with 3 rows;
  build_utils averages over whatever is there. **Check the report before
  trusting map-level means** — unequal n across maps is a real problem
  for the ΔR² comparison.
- Stuck runs (any agent with completion time <= 0) are counted here for
  visibility, but excluded downstream by build_processed_sim_one_map().
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


AGENT_COLS = [f"agent_{i}" for i in range(1, 9)]


def find_cells(tag_dir: Path):
    """Return {map_name: [csv_path, ...]} for a nested {map}_run{N}/raw/ tree."""
    cells = {}
    for run_dir in sorted(tag_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if "_run" not in name:
            continue
        map_name = name.rsplit("_run", 1)[0]
        csv_path = run_dir / "raw" / f"{map_name}.csv"
        if csv_path.exists():
            cells.setdefault(map_name, []).append(csv_path)
    return cells


def flatten_tag(sim_root: Path, tag: str, out_label: str,
                expected_runs: int = 5, dry_run: bool = False):
    tag_dir = sim_root / tag
    if not tag_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {tag_dir}")

    cells = find_cells(tag_dir)
    if not cells:
        raise FileNotFoundError(
            f"No {{map}}_run{{N}}/raw/{{map}}.csv cells found under {tag_dir}.\n"
            f"Is this already a flat run?"
        )

    out_raw = sim_root / out_label / "raw"
    if not dry_run:
        out_raw.mkdir(parents=True, exist_ok=True)

    incomplete, stuck_report = [], []
    total_rows = 0

    for map_name in sorted(cells):
        frames = []
        for path in cells[map_name]:
            df = pd.read_csv(path)
            if len(df) != 1:
                print(f"  NOTE: {path} has {len(df)} rows (expected 1)")
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)

        if "run" in merged.columns:
            merged = merged.sort_values("run").reset_index(drop=True)
            if merged["run"].duplicated().any():
                dupes = merged.loc[merged["run"].duplicated(), "run"].tolist()
                print(f"  WARNING: {map_name} has duplicate run ids {dupes}")

        n = len(merged)
        total_rows += n
        if n != expected_runs:
            incomplete.append((map_name, n))

        present = [c for c in AGENT_COLS if c in merged.columns]
        if present:
            n_stuck = int((merged[present] <= 0).any(axis=1).sum())
            if n_stuck:
                stuck_report.append((map_name, n_stuck, n))

        if not dry_run:
            merged.to_csv(out_raw / f"{map_name}.csv", index=False)

    # ---- report -----------------------------------------------------
    verb = "Would write" if dry_run else "Wrote"
    print(f"\n{tag}  ->  {out_label}")
    print(f"  {verb} {len(cells)} maps, {total_rows} runs total")

    if incomplete:
        print(f"  INCOMPLETE MAPS ({len(incomplete)}), expected {expected_runs} runs:")
        for m, n in incomplete:
            print(f"    {m:16s} {n}/{expected_runs}")
        print("    -> map means rest on unequal n. Resubmit the array to fill,")
        print("       or drop these maps before comparing models.")
    else:
        print(f"  All maps complete ({expected_runs} runs each).")

    if stuck_report:
        print(f"  STUCK RUNS (agent never finished; excluded downstream):")
        for m, s, n in stuck_report:
            print(f"    {m:16s} {s}/{n}")

    return {"maps": len(cells), "rows": total_rows, "incomplete": incomplete}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tags", nargs="+", help="Tag directories, e.g. realL1_fixedorder_zap")
    p.add_argument("--sim-root", default="data/simulations",
                   help="default: data/simulations (run from repo root)")
    p.add_argument("--out", default=None,
                   help="output run_label; only valid with a single tag. "
                        "Default: {tag}_flat")
    p.add_argument("--expected-runs", type=int, default=5)
    p.add_argument("--dry-run", action="store_true",
                   help="report completeness, write nothing")
    args = p.parse_args(argv)

    if args.out and len(args.tags) > 1:
        p.error("--out only makes sense with a single tag")

    sim_root = Path(args.sim_root)
    if not sim_root.is_dir():
        p.error(f"sim_root not found: {sim_root} (are you in the repo root?)")

    any_incomplete = False
    for tag in args.tags:
        out_label = args.out or f"{tag}_flat"
        res = flatten_tag(sim_root, tag, out_label,
                          expected_runs=args.expected_runs,
                          dry_run=args.dry_run)
        any_incomplete |= bool(res["incomplete"])

    if any_incomplete:
        print("\nSome maps are incomplete — see above before analysing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
