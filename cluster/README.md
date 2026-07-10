# cluster/

One script, `run_array.sbatch`, for every model. Submit from the repo root.

```bash
mkdir -p logs

TAG=L0_fixedorder_zap     DEPTH=0           sbatch cluster/run_array.sbatch
TAG=L1h_fixedorder_zap    DEPTH=1           sbatch cluster/run_array.sbatch
TAG=realL1_fixedorder_zap DEPTH=2           sbatch cluster/run_array.sbatch
TAG=L2h_fixedorder_zap    DEPTH=1 LEVEL=1   sbatch cluster/run_array.sbatch
TAG=realL2_fixedorder_zap DEPTH=2 LEVEL=1   sbatch cluster/run_array.sbatch
```

140 tasks = 28 maps x 5 runs. `%100` throttles concurrency.

## Why one cell per task, even for cheap models

`build_steps_to_go_estimator()` (the memoized A* cache) is built **once per
map, outside the run loop**. Run 1 pays to build it; runs 2-5 free-ride.
Locally: 361s / 292s / 262s across runs 1-3 of the same map.

Two consequences:

1. **Mean elapsed-time-per-run depends on `RUNS`.** With a warm cache, the
   more runs you do, the cheaper the average run looks. That makes elapsed
   time an artifact of experimental design rather than a property of the
   model — fatal if compute cost is a *result* (the delta-R^2-vs-compute
   figure).
2. Mixing designs across models (L0 threaded-and-warm, real L1 cold) would
   make the cost comparison meaningless.

One cell per task means every cell pays cold-cache cost. Costs become
comparable across models. We give up ~25% of wall-clock; we don't care, and
it buys the wall-clock ceiling escape and preemption resilience too.

### Caveat: is elapsed time the right cost measure at all?

The memo cache is an implementation convenience, not a cognitive claim — a
person doesn't cache A* plan costs across 8 agents. Wall-clock seconds also
mix in hardware, threads, and cache state. For the paper's resource-
rationality figure, a mechanism-level counter (A* node expansions, or
`evaluate_action` calls) may be more defensible than seconds-on-a-cluster.
Worth deciding before the figure is built.

## Idempotence

`USE_TIMESTAMP=false` gives each cell a stable path:

```
data/simulations/{TAG}/{map}_run{N}/raw/{map}.csv
```

Before running, a task checks whether its own CSV exists with >= 2 lines
(header + one data row). If so it exits 0. A truncated file (preempted
mid-write) is deleted and regenerated.

So after preemption you just **resubmit the whole array**. Completed cells
skip in seconds; only missing ones run. No bookkeeping.

```bash
sbatch cluster/run_array.sbatch                      # re-run, fills gaps
sbatch --array=17,43,91 cluster/run_array.sbatch     # or target specific cells
```

Check completeness:
```bash
ls data/simulations/realL1_fixedorder_zap/*/raw/*.csv | wc -l   # expect 140
sacct -j <ARRAY_JOB_ID> --format=JobID,State,Elapsed | grep -v COMPLETED
```

This replaces the old hand-rolled `run_l2d2_fill_r4.sh`-style fill scripts,
one per missing run.

## Seed invariant (important)

Seeds derive from each map's position in `CANONICAL_MAP_FILES` (in
`src/main.jl`), **not** its position in whatever `MAPS` subset a task
passes. So:

```
MAPS="yes_line_8" RUNS=1 RUN_OFFSET=2
```
gets exactly the seed that run 3 of `yes_line_8` would get in a full 28-map
sweep. This is what makes array jobs sound.

This was **not** true before July 2026: seeds used the subset-local index.
Full 28-map sweeps were unaffected (the current `seed_for` reproduces their
seeds exactly, verified for all 140 cells), but the April/May
`*_fill_run{0-4}` jobs passed `RUN_OFFSET` — check whether they also passed
a `MAPS` subset. If they did, they drew **different seeds than intended**,
and any fill-run data reaching the paper needs a footnote.

**Never reorder `CANONICAL_MAP_FILES`.** Append new maps at the end.
`MAPS_ARR` in `run_array.sbatch` must mirror it exactly — the array index
maps to it positionally.

## Merging for analysis

Each cell writes its own directory, so the analysis pipeline needs the 140
dirs stitched into one multi-run directory per model. Use
`merge_split_runs` — its glob likely needs updating for the new
`{TAG}/{map}_run{N}/` layout, which differs from the old flat
`{label}_{timestamp}/` one.

## Cluster facts to confirm

Taken from `run_l2d2_fill_r4.sh` (April 2026):
- `module load julia/1.10.4` — confirm still available (`module avail julia`)
- `mit_preemptable`: 48h, preemptable. `mit_normal`: 12h, no preemption.
- `--mem=16G` and `--cpus-per-task=6` were grantable.

Unverified, worth checking before the first big submit:
- Does `mit_normal` permit job arrays? Some sites restrict arrays to
  preemptable partitions. With idempotence, `mit_preemptable` is now
  perfectly safe anyway.
- `scontrol show config | grep MaxArraySize` — needs to be >= 140.
- `sinfo -s`, `scontrol show partition mit_normal` for MaxTime.
