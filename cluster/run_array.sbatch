#!/bin/bash
# ====================================================================
# Job array: one Slurm task per (map, run) cell. 28 maps x 5 runs = 140.
#
# Used for ALL models (L0, L1h, real L1, real L2), not just expensive
# ones. Two reasons:
#
#   1. Comparable compute cost. The memoized A* cache is built once per
#      map, outside the run loop, so run 1 pays for it and runs 2-5
#      free-ride. Mean elapsed-time-per-run therefore depends on RUNS,
#      which makes it an artifact rather than a model property. One
#      cell per task means every cell pays cold-cache cost, so cost
#      numbers are comparable across models.
#   2. No wall-clock ceiling; preemption kills one cell, not the job.
#
# The price is ~25% lost to cold caches on runs 2-5. Worth it.
#
# Seeds key on each map's CANONICAL index in src/main.jl, so a
# single-map task reproduces the seed that cell would get inside a full
# 28-map sweep. Verified for all 140 cells.
#
# IDEMPOTENT: a cell whose output CSV already exists is skipped. Safe to
# resubmit the whole array after preemption; only missing cells run.
#
# Launch from the repo root:
#   mkdir -p logs
#   TAG=L0_fixedorder_zap   DEPTH=0         sbatch cluster/run_array.sbatch
#   TAG=L1h_fixedorder_zap  DEPTH=1         sbatch cluster/run_array.sbatch
#   TAG=realL1_fixedorder_zap DEPTH=2       sbatch cluster/run_array.sbatch
#   TAG=realL2_fixedorder_zap DEPTH=2 LEVEL=1 sbatch cluster/run_array.sbatch
#
# Resubmit stragglers:  sbatch --array=17,43,91 cluster/run_array.sbatch
# ====================================================================
#SBATCH --job-name=lines
#SBATCH --array=0-139%100
#SBATCH --time=48:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load julia/1.10.4

REPO=~/line-cutting-model
cd "$REPO"

# MUST mirror CANONICAL_MAP_FILES in src/main.jl, in the same order.
MAPS_ARR=(no_line_1 no_line_2 no_line_3 \
          yes_line_7 yes_line_8 yes_line_9 yes_line_10 \
          7esque 9esque 10esque \
          maybe_4 maybe_5 maybe_6 \
          new_maybe_1 new_maybe_2 new_maybe_3 \
          new_maybe_4 new_maybe_5 new_maybe_6 \
          no_line_A no_line_B no_line_C no_line_D \
          yes_line_B yes_line_C yes_line_D yes_line_E yes_line_F)

N_RUNS=5
MAP_IDX=$(( SLURM_ARRAY_TASK_ID / N_RUNS ))
RUN_IDX=$(( SLURM_ARRAY_TASK_ID % N_RUNS ))
MAP_NAME="${MAPS_ARR[$MAP_IDX]}"

LEVEL="${LEVEL:-0}"
DEPTH="${DEPTH:-2}"
TAG="${TAG:-run}"

# Stable, timestamp-free output dir so reruns are idempotent.
# NOTE: requires USE_TIMESTAMP=false support in main.jl (RUN_ID = RUN_LABEL).
RUN_LABEL="${TAG}/${MAP_NAME}_run${RUN_IDX}"
OUT_CSV="data/simulations/${RUN_LABEL}/raw/${MAP_NAME}.csv"

echo "task=$SLURM_ARRAY_TASK_ID map=$MAP_NAME run=$RUN_IDX level=$LEVEL depth=$DEPTH"
echo "host=$(hostname) start=$(date -Is)"

# A cell is "done" only if its CSV has a header + at least one data row.
# A task preempted mid-write can leave a truncated file, which a mere
# existence check would wrongly accept.
if [[ -s "$OUT_CSV" ]] && [[ $(wc -l < "$OUT_CSV") -ge 2 ]]; then
    echo "SKIP: $OUT_CSV already complete ($(wc -l < "$OUT_CSV") lines)."
    exit 0
fi
if [[ -e "$OUT_CSV" ]]; then
    echo "WARN: $OUT_CSV exists but looks truncated; regenerating."
    rm -f "$OUT_CSV"
fi

MAPS="$MAP_NAME" \
RUNS=1 \
RUN_OFFSET="$RUN_IDX" \
LEVEL="$LEVEL" \
DEPTH="$DEPTH" \
RUN_LABEL="$RUN_LABEL" \
USE_TIMESTAMP=false \
TRAJECTORY_LEVEL="${TRAJECTORY_LEVEL:-none}" \
WRITE_SNAPSHOT=true \
julia --project=src -t 1 src/main.jl

echo "end=$(date -Is)"
