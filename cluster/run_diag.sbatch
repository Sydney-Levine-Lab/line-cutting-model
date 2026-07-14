#!/bin/bash
# ====================================================================
# Diagnostic array: isolate WHY depth-2 runs get stuck.
#
# Confound to untangle: the July changes altered TWO things at once —
# (1) zap on fill (vs ghosts), (2) fixed play order (vs one random
# order per run). Fixed order also collapses run diversity: all runs
# share one order, so a deadlock-prone order fails repeatedly, which
# alone could explain the higher per-map stuck counts.
#
# Design: PANEL of 7 maps x 20 runs = 140 cells per condition.
#   COND=zap_fixed      ZAP=true   fixed order        (= July baseline)
#   COND=ghost_fixed    ZAP=false  fixed order        (isolates zap)
#   COND=zap_varied     ZAP=true   order varies/run   (isolates order)
#   COND=ghost_varied   ZAP=false  order varies/run   (~= old behavior)
#   COND=penalty        ZAP=true   fixed order, FALLBACK_PENALTY=100
#                       (tests the -Inf evaluation-cliff mechanism)
#
# Reading the 2x2: stuck(zap_fixed) - stuck(ghost_fixed) = zap effect;
# stuck(zap_fixed) - stuck(zap_varied) = order effect. If penalty
# condition unsticks runs, the -Inf cliff is the mechanism.
#
# Launch (each line is one condition; run sequentially or together):
#   mkdir -p logs
#   COND=ghost_fixed  sbatch cluster/run_diag.sbatch
#   COND=zap_varied   sbatch cluster/run_diag.sbatch
#   COND=ghost_varied sbatch cluster/run_diag.sbatch
#   COND=penalty      sbatch cluster/run_diag.sbatch
#   # zap_fixed runs 0-4 already exist as realL1_fixedorder_zap;
#   # for runs 5-19 of it:  COND=zap_fixed sbatch cluster/run_diag.sbatch
#
# All conditions use TRAJECTORY_LEVEL=summary so stuck_forensics.py and
# the visualizer can dissect every run. Idempotent per cell, as usual.
# ====================================================================
#SBATCH --job-name=diag
#SBATCH --array=0-139%70
#SBATCH --time=48:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
module load julia/1.10.4
cd ~/line-cutting-model

# Panel: the worst stranders from both old and new data, plus one clean
# congested control (yes_line_8: 0 stuck in old data, stuck in new) and
# one easy control (no_line_1).
PANEL=(no_line_1 no_line_2 no_line_3 new_maybe_3 maybe_6 yes_line_10 yes_line_8)

N_PANEL=${#PANEL[@]}
RUN_IDX=$(( SLURM_ARRAY_TASK_ID / N_PANEL ))
MAP_IDX=$(( SLURM_ARRAY_TASK_ID % N_PANEL ))
MAP_NAME="${PANEL[$MAP_IDX]}"

COND="${COND:?set COND=zap_fixed|ghost_fixed|zap_varied|ghost_varied|penalty}"

# 20 fixed random permutations of 1..8 (generated once, seed 4242; do
# not regenerate — reproducibility). Used by the *_varied conditions,
# indexed by RUN_IDX, so run r uses the same order in every condition.
ORDERS=( \
 "3,6,1,8,2,7,5,4" "5,2,8,4,7,1,3,6" "7,4,2,6,8,3,1,5" "1,8,5,3,6,2,4,7" \
 "6,3,7,1,4,8,2,5" "2,7,4,8,1,5,6,3" "8,1,6,5,3,4,7,2" "4,5,3,7,2,6,8,1" \
 "3,8,2,5,7,4,1,6" "6,1,7,3,5,8,4,2" "5,4,8,2,6,1,7,3" "2,6,3,8,4,7,5,1" \
 "7,3,5,1,8,2,6,4" "1,5,4,6,2,3,8,7" "8,2,1,7,3,6,4,5" "4,7,6,2,5,1,3,8" \
 "5,1,3,8,6,7,2,4" "3,4,8,6,1,5,7,2" "6,8,2,4,7,3,5,1" "2,3,7,5,8,4,1,6" )

ZAP=true; ORDER=""; PENALTY="inf"; TEMP="0.0001"; PATIENCE="3"; DIAG_DEPTH="2"
case "$COND" in
  zap_fixed)    ZAP=true  ;;
  ghost_fixed)  ZAP=false ;;
  zap_varied)   ZAP=true  ; ORDER="${ORDERS[$RUN_IDX]}" ;;
  ghost_varied) ZAP=false ; ORDER="${ORDERS[$RUN_IDX]}" ;;
  penalty)      ZAP=true  ; PENALTY="100" ;;
  # Depth-0 temperature rescue: does agent-level noise replace ghost jitter?
  zap_temp_d0)    ZAP=true  ; TEMP="${TEMP_OVERRIDE:-0.05}" ; DIAG_DEPTH="0" ;;
  ghost_temp_d0)  ZAP=false ; TEMP="${TEMP_OVERRIDE:-0.05}" ; DIAG_DEPTH="0" ;;
  zap_d0)         ZAP=true  ; DIAG_DEPTH="0" ;;
  ghost_d0)       ZAP=false ; DIAG_DEPTH="0" ;;
  # Fully custom: set ZAP/ORDER/PENALTY/TEMP/PATIENCE/DIAG_DEPTH yourself
  custom)       ZAP="${ZAP_OVERRIDE:-true}"; PENALTY="${PENALTY_OVERRIDE:-inf}"
                TEMP="${TEMP_OVERRIDE:-0.0001}"; PATIENCE="${PATIENCE_OVERRIDE:-3}"
                DIAG_DEPTH="${DEPTH_OVERRIDE:-2}" ;;
  *) echo "unknown COND=$COND" >&2; exit 1 ;;
esac

TAG="diag_${COND}${TAG_SUFFIX:+_$TAG_SUFFIX}"
RUN_LABEL="${TAG}/${MAP_NAME}_run${RUN_IDX}"
OUT_CSV="data/simulations/${RUN_LABEL}/raw/${MAP_NAME}.csv"

echo "task=$SLURM_ARRAY_TASK_ID cond=$COND map=$MAP_NAME run=$RUN_IDX zap=$ZAP order=${ORDER:-fixed} penalty=$PENALTY temp=$TEMP patience=$PATIENCE depth=$DIAG_DEPTH"
echo "host=$(hostname) start=$(date -Is)"

if [[ -s "$OUT_CSV" ]] && [[ $(wc -l < "$OUT_CSV") -ge 2 ]]; then
    echo "SKIP: $OUT_CSV already complete."
    exit 0
fi
[[ -e "$OUT_CSV" ]] && { echo "WARN: truncated, regenerating."; rm -f "$OUT_CSV"; }

MAPS="$MAP_NAME" \
RUNS=1 \
RUN_OFFSET="$RUN_IDX" \
LEVEL=0 \
DEPTH="$DIAG_DEPTH" \
ZAP="$ZAP" \
ORDER="$ORDER" \
FALLBACK_PENALTY="$PENALTY" \
TEMPERATURE="$TEMP" \
STUCK_PATIENCE="$PATIENCE" \
RUN_LABEL="$RUN_LABEL" \
USE_TIMESTAMP=false \
TRAJECTORY_LEVEL=summary \
WRITE_SNAPSHOT=true \
EVAL_STEPCOST="${EVAL_STEPCOST:-false}" \
TIEBREAK="${TIEBREAK:-first} \
julia --project=src -t 1 src/main.jl

echo "end=$(date -Is)"
