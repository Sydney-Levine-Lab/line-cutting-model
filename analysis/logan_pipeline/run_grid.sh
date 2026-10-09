#!/usr/bin/env bash
# Idempotent grid runner: skips fits whose summary.json exists, so it can be relaunched after an interruption.
# Usage: bash run_grid.sh <tag> "<models>" "<datasets>" "<subsets>" [extra args to fit_judgment_model.py]
# Datasets are names of folders under data/anon/.  Run from Logan's repo root.
set -u
TAG=$1; MODELS=$2; DATASETS=$3; SUBSETS=$4; shift 4
for ds in $DATASETS; do for m in $MODELS; do for s in $SUBSETS; do
  out="output/${TAG}/${ds}__${m}__${s}"
  if [ -f "$out/summary.json" ]; then echo "skip $out"; continue; fi
  echo "=== $(date '+%H:%M') $out ==="
  uv run python fit_judgment_model.py "data/anon/$ds" "$out" --scores model_scores.csv \
      --model "$m" --subset "$s" --show-progress=False "$@" > "output/${TAG}_${ds}__${m}__${s}.log" 2>&1 \
      || echo "FAILED $out (see log)"
done; done; done
echo "grid done $(date '+%H:%M')"
