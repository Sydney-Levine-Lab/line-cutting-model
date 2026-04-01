#!/bin/bash
# Run depth 0, 1, 2 simulations with timing.
# Usage: bash run_depth_sweep.sh

LOG="next_depth_sweep_timing.txt"
echo "=== Depth sweep started: $(date) ===" | tee "$LOG"
echo "" | tee -a "$LOG"

# Depth 1
echo "--- Starting depth 1 ---" | tee -a "$LOG"
T1_START=$SECONDS
ORDER=random REASONING_DEPTH=1 RUNS=5 VERBOSE=true RUN_LABEL=v3_d1 \
  julia +1.10 -t 8 --project=src src/main_next.jl
T1_ELAPSED=$(( SECONDS - T1_START ))

# Depth 2 (fewer runs since it's slow)
echo "--- Starting depth 2 ---" | tee -a "$LOG"
T2_START=$SECONDS
ORDER=random REASONING_DEPTH=2 RUNS=2 VERBOSE=true RUN_LABEL=v3_d2 \
  julia +1.10 -t 8 --project=src src/main_next.jl
T2_ELAPSED=$(( SECONDS - T2_START ))

# Depth 0
echo "--- Starting depth 0 ---" | tee -a "$LOG"
T0_START=$SECONDS
ORDER=random REASONING_DEPTH=0 RUNS=5 VERBOSE=true RUN_LABEL=v3_d0 \
  julia +1.10 -t 8 --project=src src/main_next.jl
T0_ELAPSED=$(( SECONDS - T0_START ))

TOTAL=$(( T0_ELAPSED + T1_ELAPSED + T2_ELAPSED ))

# Pretty print
fmt() {
  local s=$1
  printf "%dh %02dm %02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "  Depth 0 (5 runs):  $(fmt $T0_ELAPSED)" | tee -a "$LOG"
echo "  Depth 1 (5 runs):  $(fmt $T1_ELAPSED)" | tee -a "$LOG"
echo "  Depth 2 (2 runs):  $(fmt $T2_ELAPSED)" | tee -a "$LOG"
echo "  ──────────────────────────────────────" | tee -a "$LOG"
echo "  Total:             $(fmt $TOTAL)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Timing saved to $LOG"
