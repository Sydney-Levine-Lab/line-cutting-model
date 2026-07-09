#!/bin/bash
# debug_yesline_8_9.sh
#
# Compare main8_d (buggy, +sign) vs main9_d (fixed, -sign) on yes_line_8
# and yes_line_9 with production seeds. Runs both scripts back-to-back
# with full DEBUG_EVAL trajectory logging so per-decision values are
# captured for postmortem comparison.
#
# Estimated wall time: ~2 x 11h = ~22h (yes_line_8 is the slow one).
# Both maps run in parallel within each script (Threads.@threads across maps).
#
# Usage:
#   ./debug_yesline_8_9.sh                # default (2 threads, see note)
#   THREADS=8 ./debug_yesline_8_9.sh      # if you want extra cores available
#
# Note on threading:
# main9_d.jl uses `Threads.@threads for map_index in eachindex(MAP_FILES)`,
# so parallelism is across maps. With 2 maps, only 2 worker threads get work;
# extras sit idle. Setting THREADS=2 is therefore the most efficient choice
# on a busy laptop. Use higher values only if you want headroom for OS / IDE.
#
# Run from project root (proj2-nerfed/).

set -u

MAPS="yes_line_8,yes_line_9"
THREADS="${THREADS:-2}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== debug_yesline_8_9.sh ==="
echo "Started:  $(date)"
echo "Maps:     $MAPS"
echo "Threads:  $THREADS"
echo "Mode:     DEPTH=2 RUNS=1 with DEBUG_EVAL=true and SAVE_TRAJECTORIES=true"
echo ""

# Sanity checks
echo "--- Sign convention sanity check ---"
ok=true
echo -n "main8_d.jl: "
if grep -q "return SymbolicPlanners.compute(steps_to_go" src/main8_d.jl 2>/dev/null; then
    echo "BUGGY (positive sign) - good"
elif grep -q "return -SymbolicPlanners.compute(steps_to_go" src/main8_d.jl 2>/dev/null; then
    echo "WARN: main8_d.jl has the FIXED sign - is this really the buggy debug version?"
    ok=false
else
    echo "WARN: couldn't find expected sign line in main8_d.jl"
    ok=false
fi
echo -n "main9_d.jl: "
if grep -q "return -SymbolicPlanners.compute(steps_to_go" src/main9_d.jl 2>/dev/null; then
    echo "FIXED (negative sign) - good"
else
    echo "WARN: main9_d.jl does NOT contain expected fix line"
    ok=false
fi
if [ "$ok" != "true" ]; then
    echo ""
    echo "Sign-check failed. Continue anyway? (Ctrl-C to abort, Enter to proceed)"
    read
fi
echo ""

# --- Run buggy version ---
echo "--- Running main8_d (buggy) at $(date) ---"
LABEL_BUGGY="debug_buggy_yesline_${TIMESTAMP}"
DEPTH=2 RUNS=1 \
    MAPS="$MAPS" \
    DEBUG_EVAL=true \
    SAVE_TRAJECTORIES=true \
    VERBOSE=true \
    RUN_LABEL="$LABEL_BUGGY" \
    julia +1.10 --project=src -t "$THREADS" src/main8_d.jl \
    > "${LABEL_BUGGY}.log" 2>&1
BUGGY_EXIT=$?
echo "  done at $(date) (exit $BUGGY_EXIT, log: ${LABEL_BUGGY}.log)"
echo ""

# --- Run fixed version ---
echo "--- Running main9_d (fixed) at $(date) ---"
LABEL_FIXED="debug_fixed_yesline_${TIMESTAMP}"
DEPTH=2 RUNS=1 \
    MAPS="$MAPS" \
    DEBUG_EVAL=true \
    SAVE_TRAJECTORIES=true \
    VERBOSE=true \
    RUN_LABEL="$LABEL_FIXED" \
    julia +1.10 --project=src -t "$THREADS" src/main9_d.jl \
    > "${LABEL_FIXED}.log" 2>&1
FIXED_EXIT=$?
echo "  done at $(date) (exit $FIXED_EXIT, log: ${LABEL_FIXED}.log)"
echo ""

# --- Quick CSV comparison ---
echo "=== CSV comparison ==="
for map in ${MAPS//,/ }; do
    echo ""
    echo "Map: $map"
    buggy_csv=$(ls data/simulations/${LABEL_BUGGY}_*/raw/$map.csv 2>/dev/null | head -1)
    fixed_csv=$(ls data/simulations/${LABEL_FIXED}_*/raw/$map.csv 2>/dev/null | head -1)
    if [ -f "$buggy_csv" ]; then
        echo "  buggy fill times: $(grep '^1,' $buggy_csv | cut -d, -f2-9)"
    else
        echo "  buggy CSV not found"
    fi
    if [ -f "$fixed_csv" ]; then
        echo "  fixed fill times: $(grep '^1,' $fixed_csv | cut -d, -f2-9)"
    else
        echo "  fixed CSV not found"
    fi
done

echo ""
echo "=== Trajectory logs ==="
echo "Buggy: data/simulations/${LABEL_BUGGY}_*/raw/trajectories/"
echo "Fixed: data/simulations/${LABEL_FIXED}_*/raw/trajectories/"
echo ""
echo "Inspect first chosen actions and value signs:"
echo "  for f in data/simulations/${LABEL_BUGGY}_*/raw/trajectories/*.log; do echo \"--- \$f ---\"; grep -m 3 'chosen:' \"\$f\"; done"
echo "  for f in data/simulations/${LABEL_FIXED}_*/raw/trajectories/*.log; do echo \"--- \$f ---\"; grep -m 3 'chosen:' \"\$f\"; done"
echo ""
echo "=== Done at $(date) ==="
