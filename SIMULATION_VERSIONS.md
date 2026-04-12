# Newer Simulation versions

Tracks changes to the simulation model in `src/main.jl` and corresponding runs. Older versions under src/failed_attempts_depth.

Depth 2 versions:
- main.jl (april 7): agents just plan assuming others are walls at future positions rather than predicted current ones (depth1).
- main2.jl (april 8): cleaner version of that (could be implemented for any depth).
- main3.jl (april 8): action enumeration, enumerating possible actions for focal agent, using A*, and comparing --- doesn't finish
- main4.jl (april 8): slightly better than main, where you also predict your own move.
- main5.jl (april 9): plays the game forward ratehr than enumerates mental actions. 
- main6.jl (april 9): also allows to modify the depth of reasoning (e.g., implemenmt L1 reasoning)




Ongoing (April 9):
- depth3 with main4.jl
- depth2 with main5.jl


Other thing to do:
- try out heuristic L2 reasoning. Should be easy enugh to code.











# Older Simulation versions


---

## v1 — Original depth code (March 16–17)

**Code:** First implementation of `REASONING_DEPTH` replacing old `REASONING` knob.
- All agents predict all others (not just predecessors)
- Random order redrawn every timestep (same as before)
- Agents are "blind" — plan from start-of-round snapshot
- Predictions sample a random order internally (noisy, expensive)
- Depth 2 projects walls at final predicted positions only (no real lookahead)

**Runs:**
- `depth0_03-16` — REASONING_DEPTH=0, 5 runs, 28 maps. Finished.
- `depth1_03-16` — REASONING_DEPTH=1, 5 runs, 28 maps. Finished (~25h).
- `depth2_2026-03-17_202841` — REASONING_DEPTH=2, 5 runs. Partial (very slow).

**Issues:** Depth 1 was ~6x slower than expected. Depth 2 was impractical.

---

## v2 — Fixed order + observation model (March 20)

**Code:** Major revision to the information model.
- Order drawn once per run (not per timestep). Agents observe the order.
- Observation model: agents see predecessors' current positions (just moved)
  and successors' positions from last timestep.
- Predictions use the known order — deterministic, no random sampling.
- Depth 2 still just projects walls at final predicted positions (same flaw as v1).

**Runs:**
- `v2_d0` — REASONING_DEPTH=0, 5 runs. 
- `v2_d1` — REASONING_DEPTH=1, 5 runs.
- `v2_d2` — REASONING_DEPTH=2, 2 runs.

**Notes:** v2 depth 0 already incorporates partial information from predecessors,
so it should perform better than v1 depth 0 (which was fully blind).
v2 depth 0 ≈ old L1 heuristic in spirit.

---

## v3 — True depth-2 lookahead (March 23)

**Code:** Depth 2+ now does real lookahead planning.
- Depth 1: same as v2 (predict one round, project walls, A* picks action).
- Depth 2+: predicts D rounds of positions. For each candidate first action,
  mentally moves agent there, then evaluates with A* against round-D walls.
  Picks the first action with the best lookahead. (~5 A* calls for depth 2.)
- This means depth 2 genuinely plans through time-varying obstacles:
  "avoid where people are now, head where it's good given where they'll be next."

**Runs:**
- `v3_d0` — TODO
- `v3_d1` — TODO
- `v3_d2` — TODO

---

## Baseline runs (reference)

These use the OLD `main.jl` (with `INFO`, `ORDER`, `REASONING` knobs):
- `joe_1_02-06` — Joe's original: ORDER=fixed, INFO=full, REASONING=L0
- `full_random_02-06` — ORDER=random, INFO=full, REASONING=L0
- `blind_L0_random_02-06` — ORDER=random, INFO=blind, REASONING=L0
- `blind_l1_random_02-06` — ORDER=random, INFO=blind, REASONING=L1
- `random_blind_L1_03-02` — same as above, different date
- Various `XXfull_03-*` — partial information sweep (INFO=partial, INFO_PROB=XX%)
