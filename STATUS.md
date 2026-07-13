# STATUS — line-cutting model (updated 2026-07-13)

Single source of truth for: what's decided, what's known, what's open,
which code and data to trust. Update this file when any of those change.

---

## ⚠️ HARD DEADLINE FIRST

**CogSci poster PDF upload (Underline): July 15 — two days.**
Poster P1-M-122, session Thursday July 23, Rio.
The poster still has PLACEHOLDER FIGURES and needs final stat verification.
Everything below is secondary to shipping it. The poster rests entirely on
**May data (`depth2_v3_main5_cluster` etc.), which is validated** — nothing
discovered this week touches it.

Verified poster stats:
- real L1 stuck rate: **15/140 runs (10.7%), 9/28 maps** ✓ (recomputed 07-13)
- L1h stuck rate: **0/140** ✓
- r≈.86 / ΔR²≈.40: still to re-derive from `new_compare_models.ipynb`
  before upload.

---

## The three goals (agreed 07-13)

1. **Poster** — passable, honest, ships by the 15th. Story: L1 heuristic
   fits human universalization judgments best; forward simulation (real
   L1) is ~100x more expensive AND less robust (10.7% of runs never
   finish). Fragility is evidence FOR the resource-rational heuristic,
   not a flaw to hide.
2. **Understand why real L1 fails** (the mechanism). Open; see ledger.
   This is paper material and the part coauthors weren't yet convinced by.
3. **Code improvements (zap / fixed order)** — RESOLVED, negatively for
   zap: see Decisions.

---

## Findings ledger (July 10–13)

| # | Finding | Evidence |
|---|---------|----------|
| 1 | Poster's real-L1 data is sound | `depth2_v3_main5_cluster`: 15/140 stuck, fallback ~10% (matches April notebook note of 9.93%) |
| 2 | July changes ~doubled stranding | old→new stuck: L0 1→20, L1h 0→0, realL1 15→31 (140 runs each) |
| 3 | **Zap is causal; order is not** | seed-matched L0 pairs on yes_line_E, fixed order both: ghost 0/5 stuck, zap 3/5 stuck |
| 4 | Not a dropped predicate | domain state = has-water1/2/3, has-filled, has-completed, xloc/yloc, walls; `remove_agents` carries all of it |
| 5 | Reproduces at DEPTH=0 | so mechanism ≠ rollouts, ≠ -Inf cliff, ≠ fallback machinery |
| 6 | Stranding seeds = slow-ghost seeds | 271236/7/8 strand under zap; same seeds grind 318–498 timesteps under ghosts |
| 7 | L1h is bulletproof | 0 stuck in every dataset, old and new |
| 8 | Fixed order collapses run variance on easy maps but NOT congested ones | no_line_1 SD≈1.5 timesteps; yes_line_8 means 67–83 across runs (Boltzmann ties cascade under congestion) |
| 9 | Compute cost tracks congestion, 18x spread | realL1 cells: no_line_2 ≈700s … yes_line_8 ≈12.5-13k s (cluster, cold cache, 1 thread) |
| 10 | Old fill-run seeds were fine | none of the April/May fill scripts passed MAPS subsets, so the (now-fixed) seed bug never bit production data |
| 11 | **L0 "stuck" = synchronized period-2 limit cycles**, not deadlock | zapknot (seed 271237): 5 agents flip right/left in unison at x=13/14 for 936 timesteps to TIME_MAX; 0 fallbacks, 0 waits; locks in 2 steps after the last zap |
| 12 | **Ghosts were the system's only noise source; jitter hypothesis CONFIRMED** | ghostknot, same seed: identical cycle on identical cells (agent 6: 279 oscillations at (13,4)/(14,4)) but wandering ghosts eventually break it; escapes at t=88–400 |
| 13 | **Stuck detector is blind to cycles at ANY patience** | period-2 never yields equal consecutive states; both zap L0 strands and old-code depth-2 strands end at TIME_MAX, not via the detector. STUCK_PATIENCE arm is moot — dropped |
| 14 | **Old-code depth-2 stranding ≠ L0 cycles: CONFIDENT FROZEN WAITERS, robust to noise** | nm3_forensics run 3 (old code, ghosts): agents 3,4,7 FROZEN in final window — agent 7: wait×15, all tag=OK — while ghosts wander around them for ~900 timesteps. Fallback only 1–2% overall |
| 15 | **-Inf cliff is NOT the depth-2 stranding mechanism** | waiters never hit -Inf: rollouts return finite values ⇒ FALLBACK_PENALTY predicted ≈no effect (falsification arm) |
| 16 | **THE ZENO BUG (ancestral, in poster data too): depth-2 evaluation has no step cost within the horizon** | agent 7's 15 waits: best=0.0000 gap=0.0000 — EVERY candidate's rollout ends with k filled ⇒ all evaluate to flat 0.0 ⇒ exact tie ⇒ strict `>` keeps the FIRST candidate in PDDL.available()'s arbitrary order = `wait` ⇒ agent procrastinates one square from the tank, forever. Refutes the "confident margin" reading of #14 (Claude's prediction was wrong; Julien's "did we make a mistake?" was right). Deterministic re-tie each timestep also explains noise-immunity |

**MECHANISM (resolved 07-13 evening, findings #11–15). Two distinct
failure modes:**

- **Depth 0 (and zap runs generally): deterministic tie cycles.** Near-
  equal A* routes + temp 1e-4 argmax ⇒ position-parity flip-flop,
  synchronized across agents. Broken by ANY noise: ghost wandering
  (slowly) or TEMPERATURE (predicted: quickly — Wave 2 tests this).
  Others-as-walls is NOT a bug; it is the L0 model. The cycles are an
  emergent cost of determinism.
- **Depth 2 (old code): TWO ingredients.** (a) THE ZENO BUG (#16): flat
  0.0 for any within-horizon fill + first-wins ties ⇒ permanent
  procrastination at the goal line. A plain accounting bug, ancestral,
  present in the poster data. (b) Frozen-crowd pessimism: rollout
  evaluates futures with others frozen; projected jams look impassable
  though real ones disperse ⇒ over-selection of wait/yield near
  congestion (the real -Inf fallbacks, e.g. 29% on yes_line_8 old code,
  are this). How much of depth-2's worse fit is (a) vs (b) is now THE
  open question: fix (a), rerun, re-fit. If depth-2 still fits worse
  with the bug fixed, the resource-rational story stands on (b); if the
  fit gap closes, the paper's framing changes materially.

**Why depth 2 fits humans worse than depth 1 (working explanation for
paper/coauthors):** both share the others-as-walls assumption; depth 1
applies it closed-loop at horizon 1 where it is nearly true and re-plans
every step (errors never compound — 0 stuck ever); depth 2 rolls a
misspecified model open-loop and commits, amplifying the model-of-others
error (the model-based-RL short-rollout lesson). Behaviorally: hesitation
and yielding near congestion, self-confirming jams, TIME_MAX exhaustions —
distorting universalized welfare precisely on the congested maps where
human universalization judgments are most diagnostic. Deeper planning
multiplies the error of a cheap model of others; the heuristic's myopia
is protective. Framing for the paper: our depth-2 is a NAIVE forward
simulator — smarter variants (others keep acting during evaluation,
sampled/expected-value rollouts, re-plan awareness) are the follow-up
model space, not a patch.

`nm3_forensics`: 3/10 runs analyzed (runs 1–2 likely completed; run 3 is
the TIME_MAX strander with frozen waiters). Re-run forensics when all 10
land and the CSV exists for exact finished/stranded splits.

---

## Decisions

- **D1 (07-13): No zap in production runs.** `ZAP=false` (or old data)
  for anything entering analysis until Goal 2 is resolved. Zap remains
  available as an experimental knob.
- **D2: Keep fixed play order** (agent1→8). Innocent per finding #3;
  simplifies reproduction. Note: run-to-run variance now comes only from
  Boltzmann ties (see #8) — keep RUNS=5 on congested maps.
- **D3: Poster uses May data.** `depth2_v3_main5_cluster` (real L1),
  `depth1_main5_cluster` (L1h), `depth0_cluster` (L0).
- **D4: All cluster production runs via job array**, one (map,run) cell
  per task, cold cache — makes compute-cost numbers comparable across
  models (memoization warm-up made mean cost depend on RUNS).
- **D5: Seeds keyed on CANONICAL_MAP_FILES index** — never reorder that
  list; append only. Verified backward-compatible with May full sweeps.
- **D6: elapsed-seconds is a questionable cognitive-cost measure**
  (mixes hardware/cache); consider mechanism counts (A* expansions,
  evaluate_action calls) for the paper's ΔR²-vs-compute figure. UNRESOLVED
  — raise with Katie.

---

## Code state

Canonical: `src/main.jl` (single script; all `main*.jl` variants archived
in `src/archive_pre-july2026/`). Knobs, all env vars, defaults = historical
behavior:

| knob | default | meaning |
|---|---|---|
| LEVEL / DEPTH | 0 / 0 | model space (L0=0,0; L1h=0,1; realL1=0,2) |
| RUNS / RUN_OFFSET | 5 / 0 | run count / global run index shift (arrays) |
| MAPS | all 28 | subset by name; seeds stay canonical |
| ORDER | 1..8 | play order permutation |
| ZAP | true | false = ghosts (pre-July semantics) — **use false, see D1** |
| STUCK_PATIENCE | 3 | identical states before "stuck" (period-2 oscillation invisible at any value) |
| FALLBACK_PENALTY | inf | finite ⇒ graded rollout evaluation (no -Inf cliff) |
| TEMPERATURE | 0.0001 | Boltzmann; raise to inject agent-level noise |
| EVAL_STEPCOST | false | true = rollout evaluation counts k's spent actions (fixes Zeno bug #16). false = historical flat-0 |
| TIEBREAK | first | random = uniform among tied-best candidates (uses run RNG). first = historical list-order |
| TRAJECTORY_LEVEL | none | summary = forensics/visualizer-ready; full = state dumps |
| USE_TIMESTAMP | true | false ⇒ stable path for idempotent array cells |

Cluster: `cluster/run_array.sbatch` (production; striped task→(map,run);
idempotent; resubmit fills gaps), `cluster/run_diag.sbatch` (7-map panel
× 20 runs; COND presets: ghost_fixed, zap_varied, ghost_varied, penalty,
zap_d0, ghost_d0, zap_temp_d0, ghost_temp_d0, custom).

Analysis: `flatten_array_runs.py` (nested→flat; run before pipeline),
`run_checks.py` (coverage/variance), `stuck_forensics.py` (deadlock vs
oscillation vs lag; mutual-hesitation check), `trajectory_visualizer.py`
(zap-aware; FALLBACK ring). Pipeline itself unchanged.
TODO: `build_utils.KNOB_COLS` still lists retired knobs; should be
["level","depth","order","zap","stuck_patience","fallback_penalty"].

Pinned: SymbolicPlanners SHA b7d8d6e (FunctionalVPolicy). Julia 1.10
(cluster: `module load julia/1.10.4`).

---

## Data inventory

TRUST for analysis/poster:
- `depth0_cluster`, `depth1_main5_cluster`, `depth2_v3_main5_cluster` (May; poster)
- `L1h_fixedorder_zap` (new; 0 stuck — L1h robust to everything)
- `L0_fixedorder_zap`, `realL1_fixedorder_zap`: **quarantined** for model
  comparison (zap inflates stranding, D1); fine as the zap arm of diagnostics.

Diagnostics (07-13): `l0_ghost_yE` / `l0_zap_yE` (verdict pair, finding #3),
`nl2_ghost`, `y8_old` / `y8_new`, smoke_* (knob no-op checks pass:
smoke_default reproduces 21,19,22,20,22,26,30,26 exactly).
Pending: `zapknot`, `ghostknot`, `nm3_forensics`.

Old depth2_v4 / depth2_v6: abandoned in April for stuck agents (see
notebook comments) — do not resurrect.

---

## Next actions

NOW (blocking, ~independent of everything above):
1. **Poster figures + stat verification + send v1 to Katie/Sydney.**
2. **Upload to Underline by July 15.** Print before flying.

WHEN LOCAL RUNS FINISH (tonight):
3. Read `zapknot`/`ghostknot` forensics + animation → confirm/kill jitter
   hypothesis. Update ledger.
4. Read `nm3_forensics` → mechanism of ORIGINAL (old-code) stuck runs:
   TRUE DEADLOCK + wait/OK (confident hesitation) vs OSCILLATION
   (detector blind spot) vs LAG. This is the paper's robustness section.

CLUSTER (Wave 1 = L0_ghost / L1h_ghost / realL1_ghost launched 07-13;
420 cells, ghost mode, trajectories on, seed-matched to zap runs):
5. Wave 2 (launch when queue thins): `COND=ghost_d0`, `COND=zap_d0`,
   `COND=zap_temp_d0` (temp 0.05). PREDICTION: temp arm ≈ 0 stranding
   (cycles are ties; noise breaks them).
6. Wave 3 (re-revised after #16): the PRIORITY arm is now the Zeno fix —
   depth-2, ghosts, EVAL_STEPCOST=true TIEBREAK=random, full 28 maps
   n=5 (`realL1_fixed_eval` via run_array with those envs). Question:
   with correct accounting, (i) do the strands vanish? (ii) does real L1
   still fit humans worse than L1h? That answer decides the paper's
   framing. Panel arms (ghost_fixed census, penalty falsification,
   d2+temp) remain useful but secondary. Local first: rerun
   nm3 seed of run 3 with EVAL_STEPCOST=true — does agent 7 fill?
7. STUCK_PATIENCE arm: DROPPED (finding #13 — moot at any value).
8. Wave 1 analysis when realL1_ghost completes: flatten → fit all three
   to Joe's judgments → rank maps by fit gap (L1h vs realL1) → forensics
   + visualizer on the worst → does the fit gap track waiter density?
   That is the paper's Section 4.

PAPER (post-Rio):
- Robustness/mechanism section from #4-#6.
- ΔR²-vs-compute figure; settle cost measure (D6) with Katie.
- Bootstrap CIs (Logan's individual data), per-country fits.
- Authorship/timeline conversation with Sydney in Rio (poster session
  Thu Jul 23, then coffee): firm first-authorship + concrete bar →
  invest from Paris; vague → wrap-up mode.

---

## Temperature note (Julien, 07-13)

"If dead agents help, that's a result; higher temperature might substitute
— and could help with lots of stuff." Agreed, with one caution: TEMPERATURE
also changes the *model* being fit to humans, not just the harness — L1h's
current fit numbers are at 1e-4, so any temp change for depth-2 rescue
must be applied to ALL models symmetrically before comparing fits, and
May-data comparability is lost. Rescue experiments at depth 0 first (cheap,
step 5); only touch the model-comparison temperature deliberately, as its
own decision.
