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

**Leading hypothesis for #3/#6 (UNCONFIRMED — one experiment pending):**
ghosts are annealing noise. Finished agents wander; their movement
perturbs stuck agents' planning views every timestep, re-rolling
Boltzmann ties until mutual-blocking knots dissolve (slowly — hence the
slow-ghost runs). Zap removes the jitter source; knots become absorbing
states; stuck detector fires. If confirmed: not a bug — a result. It
means the system relied on goal-less agents for exploration noise, and
the principled fix is agent-level noise (higher TEMPERATURE), not corpses.

**Decisive experiment (launched locally 07-13, results pending):**
`zapknot` / `ghostknot` — L0, yes_line_E, seed 271237, summary
trajectories → stuck_forensics + visualizer. Knot of mutually-blocking
agents that dissolves under ghosts ⇒ jitter hypothesis. Stranded agents
in open space ⇒ state corruption after all (then: TRAJECTORY_LEVEL=full
state diff around first [zap]).

Also pending: `nm3_forensics` — 10 old-code depth-2 runs on new_maybe_3
(the old data's worst map) for forensics on the ORIGINAL stuck mechanism.

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

CLUSTER (gate on #3's verdict — do not launch the factorial blind):
5. If jitter confirmed → depth-0 temperature rescue, cheap:
   `COND=zap_d0`, `COND=ghost_d0`, `COND=zap_temp_d0` (temp 0.05),
   optionally TEMP_OVERRIDE sweep {0.01, 0.05, 0.2}. Question: does
   agent-level noise substitute for ghost jitter? If yes, the fix for
   future code is temperature, not ghosts — cognitively principled
   (softmax noise) and removes the weird dependence on goal-less agents.
6. Then depth-2: `COND=penalty` (graded evaluation) vs inf — does
   removing the -Inf cliff cut old-style stuck/fallback rates, and does
   it change fit to humans? (If fragility is fixable but fit advantage
   of L1h persists ⇒ cleanest possible paper story.)
7. STUCK_PATIENCE: only worth cluster time if forensics finds
   FROZEN-but-would-escape patterns; otherwise skip (oscillation is
   invisible to the detector at any patience — forensics covers it).

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
