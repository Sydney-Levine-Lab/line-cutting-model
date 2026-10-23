# Prediction information timing correction

23 October 2026. This change follows the deposit-goal heuristic checkpoint
`d72247a`. It changes the canonical `src/main.jl` prediction functions and
adds `tests/prediction_timing.jl`. It is a separate, uncommitted correction.

## Defect and correction

Actual agents choose from the common round-start snapshot, with their
actions resolved in priority order. Previously, imagined agents were
offered the progressively updated execution state. Modeled L0 agents
could therefore react to earlier imagined moves that actual L0 agents
could not observe. Nested models could also advance an already-advanced
prefix again, and later rollout agents could react to the focal agent's
candidate action as if they had observed it.

`predict_step` now receives the round-start snapshot separately from its
`execution_state` keyword. A modeled agent chooses from the snapshot
(with private lower-level predictions when appropriate), then applies
only its own chosen action to the execution state. All first-round
predictions in `build_planning_state_d1` and `evaluate_action` share the
initial snapshot. Each later `simulate_round` uses its own fresh
round-start snapshot. Action resolution remains in priority order.

Private lower-level forecasts start from the same snapshot as the agent
making them. They do not inherit a prefix already advanced by an outer
simulation. This also prevents an imagined predecessor from collecting
multiple water phases within a single round of private prediction.

Higher-level agents still use predictions of others when choosing.
Correcting the information rule does not eliminate the intended
level-k mismatch: actual L1 opponents may act differently from an L0
model of those opponents.

## Scope

The actual `simulation_step`, route heuristic, domain, maps, agent goals,
temperature, RNG interface, rollout final selector, ghost behavior,
removal implementation and stopping rule are unchanged. The standalone
root `main.jl` legacy copy is unchanged; use the canonical entry point
`julia --project=src -t 1 src/main.jl`.

No shared real-world future random draws are supplied to predictions.
Seeded tests couple two diagnostic executions only to check that an
imagined policy obeys the same rule as the actual policy it models.
Production forecasts retain their own sampling calls.

## Validation

**1,077 / 1,077 checks passed** with the optional frozen pre-fix driver
enabled in the pinned Julia 1.10.10 environment.

The focused suite covers:

- A controlled queue where L0 agents 2 and 3 must see their predecessors'
  original occupied cells, even after those predecessors move.
- Priority-based collision resolution after intentions have been formed.
- Private models through level 3, including reversed and nonconsecutive
  agent orders, without advancing an actor's own state twice.
- Both earlier and later rollout agents, with different focal candidates,
  checking that the candidates do not become observations in round 1.
- A fresh snapshot for subsequent imagined rounds.
- Water collection phase progression within private models.
- Native Boltzmann L0 imagined rounds against a separate intention-then-
  execution oracle on all 28 maps, with three seeds per map.
- Native modeled L1 against actual shallow L1 on six maps, and modeled
  L2 against actual shallow L2 on three maps, with forward, reversed and
  nonconsecutive orders and two seeds per case.
- An optional frozen-driver regression on all 28 maps, comparing actual
  L0 states and subsequent RNG draws for two rounds with both empty
  agents and deposited ghosts. The same frozen driver reproduces the
  original information leak and doubly advanced prefix.

The tests compare states and RNG draws, not just completion summaries.
They do not run or overwrite production datasets and do not establish
effects on welfare, human fit, or the frequency of long-run jams.

## Reproduction

From the `model-validation` repository root:

```bash
julia --project=src tests/prediction_timing.jl
```

The normal suite needs no Git access or historical checkout. To include
the optional pre-fix L0 comparison, first save the driver from the
heuristic checkpoint and provide its path:

```bash
git show d72247a:src/main.jl > /tmp/lines_prediction_before.jl
LINES_PREDICTION_BASELINE=/tmp/lines_prediction_before.jl \
  julia --project=src tests/prediction_timing.jl
```

`LINES_TARGET_SRC` can point the tests at an isolated source copy. During
development, the pinned Julia 1.10.10 environment was loaded offline with
startup and compiled modules disabled. Session receipts and the frozen
driver are in the task's `work/prediction_fix/outputs` and `baseline`
directories.

The cluster heuristic comparison remains on `d72247a`. Do not update that
checkout while its queued jobs still depend on it. No prediction-fix
cluster runs or new commits are part of this correction.
