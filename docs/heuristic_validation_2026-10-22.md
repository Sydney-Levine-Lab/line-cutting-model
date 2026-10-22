# Deposit-goal heuristic correction

22 October 2026. Prepared for the `model-validation` worktree after checkpoint `504d9c0`. No commit or production simulation is part of this change.

## Change and scope

`src/water_collection_heuristic.jl` now estimates the remaining unit-cost actions to collect water and deposit it (`has-filled`). It excludes finish travel and returns zero for an agent that has already deposited, before interpreting the cleared water flags.

The guide builds a reverse breadth-first distance table over walkable positions and four water phases. It accounts for three collection actions, the deposit action, legal adjacent service squares, all well/tank alternatives, and the prohibition on moving with partially collected water. Wells and tanks are not walkable cells. Finish objects are irrelevant to this goal.

For a valid projected single-agent planning state, the table gives the exact remaining distance with the other agents held at their projected wall positions. Ignoring unprojected agent objects makes it a lower bound in a multi-agent state. A* remains the route planner. The guide supports positive `has-filled` goals with unit action costs; unsupported goals conservatively receive zero guidance.

Only the last wall/well/tank geometry is cached per heuristic instance. Cache reuse checks actual geometry contents. Unreachable states use `Inf`, avoiding integer overflow. Unsupported, empty, and already completed goals skip table construction.

The driver, PDDL domain, maps, prediction timing, final action selection, temperature, removal setting, and ghost policy are unchanged. The separately prepared Boltzmann comparison patch has not been applied.

## Exact-route validation

**6,535 / 6,535 checks passed** in Julia 1.10.10 with the pinned source environment:

- 231 route cases across all 28 canonical maps matched independent PDDL uniform-cost shortest-path search.
- All 726 legal action-value comparisons matched the exact reference.
- Returned plans were replayed through legal PDDL transitions and checked for goal completion.
- The guide matched the exact distance in the tested single-agent views and satisfied the consistency checks on sampled successors.
- Coverage included empty and partially collected water, full loads, deposited states, agents 1 and 8, snapshot obstacles, unreachable/changed/restored geometry, and unsupported goals.

The known regressions are corrected:

| State | Old route length | Corrected / exact |
|---|---:|---:|
| `no_line_1`, loaded lone agent at (1,3) | 4 | 2 |
| `yes_line_7`, loaded lone agent at (2,10) | 6 | 1 |

On the first state, corrected action values distinguish `down=-2` from `right=-4`, instead of tying them at -4. In the second state, the immediate fill remains correctly recognized by the action policy; the regression concerns the route estimator.

Completed-ghost controls on every map had identical views, available actions, and native policy values before and after the correction. Five matched seeds per map gave identical ghost choices. This establishes unchanged ghost rules at matched states; population trajectories can change because unfinished agents now receive corrected route values.

## Planning speed

The benchmark used 112 fixed route queries: agents 1 and 8 in empty and loaded phases at round-start positions on all 28 maps, with other agents projected to walls. Each query/method has three repetitions; the table sums per-query median times. Method order rotates. Compilation/startup is excluded. All timed routes happened to be reachable; unreachable cases were tested for correctness separately.

Cold trials include a fresh planner, heuristic, and distance-table construction. Warm trials reuse a geometry's table but still execute A*; they do not reuse memoized route answers.

| Measurement | Old | Corrected | Speed ratio |
|---|---:|---:|---:|
| Cold route queries | 20.793 s | 2.032 s | 10.23x |
| Warm route queries | 21.101 s | 2.019 s | 10.45x |
| Cold complete action-value decisions, 24 cases on 6 maps | 19.037 s | 1.674 s | 11.37x |

The corrected planner was faster on all 112 route queries. The median per-query cold speed ratio was 8.73x, with a range of 1.97x to 42.45x. The planner's expanded counter fell from 18,171 to 1,924, an 89.4% reduction. This API counter includes its initial count equally for both methods.

Both old and corrected route costs matched the uniform-cost oracle on these 112 timing queries, so the measured speed gain is on equal-quality results. The separate regression states above demonstrate the old estimator's correctness failures. The 26 maps without named finishes account for a 9.69x cold speed ratio; gains are therefore not confined to the two named-finish maps.

These measurements concern fixed planning queries on one machine, not full simulation runtimes, failure rates, universalization scores, or human fits. The final action-value benchmark was resumed from saved route measurements after correcting a qualified-function-name error in the benchmark harness; heuristic code and completed route measurements were unchanged.

## Reproduction

From the `model-validation` root, using Julia 1.10.10 and the existing pinned environment:

```bash
julia --project=src tests/heuristic_routes.jl
julia --project=src tests/heuristic_benchmark.jl
```

The harnesses use the sibling `cogsci-2026/src` as their default baseline. Set `LINES_BASELINE_SRC` to another frozen source directory if needed. The tests write to a new temporary directory by default; `LINES_TEST_OUTPUT` overrides it. `LINES_BENCH_OUTPUT` controls benchmark output placement. The benchmark supports `LINES_BENCH_REUSE_ROUTES=true` for resuming the decision panel from saved route results.

During this session, a SHA-verified copy of the unchanged baseline was used because the assistant could read `model-validation` files but could not read the archived worktree or parent Git metadata. File hashes guard application against concurrent source edits. No Git mutation is run by the application script.

Source SHA-256: old helper `c6ffebf85bece93a5a5202c62f5adca4b30ce674f21539aabed223f00db59e27`; corrected helper `d6d58a38d1bd2f36ac1ccd510fa817783d4b808ab6010498dcb6795a6c243e2c`.
