## Running the modeling scripts

### main_modeling_line (paper runner)
- Purpose: batch-run experiments across many maps, writing results to current (non-old) folders.
- Actions per step: sequential per-agent rollout using a single-agent projection.
- Outputs:
  - State histories: `state_histories/state_history_<map>_<noise>_iteration<k>.txt`
  - CSV data: `maps/data/<map>_<noise>.csv` (appended per run)
- How to run:
  1) Ensure `PDDL.jl` and `SymbolicPlanners.jl` deps are available in your Julia env.
  2) From the project root, run: `julia main_modeling_line`
  3) Modify `all_problems`, `T`, `boltzmann_policy_parameters`, or iterations as needed.

### automated_modeling_line (legacy automated runner)
- Purpose: older batch runner that writes to `oldmodeling_*` locations.
- Actions per step: builds a vector of actions and calls a multi-agent transition
  with simple collision handling.
- Outputs:
  - State histories: `oldmodeling_statehistories/state_history_<map>_<noise>_iteration<k>.txt`
  - CSV data: `maps/oldmodeling_data/<map>_<noise>.csv`
- How to run:
  1) From the project root, run: `julia automated_modeling_line`
  2) Adjust the map list / parameters near the top if needed.

### Notes
- Agents: `N=8` across all maps; must match the problem files.
- Termination:
  - Early stop when all agents satisfy `has-filled`.
  - Detects looping via repeated states; marks agents `-1` and stops.
- Policies: each agent uses a Boltzmann policy over a value function computed by
  A* with `WellTankHeuristic`, memoized for performance.

