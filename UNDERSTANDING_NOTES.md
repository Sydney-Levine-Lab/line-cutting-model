# List of files

### main_modeling_line
Main file, batch running xp --- see RUNNING_MODELING.md
# modify_state(State, k)
Project to single-agent view for agent #k, by deleting others and turning their last position into walls
# MultiAgentDomain
Commented out: for multi agent view
# Planning infrastructure
WellTankHeuristic -> A* Planner -> Policy -> Boltzmann Policy
- WellTankHeuristic: estimates cost to water tank filling
- A* Planner: finds optimal path using the heuristic [This is a famous AI algo]
- Policy: agent's decision making depends on this
- Boltzmann: with noise for exploration (small, see below)
- Memoized: performance optimzation technique that caches expensive function result to avoid recalculating them
# Output
- state history per run (state_histories/)
- per-agent fill times (maps/data/)
# Other
- Loads 29 PDDL maps w/ weird names
- Noise = 0.0001 each time
- 5 iterations
- Upper bound on timesteps: 1000