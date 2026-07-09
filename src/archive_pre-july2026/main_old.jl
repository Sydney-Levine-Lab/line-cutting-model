"
    Last attempt to get snapshot <> beginning timestep
    (Testing to see if memoized is causing the leak)
    April 2
"

# Entry point for running multi-agent water-collection simulations.
#
# Key modeling assumptions:
# - Agents act sequentially in a fixed random order, drawn once per run.
#   The order is a tie-breaking device: agents are meant to be making
#   simultaneous decisions, but we implement them sequentially for
#   technical reasons.
# - Each agent sees every other agent at their "last seen" position:
#   where that agent was right before they last moved. This makes the
#   planning problem symmetric — every agent is exactly one move behind
#   on every other agent, regardless of position in the order.
# - Level-0 reasoning: other agents are projected to walls (no prediction
#   of others' moves).
# - A* search with a task-specific heuristic for steps-to-go estimation.
# - Boltzmann action selection over action values.

using PDDL, PlanningDomains, SymbolicPlanners
using Random, Dates, Printf
using CSV

include(joinpath(@__DIR__, "water_collection_heuristic.jl"))
include(joinpath(@__DIR__, "utils.jl"))

# Enable array-valued fluents (e.g. wall grids) for this PDDL domain
PDDL.Arrays.@register()

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

const N_AGENTS    = 8                   # Must match map files
const RUNS        = env_int("RUNS", 5)  # Number of runs per map
const TIME_MAX    = 1000                # Upper bound on timesteps per run
const TEMPERATURE = 0.0001              # Boltzmann temperature

# Run identifier: label + timestamp
const RUN_LABEL     = get(ENV, "RUN_LABEL", "user_run")
const USE_TIMESTAMP = true
const RUN_ID = USE_TIMESTAMP ?
    RUN_LABEL * "_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS") :
    RUN_LABEL

# Random seed: BASE_SEED + MAP_SEED_OFFSET * map_number + run_number
const BASE_SEED       = 1234
const MAP_SEED_OFFSET = 10_000

const SRC_DIR     = @__DIR__
const DOMAIN_FILE = joinpath(SRC_DIR, "domain.pddl")
const MAPS_DIR    = joinpath(SRC_DIR, "maps")

# The experiment set: names correspond to files under maps/.
const MAP_FILES = [
    "no_line_1.pddl", "no_line_2.pddl", "no_line_3.pddl",
    "yes_line_7.pddl", "yes_line_8.pddl", "yes_line_9.pddl", "yes_line_10.pddl",
    "7esque.pddl", "9esque.pddl", "10esque.pddl",
    "maybe_4.pddl", "maybe_5.pddl", "maybe_6.pddl",
    "new_maybe_1.pddl", "new_maybe_2.pddl", "new_maybe_3.pddl",
    "new_maybe_4.pddl", "new_maybe_5.pddl", "new_maybe_6.pddl",
    "no_line_A.pddl", "no_line_B.pddl", "no_line_C.pddl", "no_line_D.pddl",
    "yes_line_B.pddl", "yes_line_C.pddl", "yes_line_D.pddl", "yes_line_E.pddl", "yes_line_F.pddl",
]

const OUTPUT_DIR        = joinpath(SRC_DIR, "..", "data", "simulations", RUN_ID, "raw")
const TRAJECTORY_DIR    = joinpath(OUTPUT_DIR, "trajectories")
const SNAPSHOT_DIR      = joinpath(OUTPUT_DIR, "run_info")
const SAVE_TRAJECTORIES = env_bool("SAVE_TRAJECTORIES", false)
const WRITE_SNAPSHOT    = env_bool("WRITE_SNAPSHOT", false)
const VERBOSE           = env_bool("VERBOSE", true)

# --------------------------------------------------------------------
# Precomputed terms
# --------------------------------------------------------------------

const AGENTS     = [Const(Symbol("agent$n")) for n in 1:N_AGENTS]
const XLOC       = [Compound(:xloc, Term[a]) for a in AGENTS]
const YLOC       = [Compound(:yloc, Term[a]) for a in AGENTS]
const HAS_FILLED = [Compound(Symbol("has-filled"), Term[a]) for a in AGENTS]
const HAS_WATER1 = [Compound(Symbol("has-water1"), Term[a]) for a in AGENTS]
const HAS_WATER2 = [Compound(Symbol("has-water2"), Term[a]) for a in AGENTS]
const HAS_WATER3 = [Compound(Symbol("has-water3"), Term[a]) for a in AGENTS]

# --------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------

"""
Build the steps-to-go estimator used for action selection (A* + task heuristic).
"""
function build_steps_to_go_estimator()
    heuristic         = WaterCollectionHeuristic()
    planner           = SymbolicPlanners.AStarPlanner(heuristic)
    planner_heuristic = SymbolicPlanners.PlannerHeuristic(planner)
    return  planner_heuristic # DEBUG: info leakage APRIL 2 TODO
    #SymbolicPlanners.memoized(planner_heuristic)
end

"""
Build per-agent Boltzmann policies from a steps-to-go estimator.
Each agent's goal is to reach a state where it has filled a tank.
"""
function build_agent_policies(domain::Domain, steps_to_go)
    policies = Vector{BoltzmannPolicy}(undef, N_AGENTS)
    for n in 1:N_AGENTS
        goal         = SymbolicPlanners.MinStepsGoal(Term[HAS_FILLED[n]])
        value_policy = SymbolicPlanners.FunctionalVPolicy(steps_to_go, domain, goal)
        policies[n]  = SymbolicPlanners.BoltzmannPolicy(value_policy, TEMPERATURE)
    end
    return policies
end

# --------------------------------------------------------------------
# World projection and transitions
# --------------------------------------------------------------------

"""
    project_others_to_walls(real_state, last_seen, k, domain)

Build a single-agent planning view for agent `k`.

- Agent k's own position and task status come from `real_state`
  (the actual current world).
- Other agents are placed as walls at their `last_seen` positions
  (where each agent was right before they last moved).
- Non-agent objects (wells, tanks, finishes) are static.
"""
function project_others_to_walls(real_state::State,
                                  last_seen::Vector{Tuple{Int,Int}},
                                  k::Int, domain::Domain)
    objtypes = PDDL.get_objtypes(real_state)

    # Remove other agents from the object list
    for n in 1:N_AGENTS
        n == k && continue
        delete!(objtypes, AGENTS[n])
    end

    # Add walls where other agents were last seen
    walls = copy(real_state[pddl"(walls)"])
    for n in 1:N_AGENTS
        n == k && continue
        x, y = last_seen[n]
        walls[y, x] = true
    end

    # Build fluent dictionary for the reduced state
    fluents = Dict{Term, Any}()

    # Non-agent objects (static)
    for (obj, objtype) in objtypes
        objtype == :agent && continue
        fluents[Compound(:xloc, Term[obj])] = real_state[Compound(:xloc, Term[obj])]
        fluents[Compound(:yloc, Term[obj])] = real_state[Compound(:yloc, Term[obj])]
    end

    fluents[pddl"(walls)"] = walls

    # Agent k's own position and task status from the REAL current world
    fluents[XLOC[k]]       = real_state[XLOC[k]]
    fluents[YLOC[k]]       = real_state[YLOC[k]]
    fluents[HAS_FILLED[k]] = real_state[HAS_FILLED[k]]
    fluents[HAS_WATER1[k]] = real_state[HAS_WATER1[k]]
    fluents[HAS_WATER2[k]] = real_state[HAS_WATER2[k]]
    fluents[HAS_WATER3[k]] = real_state[HAS_WATER3[k]]

    return initstate(domain, objtypes, fluents)
end

"""
Apply `act` to `state`. If preconditions fail (collision etc.),
treat it as "no move" and return the state unchanged.
"""
function safe_transition(domain::Domain, state::State, act)
    try
        return transition(domain, state, act)
    catch
        return state
    end
end

# --------------------------------------------------------------------
# Simulation step
# --------------------------------------------------------------------

"""
    simulation_step(state, policies, domain, order, last_seen)

Run one simulation timestep.

Each agent sees every other agent at their "last seen" position —
where that agent was right before they last moved. This makes the
planning problem symmetric: every agent is exactly one move behind
on every other agent, regardless of position in the order.

The update rule for each agent k:
1. Capture k's current (pre-move) position into `last_seen[k]`.
2. Plan from `last_seen` positions of others (walls).
3. Move k in the real world.

Returns the updated world state. Mutates `last_seen` in place.
"""
function simulation_step(state::State,
                         policies::AbstractVector{<:BoltzmannPolicy},
                         domain::Domain,
                         order::Vector{Int},
                         last_seen::Vector{Tuple{Int,Int}})
    for idx in 1:N_AGENTS
        k = order[idx]

        # Capture k's pre-move position (this is what others will see)
        last_seen[k] = (state[XLOC[k]], state[YLOC[k]])

        # Plan from last_seen positions of others
        planning_state = project_others_to_walls(state, last_seen, k, domain)
        act = SymbolicPlanners.get_action(policies[k], planning_state)
        state = safe_transition(domain, state, act)
    end

    return state
end

# --------------------------------------------------------------------
# Main simulation loop
# --------------------------------------------------------------------

function run_simulations()
    mkpath(OUTPUT_DIR)
    if SAVE_TRAJECTORIES
        mkpath(TRAJECTORY_DIR)
    end

    println("Starting simulations:")
    println("  maps = $(length(MAP_FILES))")
    println("  runs per map = $(RUNS)")
    println("  threads = $(Threads.nthreads())")
    println("  output dir = $(OUTPUT_DIR)")
    println("  save trajectories = $(SAVE_TRAJECTORIES)")
    println()

    if WRITE_SNAPSHOT
        write_snapshot!(SNAPSHOT_DIR, SRC_DIR)
    end

    # Parallelize over maps
    Threads.@threads for map_index in eachindex(MAP_FILES)
        map = MAP_FILES[map_index]
        map_name = replace(map, ".pddl" => "")

        if VERBOSE
            tprintln("Starting map $map_name (thread $(Threads.threadid()))")
        end

        # Load domain and build policies
        domain      = PlanningDomains.load_domain(DOMAIN_FILE)
        steps_to_go = build_steps_to_go_estimator()
        policies    = build_agent_policies(domain, steps_to_go)

        # Load initial state for this map
        problem       = PlanningDomains.load_problem(joinpath(MAPS_DIR, map))
        initial_state = initstate(domain, problem)

        results = Vector{NamedTuple}(undef, RUNS)

        for run in 1:RUNS
            if VERBOSE
                tprintln("[$(Threads.threadid())] map=$(map) run=$(run)")
            end

            # Seed per (map_index, run) for reproducibility
            seed_this_run = BASE_SEED + MAP_SEED_OFFSET * map_index + run
            Random.seed!(seed_this_run)

            # Draw the play order once for this run
            order = randperm(N_AGENTS)

            # Initialize last_seen: where each agent was right before
            # they last moved. At t=0 nobody has moved, so use initial positions.
            last_seen = [(initial_state[XLOC[n]], initial_state[YLOC[n]])
                         for n in 1:N_AGENTS]

            state           = initial_state
            state_t_minus_1 = state
            state_t_minus_2 = state
            agent_filled    = fill(0, N_AGENTS)

            traj_path = joinpath(TRAJECTORY_DIR, "trajectory_$(map_name)_run$(run).log")

            elapsed_run = @elapsed begin
                if SAVE_TRAJECTORIES
                    open(traj_path, "w") do io
                        @printf(io, "# map=%s run=%d seed=%d temp=%.6g order=%s\n",
                                map, run, seed_this_run, TEMPERATURE, string(order))
                        println(io, "# t=0")
                        show(io, state); println(io)

                        for t in 1:TIME_MAX
                            state = simulation_step(state, policies, domain,
                                                    order, last_seen)
                            println(io, "# t=$t")
                            show(io, state); println(io)

                            for n in 1:N_AGENTS
                                if agent_filled[n] == 0 && state[HAS_FILLED[n]]
                                    agent_filled[n] = t
                                end
                            end

                            if t >= 3 && state == state_t_minus_1 && state_t_minus_1 == state_t_minus_2
                                fill!(agent_filled, -1)
                                break
                            end
                            if all(>(0), agent_filled)
                                break
                            end

                            state_t_minus_2 = state_t_minus_1
                            state_t_minus_1 = state
                        end
                    end
                else
                    for t in 1:TIME_MAX
                        state = simulation_step(state, policies, domain,
                                                order, last_seen)

                        for n in 1:N_AGENTS
                            if agent_filled[n] == 0 && state[HAS_FILLED[n]]
                                agent_filled[n] = t
                            end
                        end

                        if t >= 3 && state == state_t_minus_1 && state_t_minus_1 == state_t_minus_2
                            fill!(agent_filled, -1)
                            break
                        end
                        if all(>(0), agent_filled)
                            break
                        end

                        state_t_minus_2 = state_t_minus_1
                        state_t_minus_1 = state
                    end
                end
            end

            # Mark unfinished agents
            for n in 1:N_AGENTS
                if agent_filled[n] == 0
                    agent_filled[n] = -1
                end
            end

            results[run] = (
                run                 = run,
                agent_1             = agent_filled[1],
                agent_2             = agent_filled[2],
                agent_3             = agent_filled[3],
                agent_4             = agent_filled[4],
                agent_5             = agent_filled[5],
                agent_6             = agent_filled[6],
                agent_7             = agent_filled[7],
                agent_8             = agent_filled[8],
                map                 = map_name,
                seed                = seed_this_run,
                temperature         = TEMPERATURE,
                time_max            = TIME_MAX,
                n_agents            = N_AGENTS,
                run_elapsed_seconds = elapsed_run,
                order               = string(order),
            )
        end

        csv_name = joinpath(OUTPUT_DIR, map_name * ".csv")
        CSV.write(csv_name, results)
        println("Finished map $map_name → wrote $(basename(csv_name))")
    end
end

# Run if called as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulations()
end