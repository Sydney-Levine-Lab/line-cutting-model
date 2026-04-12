" April 8: Not tested but should be too slow for depth2 (same spirit as the main2 and main_next in failed/)"
# Entry point for running multi-agent water-collection simulations.
#
# Key modeling assumptions:
# - Agents act sequentially in a fixed random order, drawn once per run.
#   The order is a tie-breaking device: agents are meant to be making
#   simultaneous decisions, but we implement them sequentially for
#   technical reasons.
# - At depth 0, all agents plan from the beginning-of-timestep snapshot
#   (others become walls at their start-of-round positions).
# - At depth 1, agents predict one L0 step for agents before them in
#   the order, giving the most accurate picture of the world at the
#   moment they act.
# - At depth >= 2, agents strictly extend depth 1: they use depth-1
#   walls for collision avoidance (which actions are valid), then
#   evaluate each candidate action by simulating (depth-1) additional
#   rounds of L0 play and scoring the post-move position against those
#   future walls. Falls back to depth-1 if lookahead fails.
# - A* search with a task-specific heuristic for steps-to-go estimation.
# - Boltzmann action selection at depth 0-1; greedy best-action at depth >= 2.

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
const DEPTH       = env_int("DEPTH", 0) # Reasoning depth (0 = blind L0)

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
    return SymbolicPlanners.memoized(planner_heuristic)
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
    project_others_to_walls(state, k, domain)

Build a single-agent planning view for agent `k`:
all other agents become walls at their positions in `state`.
"""
function project_others_to_walls(state::State, k::Int, domain::Domain)
    objtypes = PDDL.get_objtypes(state)

    agent_locs = Array{Tuple{Int,Int}}(undef, N_AGENTS - 1)
    idx = 1
    for n in 1:N_AGENTS
        n == k && continue
        agent_locs[idx] = (state[XLOC[n]], state[YLOC[n]])
        idx += 1
        delete!(objtypes, AGENTS[n])
    end

    walls = copy(state[pddl"(walls)"])
    for (x, y) in agent_locs
        walls[y, x] = true
    end

    fluents = Dict{Term, Any}()
    for (obj, objtype) in objtypes
        objtype == :agent && continue
        fluents[Compound(:xloc, Term[obj])] = state[Compound(:xloc, Term[obj])]
        fluents[Compound(:yloc, Term[obj])] = state[Compound(:yloc, Term[obj])]
    end

    fluents[pddl"(walls)"] = walls
    fluents[XLOC[k]]       = state[XLOC[k]]
    fluents[YLOC[k]]       = state[YLOC[k]]
    fluents[HAS_FILLED[k]] = state[HAS_FILLED[k]]
    fluents[HAS_WATER1[k]] = state[HAS_WATER1[k]]
    fluents[HAS_WATER2[k]] = state[HAS_WATER2[k]]
    fluents[HAS_WATER3[k]] = state[HAS_WATER3[k]]

    return initstate(domain, objtypes, fluents)
end

"""
    build_single_agent_view(ref_state, k, domain, k_pos, wall_positions)

Build a single-agent PDDL state for agent `k`:
- k is placed at `k_pos`
- walls at each position in `wall_positions`
- non-agent objects from `ref_state`

Used during depth >= 2 lookahead to evaluate k at a hypothetical
post-move position against predicted future walls.
"""
function build_single_agent_view(ref_state::State, k::Int, domain::Domain,
                                  k_pos::Tuple{Int,Int},
                                  wall_positions::Vector{Tuple{Int,Int}})
    objtypes = PDDL.get_objtypes(ref_state)
    for n in 1:N_AGENTS
        n == k && continue
        delete!(objtypes, AGENTS[n])
    end

    walls = copy(ref_state[pddl"(walls)"])
    for (x, y) in wall_positions
        walls[y, x] = true
    end

    fluents = Dict{Term, Any}()
    for (obj, objtype) in objtypes
        objtype == :agent && continue
        fluents[Compound(:xloc, Term[obj])] = ref_state[Compound(:xloc, Term[obj])]
        fluents[Compound(:yloc, Term[obj])] = ref_state[Compound(:yloc, Term[obj])]
    end

    fluents[pddl"(walls)"] = walls
    fluents[XLOC[k]] = k_pos[1]
    fluents[YLOC[k]] = k_pos[2]
    fluents[HAS_FILLED[k]] = ref_state[HAS_FILLED[k]]
    fluents[HAS_WATER1[k]] = ref_state[HAS_WATER1[k]]
    fluents[HAS_WATER2[k]] = ref_state[HAS_WATER2[k]]
    fluents[HAS_WATER3[k]] = ref_state[HAS_WATER3[k]]

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
# Prediction
# --------------------------------------------------------------------

"""
    predict_l0_step(predicted_state, agent_idx, domain, policies)

Predict one L0 step for agent `agent_idx`:
project others to walls, pick an action, apply it.
Returns the updated state.
"""
function predict_l0_step(predicted_state::State, agent_idx::Int,
                          domain::Domain,
                          policies::AbstractVector{<:BoltzmannPolicy})
    l0_view = project_others_to_walls(predicted_state, agent_idx, domain)
    act = SymbolicPlanners.get_action(policies[agent_idx], l0_view)
    return safe_transition(domain, predicted_state, act)
end

# --------------------------------------------------------------------
# Depth 1: best picture of the current world
# --------------------------------------------------------------------

"""
    build_depth1_state(initial_state, k, domain, policies, order, idx_in_order)

Build the planning state for agent `k` at reasoning depth 1.

Predict one L0 step for each agent before k in the order. This gives k
the most accurate picture of the world at the moment k acts: agents
before k at their predicted post-move positions, agents after k at
their actual current positions.
"""
function build_depth1_state(initial_state::State, k::Int,
                             domain::Domain,
                             policies::AbstractVector{<:BoltzmannPolicy},
                             order::Vector{Int}, idx_in_order::Int)
    predicted_state = initial_state
    for j_idx in 1:(idx_in_order - 1)
        j = order[j_idx]
        predicted_state = predict_l0_step(predicted_state, j, domain, policies)
    end
    return project_others_to_walls(predicted_state, k, domain)
end

# --------------------------------------------------------------------
# Depth >= 2: depth-1 for collision avoidance + lookahead for routing
# --------------------------------------------------------------------

"""
    choose_action_with_lookahead(initial_state, k, domain, policies, order,
                                 idx_in_order, depth)

Choose an action for agent `k` at reasoning depth >= 2.

This strictly extends depth 1: k has all the depth-1 knowledge (accurate
picture of the current world) PLUS a forward-looking evaluation of each
candidate action.

1. Build the depth-1 predicted world (best picture of now). This
   determines which actions are valid and avoids real collisions.
2. Get candidate actions from the depth-1 planning state.
3. From the depth-1 predicted world, simulate (depth-1) additional
   rounds of L0 play for all other agents (in play order, skipping k)
   to predict where others will be in the future.
4. For each candidate action, mentally move k there and evaluate the
   resulting position with A* against the future walls.
5. Pick the action with the best lookahead value.
6. If lookahead fails for all actions, fall back to the depth-1
   Boltzmann policy choice (strictly no worse than depth 1).
"""
function choose_action_with_lookahead(initial_state::State, k::Int,
                                       domain::Domain,
                                       policies::AbstractVector{<:BoltzmannPolicy},
                                       order::Vector{Int},
                                       idx_in_order::Int,
                                       depth::Int)
    # --- Step 1: build depth-1 predicted world ---
    depth1_predicted = initial_state
    for j_idx in 1:(idx_in_order - 1)
        j = order[j_idx]
        depth1_predicted = predict_l0_step(depth1_predicted, j, domain, policies)
    end
    depth1_view = project_others_to_walls(depth1_predicted, k, domain)

    # --- Step 2: get candidate actions ---
    candidate_actions = collect(PDDL.available(domain, depth1_view))

    if isempty(candidate_actions)
        return SymbolicPlanners.get_action(policies[k], depth1_view)
    end

    # --- Step 3: predict future positions ---
    # From the depth-1 predicted world, simulate (depth-1) more rounds.
    # Each round: complete the current cycle (agents after k), then
    # start the next (agents before k) — always in play order, skipping k.
    future_state = depth1_predicted
    for _round in 1:(depth - 1)
        for j_idx in (idx_in_order + 1):N_AGENTS
            j = order[j_idx]
            future_state = predict_l0_step(future_state, j, domain, policies)
        end
        for j_idx in 1:(idx_in_order - 1)
            j = order[j_idx]
            future_state = predict_l0_step(future_state, j, domain, policies)
        end
    end

    # Collect future wall positions
    future_walls = Tuple{Int,Int}[]
    for n in 1:N_AGENTS
        n == k && continue
        push!(future_walls, (future_state[XLOC[n]], future_state[YLOC[n]]))
    end

    # --- Step 4: evaluate each candidate action ---
    best_action = nothing
    best_value = -Inf
    goal = SymbolicPlanners.MinStepsGoal(Term[HAS_FILLED[k]])

    for act in candidate_actions
        # Mentally apply action to get k's new position
        test_state = safe_transition(domain, depth1_view, act)
        k_new_pos = (test_state[XLOC[k]], test_state[YLOC[k]])

        # Build lookahead view: k at new position, future walls
        lookahead_view = build_single_agent_view(initial_state, k, domain,
                                                   k_new_pos, future_walls)

        # Evaluate with bounded A*
        planner = SymbolicPlanners.AStarPlanner(WaterCollectionHeuristic(),
                                                 max_nodes=2^16)
        sol = planner(domain, lookahead_view, goal)
        value = if sol.status == :success
            -length(sol.plan)
        else
            -Inf
        end

        if value > best_value
            best_value = value
            best_action = act
        end
    end

    # --- Step 5: fall back to depth-1 if lookahead failed for all actions ---
    if best_value == -Inf
        return SymbolicPlanners.get_action(policies[k], depth1_view)
    end

    return best_action
end

# --------------------------------------------------------------------
# Simulation step
# --------------------------------------------------------------------

"""
    simulation_step(state, policies, domain, order)

Run one simulation timestep.

- Depth 0: plan from beginning-of-timestep snapshot (Boltzmann policy).
- Depth 1: predict L0 steps for agents before k, then Boltzmann policy.
- Depth >= 2: depth-1 walls for collision avoidance, action enumeration
  with future evaluation for route selection. Falls back to depth-1
  choice if lookahead fails.
"""
function simulation_step(state::State,
                         policies::AbstractVector{<:BoltzmannPolicy},
                         domain::Domain,
                         order::Vector{Int})
    initial_state = state

    for idx in 1:N_AGENTS
        k = order[idx]

        if DEPTH == 0
            planning_state = project_others_to_walls(initial_state, k, domain)
            act = SymbolicPlanners.get_action(policies[k], planning_state)
        elseif DEPTH == 1
            planning_state = build_depth1_state(initial_state, k, domain,
                                                 policies, order, idx)
            act = SymbolicPlanners.get_action(policies[k], planning_state)
        else
            act = choose_action_with_lookahead(initial_state, k, domain,
                                                policies, order, idx, DEPTH)
        end

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
    println("  depth = $(DEPTH)")
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

            state           = initial_state
            state_t_minus_1 = state
            state_t_minus_2 = state
            agent_filled    = fill(0, N_AGENTS)

            traj_path = joinpath(TRAJECTORY_DIR, "trajectory_$(map_name)_run$(run).log")

            elapsed_run = @elapsed begin
                if SAVE_TRAJECTORIES
                    open(traj_path, "w") do io
                        @printf(io, "# map=%s run=%d seed=%d temp=%.6g depth=%d order=%s\n",
                                map, run, seed_this_run, TEMPERATURE, DEPTH, string(order))
                        println(io, "# t=0")
                        show(io, state); println(io)

                        for t in 1:TIME_MAX
                            state = simulation_step(state, policies, domain, order)
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
                        state = simulation_step(state, policies, domain, order)

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
                depth               = DEPTH,
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