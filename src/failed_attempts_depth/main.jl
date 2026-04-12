# Entry point for running multi-agent water-collection simulations.

using PDDL, PlanningDomains, SymbolicPlanners
# CHANGED: removed Mar 3
## Monkey-patch: restore lazy generator behavior from SymbolicPlanners v0.1.9
## (v0.1.27 changed this to Dict, which can cause issues with Inf/NaN values)
#import SymbolicPlanners: get_action_values, FunctionalVPolicy
#SymbolicPlanners.get_action_values(sol::FunctionalVPolicy, state::State) =
#    (act => SymbolicPlanners.get_value(sol, state, act) 
#     for act in PDDL.available(sol.domain, state))
using Random, Dates, Printf
using CSV

include(joinpath(@__DIR__, "water_collection_heuristic.jl"))
include(joinpath(@__DIR__, "utils.jl"))

# Enable array-valued fluents (e.g. wall grids) for this PDDL domain
PDDL.Arrays.@register()

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

const N_AGENTS    = 8                   #  Must match map files
const RUNS        = env_int("RUNS", 5)  # Number of runs per map
const TIME_MAX    = 1000                # Upper bound on timesteps for a single run
const TEMPERATURE = 0.0001              # Boltzmann temperature

# Run is identified by label written at launch and, by default, a time stamp
const RUN_LABEL     = get(ENV, "RUN_LABEL", "user_run")
const USE_TIMESTAMP = true
const RUN_ID = USE_TIMESTAMP ?
    RUN_LABEL * "_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS") :
    RUN_LABEL

# Random seed for reproducibility (= BASE_SEED + MAP_SEED_OFFSET * map_number + run_number)
const BASE_SEED         = 1234
const MAP_SEED_OFFSET   = 10_000

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
# Note: "yes_line_A.pddl" not used in stimuli

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
# Modeling knobs
# --------------------------------------------------------------------

# How agents are scheduled within a timestep.
# Order is drawn once per run and agents observe it.
#   "fixed"  = deterministic order 1..N_AGENTS
#   "random" = random permutation, drawn once at start of each run
const ORDER = get(ENV, "ORDER", "random")

# How sophisticated agents' model of others is (depth of forward prediction).
#   0 = L0: others are static walls at their last-observed positions
#   1 = predict one full round of L0 play for all others (in known order),
#       project walls at predicted positions, plan from current position
#   2+= predict D rounds ahead
#
# At depth >= 1, predictions use the known order and observed positions.
# This is deterministic (no sampling), fast, and cognitively interpretable.
const REASONING_DEPTH = env_int("REASONING_DEPTH", 0)

# How agents search
#   "astar"         = A*
#   "proba_astar"   = probabilistic A*
#   "weighted_astar" = weighted A*
const PLANNING = get(ENV, "PLANNING", "astar")

# Planning parameters
const PLANNING_H_MULT       = env_float("PLANNING_H_MULT", 2.0)
const PLANNING_SEARCH_NOISE = env_float("PLANNING_SEARCH_NOISE", 0.5)

# --------------------------------------------------------------------
# Agent     
# --------------------------------------------------------------------
# Key assumptions:
# - state evaluation: A* + task-specific heuristic → steps-to-go estimate
# - policy: Boltzmann sampling from action values

"""
Build the steps-to-go estimator used for action selection.
"""
function build_steps_to_go_estimator()
    inner_heuristic   = WaterCollectionHeuristic()                        # lower-bound estimate of steps-to-go (task-specific heuristic)
    # Search guided by heuristic; simulations vary in what type of planner is used
    if PLANNING == "astar"
        planner = SymbolicPlanners.AStarPlanner(inner_heuristic)
    elseif PLANNING == "weighted_astar"
        # Greedier A*; more weight on the heuristic term
        planner = SymbolicPlanners.WeightedAStarPlanner(inner_heuristic, PLANNING_H_MULT)
    elseif PLANNING == "proba_astar"
         # Probabilistic A*: samples which node to expand, controlled by search_noise
        planner = SymbolicPlanners.ProbAStarPlanner(inner_heuristic; search_noise = PLANNING_SEARCH_NOISE)
    else
        error("Unknown PLANNING: $PLANNING")
    end
    planner_heuristic = SymbolicPlanners.PlannerHeuristic(planner)        # estimate steps-to-go from a state
    return SymbolicPlanners.memoized(planner_heuristic)                   # cache for speed
end

"""
Build per-agent Boltzmann policies from a steps-to-go estimator.

Each agent's goal is to reach a state where it has filled a tank.
"""
function build_agent_policies(domain::Domain, steps_to_go)
    policies = Vector{BoltzmannPolicy}(undef, N_AGENTS)
    for n in 1:N_AGENTS
        agent         = AGENTS[n]
        goal          = SymbolicPlanners.MinStepsGoal(Term[HAS_FILLED[n]])
        
        
        #value_policy  = SymbolicPlanners.HeuristicVPolicy(steps_to_go, domain, goal) 
        #CHANGED from FunctionalVPolicy (which only works with AStar bcz that only has one argument)
        # TODO: check that doesn't impact AStar runs
        # It does...


        # CHANGED: trying to use FunctionalVPolicy everywhere
        #value_fn     = state -> -steps_to_go(domain, state, goal)
        #value_policy = SymbolicPlanners.FunctionalVPolicy(value_fn, domain, goal)
        # This never works, just giving -1 everywhere

        # CHANGED Mar 3: back to original
        value_policy = SymbolicPlanners.FunctionalVPolicy(steps_to_go, domain, goal)


        policies[n]   = SymbolicPlanners.BoltzmannPolicy(value_policy, TEMPERATURE)
    end
    return policies
end

# --------------------------------------------------------------------
# Agent reasoning model and information
# --------------------------------------------------------------------
# Key assumptions:
# - agents act sequentially, in fixed order (agent 1, then 2, etc.)
# - full observability: later agents see earlier moves
# - level-0 projection: agents plan by treating others as static obstacles


"""
Draw the within-round order of agents (once per run).
"""
function draw_run_order()
    if ORDER == "fixed"
        return collect(1:N_AGENTS)
    elseif ORDER == "random"
        return randperm(N_AGENTS)
    else
        error("Unknown ORDER: $ORDER")
    end
end

"""
Extract (x, y) positions for all agents from a PDDL state.
"""
function get_all_positions(state::State)
    return [(state[XLOC[n]], state[YLOC[n]]) for n in 1:N_AGENTS]
end

"""
Build a PDDL state where agent `k` is at its current position and all other
agents are replaced by walls at the positions given in `positions`.

`positions` is a Vector of (x, y) tuples, one per agent.
"""
function project_others_to_walls_at(state::State, k::Int, domain::Domain,
                                     positions::Vector{Tuple{Int,Int}})
    objtypes = PDDL.get_objtypes(state)

    # Delete all agents except k, record wall positions from `positions`
    agent_locs = Vector{Tuple{Int,Int}}()
    for n in 1:N_AGENTS
        if n == k
            continue
        end
        push!(agent_locs, positions[n])
        delete!(objtypes, AGENTS[n])
    end

    walls = copy(state[pddl"(walls)"])
    for (x, y) in agent_locs
        walls[y, x] = true
    end

    fluents = Dict{Term, Any}()
    for (obj, objtype) in objtypes
        if objtype == :agent
            continue
        end
        fluents[Compound(:xloc, Term[obj])] = state[Compound(:xloc, Term[obj])]
        fluents[Compound(:yloc, Term[obj])] = state[Compound(:yloc, Term[obj])]
    end

    fluents[pddl"(walls)"] = walls
    fluents[XLOC[k]] = state[XLOC[k]]
    fluents[YLOC[k]] = state[YLOC[k]]
    fluents[HAS_FILLED[k]] = state[HAS_FILLED[k]]
    fluents[HAS_WATER1[k]] = state[HAS_WATER1[k]]
    fluents[HAS_WATER2[k]] = state[HAS_WATER2[k]]
    fluents[HAS_WATER3[k]] = state[HAS_WATER3[k]]

    return initstate(domain, objtypes, fluents)
end

"""
Apply `act` to `state`. If the action is invalid (preconditions fail),
treat it as "no move" and keep the state unchanged.
"""
function safe_transition(domain::Domain, state::State, act)
    try
        return transition(domain, state, act)
    catch
        # collision or other invalid move → agent bumps and stays put
        return state
    end
end

"""
Forward-looking planning state for agent `k`.

Given the known order and observed positions of all agents, predict where
all other agents will be after `depth` rounds of L0 play, then project
walls at those predicted positions.

Each predicted round simulates all agents (except k) taking one L0 step
in the known order. Each agent in the prediction treats others as walls
at their current predicted positions and picks an action via Boltzmann policy.

Agent k does NOT move in the prediction — it stays at its current position
and imagines where others will go.

- depth=0: not called (handled directly as wall projection)
- depth=1: one round of L0 prediction for all others
- depth=2+: deeper forward prediction
"""
function predict_others_forward(state::State,
                                k::Int,
                                domain::Domain,
                                policies::AbstractVector{<:BoltzmannPolicy},
                                order::AbstractVector{Int},
                                observed_positions::Vector{Tuple{Int,Int}},
                                depth::Int)
    predicted_positions = copy(observed_positions)

    for _round in 1:depth
        for n in order
            if n == k
                continue
            end
            # Build L0 view for agent n at its predicted position,
            # with others as walls at their predicted positions.
            l0_view = build_single_agent_view(state, n, domain,
                                              predicted_positions[n],
                                              predicted_positions)
            act_n = SymbolicPlanners.get_action(policies[n], l0_view)
            # Simulate: apply action in the L0 view to get new position
            new_l0_state = safe_transition(domain, l0_view, act_n)
            # Extract agent n's new position from the single-agent state
            predicted_positions[n] = (new_l0_state[XLOC[n]], new_l0_state[YLOC[n]])
        end
    end

    # Project walls at predicted-future positions, k at current position
    return project_others_to_walls_at(state, k, domain, predicted_positions)
end


"""
Build a single-agent PDDL state for agent `n`:
- agent n is placed at `agent_pos`
- all other agents are walls at positions given in `all_positions`
- non-agent objects (wells, tanks, finishes) are taken from `ref_state`
"""
function build_single_agent_view(ref_state::State, n::Int, domain::Domain,
                                  agent_pos::Tuple{Int,Int},
                                  all_positions::Vector{Tuple{Int,Int}})
    objtypes = PDDL.get_objtypes(ref_state)

    # Delete all agents except n, collect wall positions
    wall_positions = Vector{Tuple{Int,Int}}()
    for m in 1:N_AGENTS
        if m == n
            continue
        end
        push!(wall_positions, all_positions[m])
        delete!(objtypes, AGENTS[m])
    end

    walls = copy(ref_state[pddl"(walls)"])
    for (x, y) in wall_positions
        walls[y, x] = true
    end

    fluents = Dict{Term, Any}()
    for (obj, objtype) in objtypes
        if objtype == :agent
            continue
        end
        fluents[Compound(:xloc, Term[obj])] = ref_state[Compound(:xloc, Term[obj])]
        fluents[Compound(:yloc, Term[obj])] = ref_state[Compound(:yloc, Term[obj])]
    end

    fluents[pddl"(walls)"] = walls
    fluents[XLOC[n]] = agent_pos[1]
    fluents[YLOC[n]] = agent_pos[2]
    fluents[HAS_FILLED[n]] = ref_state[HAS_FILLED[n]]
    fluents[HAS_WATER1[n]] = ref_state[HAS_WATER1[n]]
    fluents[HAS_WATER2[n]] = ref_state[HAS_WATER2[n]]
    fluents[HAS_WATER3[n]] = ref_state[HAS_WATER3[n]]

    return initstate(domain, objtypes, fluents)
end


"""
Build single-agent view for focal agent `k` (level-0 projection), where other agents become walls.
"""
function project_others_to_walls(state::State, k::Int, domain::Domain)
    # define objects and their types
    objtypes = PDDL.get_objtypes(state)

    # store the coordinates of all the agents that get deleted, to place walls there later
    agent_locs = Array{Tuple{Int,Int}}(undef, N_AGENTS - 1)
    agent_index = 1

    # iterate over all agent objects. Unless it is the current agent, turn it into a wall
    for n in 1:N_AGENTS
        if n == k
            continue
        end
        agent = AGENTS[n]
        # save the coordinates of the agent, in order to place a wall there
        agent_locs[agent_index] = (state[XLOC[n]], state[YLOC[n]])
        agent_index += 1
        # then, delete the agent completely
        delete!(objtypes, agent)
    end

    # get the old wall matrix and modify it to include walls where agents were
    walls = copy(state[pddl"(walls)"])
    for (x, y) in agent_locs
        walls[y, x] = true
    end

    fluents = Dict{Term, Any}()

    # Loop over non-agents and assign their locations
    for (obj, objtype) in objtypes
        if objtype == :agent
            continue
        end
        fluents[Compound(:xloc, Term[obj])] = state[Compound(:xloc, Term[obj])]
        fluents[Compound(:yloc, Term[obj])] = state[Compound(:yloc, Term[obj])]
    end

    # Assign walls fluent
    fluents[pddl"(walls)"] = walls
    # Assign position of remaining agent
    fluents[XLOC[k]] = state[XLOC[k]]
    fluents[YLOC[k]] = state[YLOC[k]]
    
    # Task-status fluents for the remaining agent
    fluents[HAS_FILLED[k]] = state[HAS_FILLED[k]]
    fluents[HAS_WATER1[k]] = state[HAS_WATER1[k]]
    fluents[HAS_WATER2[k]] = state[HAS_WATER2[k]]
    fluents[HAS_WATER3[k]] = state[HAS_WATER3[k]]
    
    new_state = initstate(domain, objtypes, fluents)
    return new_state
end

"""
Run one simulation timestep.

Agents move in a fixed order (drawn once per run). Each agent observes:
- Agents who already moved this timestep: their current (just-moved) position
- Agents who haven't moved yet: their position after last timestep's move

With REASONING_DEPTH=0, agents project others to walls at observed positions.
With REASONING_DEPTH>=1, agents predict D rounds of L0 play (in known order)
from observed positions, then project walls at predicted positions.

Returns (new_state, updated_last_positions).
"""
function simulation_step(state::State,
                         policies::AbstractVector{<:BoltzmannPolicy},
                         domain::Domain,
                         order::AbstractVector{Int},
                         last_positions::Vector{Tuple{Int,Int}})

    interim_state = state

    for idx in 1:N_AGENTS
        k = order[idx]

        # --- Build observed positions for agent k ---
        # Agents before k in the order: already moved this timestep → current position
        # Agents after k (and k itself): last known position from previous timestep
        observed = copy(last_positions)
        for j in 1:(idx - 1)
            m = order[j]
            observed[m] = (interim_state[XLOC[m]], interim_state[YLOC[m]])
        end

        # --- Build planning state ---
        if REASONING_DEPTH == 0
            # L0: project others to walls at observed positions
            planning_state = project_others_to_walls_at(interim_state, k, domain, observed)
        else
            # Forward-looking: predict D rounds of L0 for all others
            planning_state = predict_others_forward(interim_state, k, domain,
                                                     policies, order, observed,
                                                     REASONING_DEPTH)
        end

        # --- Choose and apply action ---
        act = SymbolicPlanners.get_action(policies[k], planning_state)
        interim_state = safe_transition(domain, interim_state, act)
    end

    # Update last_positions to where everyone ended up this timestep
    new_last_positions = get_all_positions(interim_state)

    return interim_state, new_last_positions
end



# --------------------------------------------------------------------
# Main simulation loop
# --------------------------------------------------------------------

function run_simulations()
    # Make sure output directories exist
    mkpath(OUTPUT_DIR)
    if SAVE_TRAJECTORIES
        mkpath(TRAJECTORY_DIR)
    end

    println("Starting simulations:")
    println("  order = $(ORDER)")
    println("  reasoning depth = $(REASONING_DEPTH)")
    println("  planning = $(PLANNING)")
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
        domain = PlanningDomains.load_domain(DOMAIN_FILE)
        steps_to_go = build_steps_to_go_estimator()
        policies = build_agent_policies(domain, steps_to_go)

        # Load problem / initial state for this map
        problem = PlanningDomains.load_problem(joinpath(MAPS_DIR, map))
        initial_state = initstate(domain, problem)

        # Results container (one row per run) for this map
        results = Vector{NamedTuple}(undef, RUNS)

        for run in 1:RUNS
            if VERBOSE
                tprintln("[$(Threads.threadid())] map=$(map) run=$(run)")
            end

            # Seed per (map_index, run) so results are reproducible independent of thread scheduling.
            seed_this_run = BASE_SEED + MAP_SEED_OFFSET * map_index + run
            Random.seed!(seed_this_run)

            # Draw order once per run — agents observe this order
            run_order = draw_run_order()

            # Reset current state
            state = initial_state
            # Initialize last_positions to starting positions
            last_positions = get_all_positions(state)
            # Keep previous two states for detecting if an agent is stuck
            state_t_minus_1 = state
            state_t_minus_2 = state

            # Record timestep when agent n first fills a tank (0 if not yet, -1 if stuck)
            agent_filled = fill(0, N_AGENTS)

            traj_name = "trajectory_$(map_name)_run$(run).log"
            traj_path = joinpath(TRAJECTORY_DIR, traj_name)
            
            elapsed_run = @elapsed begin
                if SAVE_TRAJECTORIES
                    open(traj_path, "w") do io
                    @printf(io, "# map=%s run=%d seed=%d temp=%.6g order=%s\n",
                            map, run, seed_this_run, TEMPERATURE, string(run_order))
                    println(io, "# t=0")
                    show(io, state); println(io)
            
                    for t in 1:TIME_MAX
                        state, last_positions = simulation_step(state, policies, domain,
                                                                 run_order, last_positions)

                        println(io, "# t=$t")
                        show(io, state); println(io)
                
                        for n in 1:N_AGENTS
                            if agent_filled[n] == 0 && state[HAS_FILLED[n]]
                                agent_filled[n] = t
                            end
                        end

                        if t>=3 && state == state_t_minus_1 && state_t_minus_1 == state_t_minus_2
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
                        state, last_positions = simulation_step(state, policies, domain,
                                                                 run_order, last_positions)

                        for n in 1:N_AGENTS
                            if agent_filled[n] == 0 && state[HAS_FILLED[n]]
                                agent_filled[n] = t
                            end
                        end

                        if t>=3 && state == state_t_minus_1 && state_t_minus_1 == state_t_minus_2
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

            # mark unfinished agents as -1
            for n in 1:N_AGENTS
                if agent_filled[n] == 0
                    agent_filled[n] = -1
                end
            end
            
            results[run] = (
                run = run, 
                agent_1 = agent_filled[1],
                agent_2 = agent_filled[2],
                agent_3 = agent_filled[3],
                agent_4 = agent_filled[4],
                agent_5 = agent_filled[5],
                agent_6 = agent_filled[6],
                agent_7 = agent_filled[7],
                agent_8 = agent_filled[8],
                map = map_name,
                seed = seed_this_run,
                temperature = TEMPERATURE,
                time_max = TIME_MAX,
                n_agents = N_AGENTS,
                run_elapsed_seconds = elapsed_run,
                order = ORDER,
                run_order = string(run_order),
                reasoning_depth = REASONING_DEPTH,
                planning = PLANNING,
                planning_h_mult = PLANNING_H_MULT,
                planning_search_noise = PLANNING_SEARCH_NOISE,            
            )    
        end

        # Save per-(map, runs) results.
        csv_name = joinpath(OUTPUT_DIR, replace(map, ".pddl" => "") * ".csv")
        CSV.write(csv_name, results)

        println("Finished map $map_name → wrote $(basename(csv_name))")
    end
end

# Run if called as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulations()
end