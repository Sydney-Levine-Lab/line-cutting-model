# Entry point for running multi-agent water-collection simulations.

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

# How agents are scheduled within a timestep
#   "fixed"  = deterministic order 1..N_AGENTS
#   "random" = random permutation each timestep
const ORDER = get(ENV, "ORDER", "random")
# possible TODO: implement simultaneous moves,
# but that would require a conflict-resolution layer (as agents try to move in same cell)

# What information agents use to plan this timestep
#   "blind" = everyone reasons from the beginning-of-round snapshot
#   "full"  = each agent reasons from the current world (sees earlier moves)
const INFO = get(ENV, "INFO", "blind")

# CHANGED March 4: continuous variants btw blinds and full; can ultimately replace blind & full
# (also added a "partial" INFO knob value to be removed along w/ the entire INFO knob)
const INFO_PROB = env_float("INFO_PROB", 0.5) # probability of seeing where others go
# DEBUG: track info draws
const INFO_DRAWS = Threads.Atomic{Int}(0)
const INFO_HITS  = Threads.Atomic{Int}(0)

# How sophisticated agents' model of others is (depth of forward prediction).
#   0 = L0: others are static walls at current positions (no prediction)
#   1 = predict one round of L0 play for all others, project walls at predicted
#       positions, then plan from agent's current position
#   2+ = predict D rounds of L0 play, project walls at final predicted positions,
#        plan from current position. Higher depth = more forward-looking.
#
# At depth >= 1, each predicted round simulates all other agents taking one L0
# step (treating others as walls, picking action via Boltzmann policy).
# The focal agent then plans through a world where walls are at others'
# predicted-future positions, but from its own current position.
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
Get the within-round order of agents for this timestep.
"""
function get_order()
    if ORDER == "fixed"
        return collect(1:N_AGENTS)
    elseif ORDER == "random"
        return randperm(N_AGENTS)
    else
        error("Unknown ORDER: $ORDER")
    end
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

Predicts where all *other* agents will be after `depth` rounds of L0 play,
then projects those predicted positions as walls and returns a single-agent
planning state for `k` at its *current* position.

Each predicted round: all agents except `k` take one L0 step (each treating
the rest as walls, picking an action via their Boltzmann policy). Rounds use
fresh random orders. Agent `k` does NOT move in the prediction — it's just
imagining where others will go.

- depth=0: equivalent to basic L0 (project others to walls at current positions).
- depth=1: predict one round of L0 for all others, project walls at predicted
           positions. Similar to the old L1 heuristic but predicts all agents,
           not just those preceding k in the round order.
- depth=2+: deeper forward prediction.
"""
function predict_others_forward(state::State,
                                k::Int,
                                domain::Domain,
                                policies::AbstractVector{<:BoltzmannPolicy},
                                depth::Int)
    predicted_state = state

    for _round in 1:depth
        # All agents except k take one L0 step
        round_order = randperm(N_AGENTS)
        for n in round_order
            if n == k
                continue  # focal agent doesn't move in the prediction
            end
            l0_view = project_others_to_walls(predicted_state, n, domain)
            act_n = SymbolicPlanners.get_action(policies[n], l0_view)
            predicted_state = safe_transition(domain, predicted_state, act_n)
        end
    end

    # Project others to walls at their predicted-future positions,
    # but k is still at its current position (unchanged above).
    return project_others_to_walls(predicted_state, k, domain)
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
Run one simulation timestep: each agent takes one action.

Knobs:

- ORDER:
    "fixed"  → order = 1..N_AGENTS
    "random" → order = random permutation each timestep

- INFO:
    "blind"   → agents reason from the beginning-of-round snapshot
    "full"    → agents reason from the current world (see earlier moves)
    "partial" → each agent independently draws: with probability INFO_PROB
                sees the current world, otherwise uses the snapshot

- REASONING_DEPTH (meaningful when INFO ∈ {"blind", "partial"} and agent
  does not receive full info):
    0  → L0: others are static walls at snapshot positions
    1+ → predict D rounds of L0 play for all other agents, project walls
         at their predicted-future positions, plan from current position
"""
function simulation_step(state::State,
                         policies::AbstractVector{<:BoltzmannPolicy},
                         domain::Domain)

    initial_state = state          # snapshot at start of this timestep
    interim_state = initial_state  # actual evolving world

    order = get_order()

    for idx in 1:N_AGENTS
        k = order[idx]  # focal agent index

        # -----------------------------
        # Build the planning state
        # -----------------------------
        planning_state = nothing

        if INFO == "full"
            # Agents see earlier moves and plan from the current world.
            # REASONING_DEPTH is irrelevant in this mode (already has perfect info).
            planning_state = project_others_to_walls(interim_state, k, domain)

        elseif INFO == "blind"
            if REASONING_DEPTH == 0
                # L0: project others to walls at start-of-round positions.
                planning_state = project_others_to_walls(initial_state, k, domain)
            else
                # Forward-looking: predict D rounds of L0 for all others,
                # project walls at predicted positions, plan from current position.
                planning_state = predict_others_forward(initial_state, k, domain,
                                                        policies, REASONING_DEPTH)
            end

        # Interpolation between full and blind (KC request)
        elseif INFO == "partial"
            Threads.atomic_add!(INFO_DRAWS, 1) # DEBUG
            if rand() < INFO_PROB
                Threads.atomic_add!(INFO_HITS, 1) # DEBUG
                # Got full info this round
                planning_state = project_others_to_walls(interim_state, k, domain)
            else
                # Blind this round — apply reasoning depth
                if REASONING_DEPTH == 0
                    planning_state = project_others_to_walls(initial_state, k, domain)
                else
                    planning_state = predict_others_forward(initial_state, k, domain,
                                                            policies, REASONING_DEPTH)
                end
            end

        else
            error("Unknown INFO: $INFO")
        end

        # -----------------------------
        # Choose and apply action
        # -----------------------------
        act = SymbolicPlanners.get_action(policies[k], planning_state)

        if INFO == "full"
            interim_state = safe_transition(domain, interim_state, act)
        else
            # In blind/partial modes, invalid moves are expected (collisions, etc.).
            # Interpret them as "bump and stay put".
            interim_state = safe_transition(domain, interim_state, act)
        end
    end

    return interim_state
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
    println("  information = $(INFO)")
    println("  information probability = $(INFO_PROB)")
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

            # Reset current state
            state = initial_state
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
                    # write header + t=0 in trajectory file
                    @printf(io, "# map=%s run=%d seed=%d temp=%.6g\n", map, run, seed_this_run, TEMPERATURE)
                    println(io, "# t=0")
                    show(io, state); println(io)
            
                    for t in 1:TIME_MAX
                        state = simulation_step(state, policies, domain)

                        println(io, "# t=$t")
                        show(io, state); println(io)
                
                        # Record first time any agent fills tank
                        for n in 1:N_AGENTS
                            if agent_filled[n] == 0 && state[HAS_FILLED[n]]
                                agent_filled[n] = t
                            end
                        end

                        # Break if stuck for 3 consecutive steps
                        if t>=3 && state == state_t_minus_1 && state_t_minus_1 == state_t_minus_2
                            fill!(agent_filled, -1)
                            break
                        end
                
                        # Break if all agents filled tank
                        if all(>(0), agent_filled)
                            break
                        end

                        # Shift states
                        state_t_minus_2 = state_t_minus_1
                        state_t_minus_1 = state
                    end
                end
                else
                    # Run the simulation without saving trajectory log
                    for t in 1:TIME_MAX
                        state = simulation_step(state, policies, domain)

                        # Record first time any agent fills tank
                        for n in 1:N_AGENTS
                            if agent_filled[n] == 0 && state[HAS_FILLED[n]]
                                agent_filled[n] = t
                            end
                        end

                        # Break if stuck for 3 consecutive steps
                        if t>=3 && state == state_t_minus_1 && state_t_minus_1 == state_t_minus_2
                            fill!(agent_filled, -1)
                            break
                        end
                
                        # Break if all agents filled tank
                        if all(>(0), agent_filled)
                            break
                        end

                        # Shift states
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
                info = INFO,
                info_prob = INFO_PROB,
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
    if INFO == "partial"
        total = INFO_DRAWS[]
        hits  = INFO_HITS[]
        pct   = total > 0 ? round(100.0 * hits / total, digits=2) : 0.0
        println("\n[DEBUG] Info draws: $hits / $total = $pct% (expected: $(100*INFO_PROB)%)")
    end    
end

# Run if called as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulations()
end