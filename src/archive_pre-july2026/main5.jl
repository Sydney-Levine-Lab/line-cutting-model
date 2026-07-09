" April 9: play the game forward rather than enumerate options (v3) "
# Entry point for running multi-agent water-collection simulations.
#
# Key modeling assumptions:
# - Agents act sequentially in a fixed random order, drawn once per run.
# - At depth 0, all agents plan from the beginning-of-timestep snapshot
#   (others become walls at their start-of-round positions). Boltzmann policy.
# - At depth 1, agents predict one L0 step for agents before them in
#   the order (depth-1 heuristic). Boltzmann policy.
# - At depth >= 2, agents do "real L1" planning: for each candidate
#   action available from the depth-1 view, simulate d-1 full rounds of
#   L0 play forward (all agents including k acting as L0 reasoners),
#   and pick the first action that leads to the best outcome for k
#   after those rounds. This is a true forward simulation of the game,
#   not just a static wall prediction.
# - A* search with a task-specific heuristic for steps-to-go estimation.
# - Boltzmann action selection at depth 0-1; greedy best-action at depth >= 2.

using PDDL, PlanningDomains, SymbolicPlanners
using Random, Dates, Printf
using CSV

include(joinpath(@__DIR__, "water_collection_heuristic.jl"))
include(joinpath(@__DIR__, "utils.jl"))

PDDL.Arrays.@register()

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

const N_AGENTS    = 8
const RUNS        = env_int("RUNS", 5)
const TIME_MAX    = 1000
const TEMPERATURE = 0.0001
const DEPTH       = env_int("DEPTH", 0)

const RUN_LABEL     = get(ENV, "RUN_LABEL", "user_run")
const USE_TIMESTAMP = true
const RUN_ID = USE_TIMESTAMP ?
    RUN_LABEL * "_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS") :
    RUN_LABEL

const BASE_SEED       = 1234
const MAP_SEED_OFFSET = 10_000

const FALLBACK_COUNT = Threads.Atomic{Int}(0)
const TOTAL_D2_DECISIONS = Threads.Atomic{Int}(0)

const SRC_DIR     = @__DIR__
const DOMAIN_FILE = joinpath(SRC_DIR, "domain.pddl")
const MAPS_DIR    = joinpath(SRC_DIR, "maps")

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

function build_steps_to_go_estimator()
    heuristic         = WaterCollectionHeuristic()
    planner           = SymbolicPlanners.AStarPlanner(heuristic)
    planner_heuristic = SymbolicPlanners.PlannerHeuristic(planner)
    return SymbolicPlanners.memoized(planner_heuristic)
end

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

function safe_transition(domain::Domain, state::State, act)
    try
        return transition(domain, state, act)
    catch
        return state
    end
end

# --------------------------------------------------------------------
# L0 prediction step
# --------------------------------------------------------------------

"""
Predict one L0 step for agent `agent_idx`:
project others to walls, pick an action, apply it.
"""
function predict_l0_step(predicted_state::State, agent_idx::Int,
                          domain::Domain,
                          policies::AbstractVector{<:BoltzmannPolicy})
    l0_view = project_others_to_walls(predicted_state, agent_idx, domain)
    act = SymbolicPlanners.get_action(policies[agent_idx], l0_view)
    return safe_transition(domain, predicted_state, act)
end

# --------------------------------------------------------------------
# Depth 1: predict agents before k
# --------------------------------------------------------------------

"""
Build the planning state for agent `k` at reasoning depth 1.
Predict one L0 step for each agent before k in the order.
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
# Depth >= 2: "real L1" — simulate L0 play forward, evaluate each
#             candidate first action by outcome
# --------------------------------------------------------------------

"""
    simulate_l0_round(state, domain, policies, order)

Simulate one full round of L0 play: each agent in `order` takes
one L0 step (project others to walls, pick action, apply).
Returns the state after all agents have moved.
"""
function simulate_l0_round(state::State, domain::Domain,
                            policies::AbstractVector{<:BoltzmannPolicy},
                            order::Vector{Int})
    for idx in 1:N_AGENTS
        n = order[idx]
        state = predict_l0_step(state, n, domain, policies)
    end
    return state
end

"""
    evaluate_action(initial_state, k, first_action, domain, policies,
                     order, idx_in_order, lookahead_rounds, steps_to_go)

Evaluate a candidate first action for agent k by simulating the game
forward.

1. Start from `initial_state` (beginning of timestep).
2. Predict L0 steps for agents before k (they already moved).
3. Apply `first_action` for k.
4. Predict L0 steps for agents after k (completing the round).
5. Simulate `lookahead_rounds - 1` more full rounds of L0 play
   (all agents including k acting as L0 reasoners).
6. Evaluate k's position at the end: how many steps to goal?

Returns a value (higher = better for k).
"""
function evaluate_action(initial_state::State, k::Int,
                          first_action,
                          domain::Domain,
                          policies::AbstractVector{<:BoltzmannPolicy},
                          order::Vector{Int}, idx_in_order::Int,
                          lookahead_rounds::Int,
                          steps_to_go)
    sim_state = initial_state

    # Agents before k take their L0 step
    for j_idx in 1:(idx_in_order - 1)
        j = order[j_idx]
        sim_state = predict_l0_step(sim_state, j, domain, policies)
    end

    # Apply k's candidate first action
    sim_state = safe_transition(domain, sim_state, first_action)

    # Agents after k take their L0 step (completing round 1)
    for j_idx in (idx_in_order + 1):N_AGENTS
        j = order[j_idx]
        sim_state = predict_l0_step(sim_state, j, domain, policies)
    end

    # Simulate additional full rounds of L0 play (all agents including k)
    for _round in 2:lookahead_rounds
        sim_state = simulate_l0_round(sim_state, domain, policies, order)
    end

    # Evaluate: how far is k from their goal in this simulated future?
    # Use the single-agent view so A* can plan k's remaining path
    k_view = project_others_to_walls(sim_state, k, domain)
    goal = SymbolicPlanners.MinStepsGoal(Term[HAS_FILLED[k]])

    # If k already filled in the simulation, that's the best outcome
    if sim_state[HAS_FILLED[k]]
        return 0.0
    end

    # Use the memoized steps_to_go heuristic for evaluation
    value_policy = SymbolicPlanners.FunctionalVPolicy(steps_to_go, domain, goal)
    return SymbolicPlanners.get_value(value_policy, k_view)
end

"""
    choose_real_l1_action(initial_state, k, domain, policies, steps_to_go,
                           order, idx_in_order, depth)

Choose an action for agent k using "real L1" forward simulation.

Get candidate actions from the depth-1 view (accurate for right now,
avoids immediate collisions). For each candidate, simulate (depth-1)
rounds of L0 play forward and evaluate k's outcome. Pick the action
with the best outcome.

Falls back to depth-1 Boltzmann policy if all candidates tie.
"""
function choose_real_l1_action(initial_state::State, k::Int,
                                domain::Domain,
                                policies::AbstractVector{<:BoltzmannPolicy},
                                steps_to_go,
                                order::Vector{Int},
                                idx_in_order::Int,
                                depth::Int)
    # Build depth-1 state to get candidate actions (collision avoidance)
    depth1_predicted = initial_state
    for j_idx in 1:(idx_in_order - 1)
        j = order[j_idx]
        depth1_predicted = predict_l0_step(depth1_predicted, j, domain, policies)
    end
    depth1_view = project_others_to_walls(depth1_predicted, k, domain)

    candidate_actions = collect(PDDL.available(domain, depth1_view))

    if isempty(candidate_actions)
        return SymbolicPlanners.get_action(policies[k], depth1_view)
    end

    # If only one candidate, no need to simulate
    if length(candidate_actions) == 1
        return candidate_actions[1]
    end

    # Evaluate each candidate by forward simulation
    lookahead_rounds = depth - 1  # depth=2 → 1 round of lookahead after the first move
    best_action = nothing
    best_value = -Inf

    for act in candidate_actions
        value = evaluate_action(initial_state, k, act, domain, policies,
                                 order, idx_in_order, lookahead_rounds,
                                 steps_to_go)

        if value > best_value
            best_value = value
            best_action = act
        end
    end

    # Fall back to depth-1 if all evaluations returned -Inf
    if best_value == -Inf
        Threads.atomic_add!(FALLBACK_COUNT, 1)
        Threads.atomic_add!(TOTAL_D2_DECISIONS, 1)
        return SymbolicPlanners.get_action(policies[k], depth1_view)
    end

    Threads.atomic_add!(TOTAL_D2_DECISIONS, 1)
    return best_action
end

# --------------------------------------------------------------------
# Simulation step
# --------------------------------------------------------------------

"""
Run one simulation timestep.

- Depth 0: plan from snapshot, Boltzmann policy.
- Depth 1: predict agents before k, Boltzmann policy.
- Depth >= 2: get valid moves from depth-1 view, evaluate each by
  simulating (depth-1) rounds of L0 play forward, pick the best.
"""
function simulation_step(state::State,
                         policies::AbstractVector{<:BoltzmannPolicy},
                         domain::Domain,
                         order::Vector{Int},
                         steps_to_go)
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
            act = choose_real_l1_action(initial_state, k, domain, policies,
                                         steps_to_go, order, idx, DEPTH)
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

    println("Starting simulations (main2 — real L1 for depth>=2):")
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

    Threads.@threads for map_index in eachindex(MAP_FILES)
        map = MAP_FILES[map_index]
        map_name = replace(map, ".pddl" => "")

        if VERBOSE
            tprintln("Starting map $map_name (thread $(Threads.threadid()))")
        end

        domain      = PlanningDomains.load_domain(DOMAIN_FILE)
        steps_to_go = build_steps_to_go_estimator()
        policies    = build_agent_policies(domain, steps_to_go)

        problem       = PlanningDomains.load_problem(joinpath(MAPS_DIR, map))
        initial_state = initstate(domain, problem)

        results = Vector{NamedTuple}(undef, RUNS)

        for run in 1:RUNS
            if VERBOSE
                tprintln("[$(Threads.threadid())] map=$(map) run=$(run)")
            end

            seed_this_run = BASE_SEED + MAP_SEED_OFFSET * map_index + run
            Random.seed!(seed_this_run)

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
                            state = simulation_step(state, policies, domain,
                                                    order, steps_to_go)
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
                                                order, steps_to_go)

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

    if DEPTH >= 2
        total = TOTAL_D2_DECISIONS[]
        fallbacks = FALLBACK_COUNT[]
        pct = total > 0 ? round(100.0 * fallbacks / total, digits=2) : 0.0
        println("\n[depth>=2 stats] fallbacks: $fallbacks / $total decisions ($pct%)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_simulations()
end