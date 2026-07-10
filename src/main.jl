# ====================================================================
# main.jl — CANONICAL simulation script (July 2026 consolidation)
#
# Replaces: main_last.jl / main_last_debug.jl (May 6) and all earlier
# main{2..10}*.jl variants. Single script, two model knobs.
#
# Changes vs. the May 2026 runs (main_last*):
#   1. FIXED play order (agent1, agent2, ..., agent8). Previously
#      randomized per run (randperm). Set ORDER env to override.
#   2. Agents are REMOVED from the world once they fill (has-filled):
#      they no longer block anyone's movement, planning views, or
#      forward simulations. Previously they sat at their final position
#      as permanent obstacles (ghost agents).
#   3. RUN_OFFSET env for split cluster runs (compatible with
#      merge_split_runs).
#   4. Dropped for good: INFO/INFO_PROB interpolation knob, random
#      order, dead FunctionalVPolicy/HeuristicVPolicy code path in
#      evaluate_action.
#
# Two independent reasoning knobs (env vars LEVEL and DEPTH):
#
# LEVEL (how sophisticated is k's model of others):
#   0 = others are static walls (L0 model of others)
#   1 = others are L0 reasoners (predict one L0 step for agents
#       before them)
#   2 = others are L1 reasoners (each predicted agent itself predicts
#       agents before it using L0), etc. (recursive)
#
# DEPTH (how many rounds of forward simulation):
#   0 = no prediction at all: others are walls at their start-of-
#       timestep positions, Boltzmann policy
#   1 = predict one step (at LEVEL) for agents before k in the play
#       order, project to walls, Boltzmann policy
#   >=2 = get candidate actions from the depth-1 view; for each,
#         simulate (DEPTH-1) full rounds of play (all agents acting
#         at LEVEL), pick the action with the best outcome
#
# Named models:
#   LEVEL=0 DEPTH=0 → L0 (blind baseline)
#   LEVEL=0 DEPTH=1 → L1 heuristic  (the winner)
#   LEVEL=0 DEPTH=2 → real L1       (forward simulation)
#   LEVEL=1 DEPTH=1 → L2 heuristic
#   LEVEL=1 DEPTH=2 → real L2
#
# Usage examples:
#   LEVEL=0 DEPTH=1 RUNS=5 RUN_LABEL=L1_heuristic julia -t 8 main.jl
#   LEVEL=0 DEPTH=2 RUNS=1 RUN_OFFSET=2 RUN_LABEL=real_L1_fill julia main.jl
#   MAPS="yes_line_8,maybe_6" TRAJECTORY_LEVEL=summary julia main.jl
# ====================================================================

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
const RUN_OFFSET  = env_int("RUN_OFFSET", 0)   # for split cluster runs
const TIME_MAX    = 1000
const TEMPERATURE = 0.0001
const DEPTH       = env_int("DEPTH", 0)  # forward simulation rounds
const LEVEL       = env_int("LEVEL", 0)  # nested reasoning depth

# Play order: FIXED identity by default (agent1 plays first, then
# agent2, ...). Override with e.g. ORDER="8,7,6,5,4,3,2,1".
const PLAY_ORDER = let
    raw = strip(get(ENV, "ORDER", ""))
    if isempty(raw)
        collect(1:N_AGENTS)
    else
        parsed = [parse(Int, strip(s)) for s in split(raw, ',')]
        sort(parsed) == collect(1:N_AGENTS) ||
            error("ORDER must be a permutation of 1:$(N_AGENTS), got $(parsed)")
        parsed
    end
end

const RUN_LABEL = get(ENV, "RUN_LABEL", "user_run")
# USE_TIMESTAMP=false gives a stable, reproducible output path — required
# for idempotent cluster array jobs (a task can check whether its own CSV
# already exists and skip). Default true for interactive use, where you
# usually don't want to clobber the previous run.
const USE_TIMESTAMP = env_bool("USE_TIMESTAMP", true)
const RUN_ID = USE_TIMESTAMP ?
    RUN_LABEL * "_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS") :
    RUN_LABEL

const BASE_SEED       = 1234
const MAP_SEED_OFFSET = 10_000

const FALLBACK_COUNT     = Threads.Atomic{Int}(0)
const TOTAL_D2_DECISIONS = Threads.Atomic{Int}(0)

const SRC_DIR     = @__DIR__
const DOMAIN_FILE = joinpath(SRC_DIR, "domain.pddl")
const MAPS_DIR    = joinpath(SRC_DIR, "maps")

# CANONICAL_MAP_FILES defines the authoritative map ordering. A map's
# position here (1-based) is its CANONICAL INDEX, which is what seeds are
# derived from. This must never be reordered: doing so silently changes
# every seed and makes past runs irreproducible. Append new maps at the end.
const CANONICAL_MAP_FILES = [
    "no_line_1.pddl", "no_line_2.pddl", "no_line_3.pddl",
    "yes_line_7.pddl", "yes_line_8.pddl", "yes_line_9.pddl", "yes_line_10.pddl",
    "7esque.pddl", "9esque.pddl", "10esque.pddl",
    "maybe_4.pddl", "maybe_5.pddl", "maybe_6.pddl",
    "new_maybe_1.pddl", "new_maybe_2.pddl", "new_maybe_3.pddl",
    "new_maybe_4.pddl", "new_maybe_5.pddl", "new_maybe_6.pddl",
    "no_line_A.pddl", "no_line_B.pddl", "no_line_C.pddl", "no_line_D.pddl",
    "yes_line_B.pddl", "yes_line_C.pddl", "yes_line_D.pddl", "yes_line_E.pddl", "yes_line_F.pddl",
]

const CANONICAL_INDEX = Dict(m => i for (i, m) in enumerate(CANONICAL_MAP_FILES))

"""
Seed for a given map and run. Keyed on the map's CANONICAL index, so
`MAPS="yes_line_8" RUNS=1 RUN_OFFSET=2` yields exactly the same seed as
run 3 of yes_line_8 inside a full 28-map sweep. This is what makes
split/array cluster jobs reproduce single-job runs.
"""
seed_for(map_file::AbstractString, run_global::Int) =
    BASE_SEED + MAP_SEED_OFFSET * CANONICAL_INDEX[map_file] + run_global

# MAPS env var: comma-separated list of map names, with or without .pddl.
# Example: MAPS="yes_line_E,new_maybe_6" julia main.jl
function _parse_maps_env()
    raw = get(ENV, "MAPS", "")
    isempty(raw) && return CANONICAL_MAP_FILES
    parts = String[]
    for p in split(raw, ',')
        s = strip(p)
        isempty(s) && continue
        endswith(s, ".pddl") || (s = s * ".pddl")
        haskey(CANONICAL_INDEX, s) ||
            error("Unknown map '$(s)'. Must be one of: $(join(CANONICAL_MAP_FILES, ", "))")
        push!(parts, String(s))
    end
    return isempty(parts) ? CANONICAL_MAP_FILES : parts
end
const MAP_FILES = _parse_maps_env()

const OUTPUT_DIR     = joinpath(SRC_DIR, "..", "data", "simulations", RUN_ID, "raw")
const TRAJECTORY_DIR = joinpath(OUTPUT_DIR, "trajectories")
const SNAPSHOT_DIR   = joinpath(OUTPUT_DIR, "run_info")
const WRITE_SNAPSHOT = env_bool("WRITE_SNAPSHOT", false)
const VERBOSE        = env_bool("VERBOSE", true)
const DEBUG_EVAL     = env_bool("DEBUG_EVAL", false)  # print depth>=2 evals to stdout

# TRAJECTORY_LEVEL controls per-run trajectory log content:
#   "none"    : no trajectory file
#   "summary" : start/end states + one [decision] line per agent per
#               timestep, plus [candidates] line for depth>=2
#   "full"    : full state dump every timestep + decision/candidate lines
const SAVE_TRAJECTORIES = env_bool("SAVE_TRAJECTORIES", false)  # legacy alias
const TRAJECTORY_LEVEL = let
    raw = lowercase(strip(get(ENV, "TRAJECTORY_LEVEL", "")))
    if !isempty(raw)
        raw
    elseif SAVE_TRAJECTORIES
        "summary"
    else
        "none"
    end
end
const SAVE_TRAJ = TRAJECTORY_LEVEL != "none"
const FULL_TRAJ = TRAJECTORY_LEVEL == "full"

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
# Agent policies
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
# World projection, agent removal, transitions
# --------------------------------------------------------------------

"Agent ids (1..N_AGENTS) still present in `state`."
function agents_in_state(state::State)
    objtypes = PDDL.get_objtypes(state)
    return [n for n in 1:N_AGENTS if haskey(objtypes, AGENTS[n])]
end

"""
Build a single-agent planning view for agent `k`:
all OTHER agents still present in `state` become walls at their
positions. Agents that have been removed (finished) are simply absent —
they block nothing.
"""
function project_others_to_walls(state::State, k::Int, domain::Domain)
    objtypes = PDDL.get_objtypes(state)

    agent_locs = Tuple{Int,Int}[]
    for n in agents_in_state(state)
        n == k && continue
        push!(agent_locs, (state[XLOC[n]], state[YLOC[n]]))
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
Rebuild the world state without the agents in `zap` (they have filled
and leave the world: no longer obstacles for movement, planning, or
forward simulation).
"""
function remove_agents(state::State, domain::Domain, zap::Vector{Int})
    isempty(zap) && return state

    objtypes = PDDL.get_objtypes(state)
    for n in zap
        delete!(objtypes, AGENTS[n])
    end

    fluents = Dict{Term, Any}()
    for (obj, objtype) in objtypes
        objtype == :agent && continue
        fluents[Compound(:xloc, Term[obj])] = state[Compound(:xloc, Term[obj])]
        fluents[Compound(:yloc, Term[obj])] = state[Compound(:yloc, Term[obj])]
    end
    fluents[pddl"(walls)"] = copy(state[pddl"(walls)"])

    for n in 1:N_AGENTS
        haskey(objtypes, AGENTS[n]) || continue
        fluents[XLOC[n]]       = state[XLOC[n]]
        fluents[YLOC[n]]       = state[YLOC[n]]
        fluents[HAS_FILLED[n]] = state[HAS_FILLED[n]]
        fluents[HAS_WATER1[n]] = state[HAS_WATER1[n]]
        fluents[HAS_WATER2[n]] = state[HAS_WATER2[n]]
        fluents[HAS_WATER3[n]] = state[HAS_WATER3[n]]
    end

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
# Prediction steps at different levels of reasoning
# --------------------------------------------------------------------

"""
    predict_step(state, agent_idx, domain, policies, order, idx_in_order, level)

Predict one step for `agent_idx` at the given reasoning level.

- level 0: L0 — project others to walls at current positions, pick action.
- level 1: L1 — predict L0 steps for agents before `agent_idx` in `order`,
  then project to walls, pick action.
- level >= 2: recursive.

`order` contains only ACTIVE agents; `idx_in_order` is the index of
`agent_idx` within it. Returns the state after the predicted action.
"""
function predict_step(state::State, agent_idx::Int,
                       domain::Domain,
                       policies::AbstractVector{<:BoltzmannPolicy},
                       order::Vector{Int}, idx_in_order::Int,
                       level::Int)
    if level == 0
        l0_view = project_others_to_walls(state, agent_idx, domain)
        act = SymbolicPlanners.get_action(policies[agent_idx], l0_view)
        return safe_transition(domain, state, act)
    end

    predicted_state = state
    for j_idx in 1:(idx_in_order - 1)
        j = order[j_idx]
        predicted_state = predict_step(predicted_state, j, domain, policies,
                                        order, j_idx, level - 1)
    end

    planning_view = project_others_to_walls(predicted_state, agent_idx, domain)
    act = SymbolicPlanners.get_action(policies[agent_idx], planning_view)
    return safe_transition(domain, state, act)
end

# --------------------------------------------------------------------
# Build planning state (depth 1)
# --------------------------------------------------------------------

"""
    build_planning_state_d1(initial_state, k, domain, policies, order,
                             idx_in_order, level)

Predict one step (at LEVEL) for each ACTIVE agent before k in the play
order, then project everyone else to walls.
"""
function build_planning_state_d1(initial_state::State, k::Int,
                                  domain::Domain,
                                  policies::AbstractVector{<:BoltzmannPolicy},
                                  order::Vector{Int}, idx_in_order::Int,
                                  level::Int)
    predicted_state = initial_state
    for j_idx in 1:(idx_in_order - 1)
        j = order[j_idx]
        predicted_state = predict_step(predicted_state, j, domain, policies,
                                        order, j_idx, level)
    end
    return project_others_to_walls(predicted_state, k, domain)
end

# --------------------------------------------------------------------
# Depth >= 2: forward simulation with agents at the given LEVEL
# --------------------------------------------------------------------

"""
    simulate_round(state, domain, policies, order, level)

Simulate one full round: each ACTIVE agent in `order` takes one step at
the given reasoning LEVEL.
"""
function simulate_round(state::State, domain::Domain,
                         policies::AbstractVector{<:BoltzmannPolicy},
                         order::Vector{Int}, level::Int)
    for idx in eachindex(order)
        n = order[idx]
        state = predict_step(state, n, domain, policies, order, idx, level)
    end
    return state
end

"""
    evaluate_action(initial_state, k, first_action, domain, policies,
                     order, idx_in_order, lookahead_rounds, level, steps_to_go)

Evaluate a candidate first action for agent k by simulating the game
forward with all ACTIVE agents acting at the given LEVEL.
"""
function evaluate_action(initial_state::State, k::Int,
                          first_action,
                          domain::Domain,
                          policies::AbstractVector{<:BoltzmannPolicy},
                          order::Vector{Int}, idx_in_order::Int,
                          lookahead_rounds::Int,
                          level::Int,
                          steps_to_go)
    sim_state = initial_state

    # Agents before k take their step
    for j_idx in 1:(idx_in_order - 1)
        j = order[j_idx]
        sim_state = predict_step(sim_state, j, domain, policies,
                                  order, j_idx, level)
    end

    # Apply k's candidate first action
    sim_state = safe_transition(domain, sim_state, first_action)

    # Agents after k take their step (completing round 1)
    for j_idx in (idx_in_order + 1):length(order)
        j = order[j_idx]
        sim_state = predict_step(sim_state, j, domain, policies,
                                  order, j_idx, level)
    end

    # Simulate additional full rounds
    for _round in 2:lookahead_rounds
        sim_state = simulate_round(sim_state, domain, policies, order, level)
    end

    # Evaluate k's position at the end
    if sim_state[HAS_FILLED[k]]
        return 0.0
    end

    k_view = project_others_to_walls(sim_state, k, domain)
    goal = SymbolicPlanners.MinStepsGoal(Term[HAS_FILLED[k]])
    return -SymbolicPlanners.compute(steps_to_go, domain, k_view, goal)
end

"""
    choose_forward_action(initial_state, k, domain, policies, steps_to_go,
                           order, idx_in_order, depth, level; log_io, t)

Choose an action for agent k using forward simulation at the given LEVEL.
Candidates come from the depth-1 view; each is evaluated by simulating
(depth-1) rounds of play. Falls back to the depth-1 policy if every
candidate evaluates to -Inf (no feasible plan from the rolled-out state).
"""
function choose_forward_action(initial_state::State, k::Int,
                                domain::Domain,
                                policies::AbstractVector{<:BoltzmannPolicy},
                                steps_to_go,
                                order::Vector{Int},
                                idx_in_order::Int,
                                depth::Int,
                                level::Int;
                                log_io::Union{IO,Nothing}=nothing,
                                t::Int=0)
    d1_view = build_planning_state_d1(initial_state, k, domain, policies,
                                       order, idx_in_order, level)

    candidate_actions = collect(PDDL.available(domain, d1_view))

    if isempty(candidate_actions)
        chosen = SymbolicPlanners.get_action(policies[k], d1_view)
        _log_decision(log_io, t, idx_in_order, k, initial_state, chosen, "EMPTY")
        return chosen
    end

    if length(candidate_actions) == 1
        chosen = candidate_actions[1]
        _log_decision(log_io, t, idx_in_order, k, initial_state, chosen, "SOLE")
        return chosen
    end

    lookahead_rounds = depth - 1
    best_action = nothing
    best_value = -Inf
    second_best_value = -Inf

    log_pairs = (DEBUG_EVAL || log_io !== nothing) ?
                Vector{Tuple{Any,Float64}}() : nothing

    for act in candidate_actions
        value = evaluate_action(initial_state, k, act, domain, policies,
                                 order, idx_in_order, lookahead_rounds,
                                 level, steps_to_go)
        log_pairs !== nothing && push!(log_pairs, (act, Float64(value)))
        if value > best_value
            second_best_value = best_value
            best_value = value
            best_action = act
        elseif value > second_best_value
            second_best_value = value
        end
    end

    if DEBUG_EVAL
        kpos = (initial_state[XLOC[k]], initial_state[YLOC[k]])
        @printf("[DEBUG d>=2] k=%d (idx=%d in order) at %s — %d candidates, lookahead=%d\n",
                k, idx_in_order, kpos, length(candidate_actions), lookahead_rounds)
        for (a, v) in log_pairs
            marker = (a === best_action) ? "  *" : "   "
            @printf("%s %-50s  value=%.4f\n", marker, string(a), v)
        end
        @printf("        chosen: %s  (best_value=%.4f)\n",
                best_action === nothing ? "FALLBACK" : string(best_action), best_value)
    end

    fallback = best_value == -Inf
    chosen = fallback ? SymbolicPlanners.get_action(policies[k], d1_view) : best_action

    if log_io !== nothing
        tag = fallback ? "FALLBACK" : "OK"
        _log_decision(log_io, t, idx_in_order, k, initial_state, chosen, tag;
                      best_value=best_value, second_best=second_best_value)
        _log_candidates(log_io, t, idx_in_order, k, log_pairs, best_action)
    end

    Threads.atomic_add!(TOTAL_D2_DECISIONS, 1)
    fallback && Threads.atomic_add!(FALLBACK_COUNT, 1)
    return chosen
end

# --------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------

function _log_decision(io::Union{IO,Nothing}, t::Int, idx::Int, k::Int,
                       state::State, action, tag::String;
                       best_value=nothing, second_best=nothing)
    io === nothing && return
    pos = (state[XLOC[k]], state[YLOC[k]])
    if best_value === nothing
        @printf(io, "[decision] t=%d idx=%d k=%d pos=(%d,%d) chose=%s tag=%s\n",
                t, idx, k, pos[1], pos[2], string(action), tag)
    else
        gap = (second_best === nothing || second_best == -Inf) ?
              NaN : (best_value - second_best)
        @printf(io, "[decision] t=%d idx=%d k=%d pos=(%d,%d) chose=%s tag=%s best=%.4f gap=%.4f\n",
                t, idx, k, pos[1], pos[2], string(action), tag, best_value, gap)
    end
end

function _log_candidates(io::Union{IO,Nothing}, t::Int, idx::Int, k::Int,
                         pairs, best_action)
    io === nothing && return
    pairs === nothing && return
    parts = String[]
    for (a, v) in pairs
        push!(parts, @sprintf("%s=%.4f", string(a), v))
    end
    @printf(io, "[candidates] t=%d idx=%d k=%d %s best=%s\n",
            t, idx, k, join(parts, " "),
            best_action === nothing ? "NONE" : string(best_action))
end

# --------------------------------------------------------------------
# Simulation step
# --------------------------------------------------------------------

"""
Run one simulation timestep over the ACTIVE agents in `order`.

- Depth 0: others are walls at start-of-timestep positions.
- Depth 1: predict agents before k at LEVEL, Boltzmann policy.
- Depth >= 2: candidates from depth-1 view, evaluated by forward
  simulation, pick the best.
"""
function simulation_step(state::State,
                         policies::AbstractVector{<:BoltzmannPolicy},
                         domain::Domain,
                         order::Vector{Int},
                         steps_to_go;
                         log_io::Union{IO,Nothing}=nothing,
                         t::Int=0)
    initial_state = state

    for idx in eachindex(order)
        k = order[idx]

        if DEPTH == 0
            planning_state = project_others_to_walls(initial_state, k, domain)
            act = SymbolicPlanners.get_action(policies[k], planning_state)
            _log_decision(log_io, t, idx, k, initial_state, act, "L0")
        elseif DEPTH == 1
            planning_state = build_planning_state_d1(initial_state, k, domain,
                                                      policies, order, idx, LEVEL)
            act = SymbolicPlanners.get_action(policies[k], planning_state)
            _log_decision(log_io, t, idx, k, initial_state, act, "HEUR")
        else
            act = choose_forward_action(initial_state, k, domain, policies,
                                         steps_to_go, order, idx, DEPTH, LEVEL;
                                         log_io=log_io, t=t)
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
    SAVE_TRAJ && mkpath(TRAJECTORY_DIR)

    println("Starting simulations (LEVEL×DEPTH, fixed order, zap-on-fill):")
    println("  level = $(LEVEL)")
    println("  depth = $(DEPTH)")
    println("  play order = $(PLAY_ORDER)")
    println("  maps = $(length(MAP_FILES))")
    println("  runs per map = $(RUNS) (offset $(RUN_OFFSET))")
    println("  threads = $(Threads.nthreads())")
    println("  output dir = $(OUTPUT_DIR)")
    println("  trajectory level = $(TRAJECTORY_LEVEL)")
    println()

    WRITE_SNAPSHOT && write_snapshot!(SNAPSHOT_DIR, SRC_DIR)

    Threads.@threads for map_index in eachindex(MAP_FILES)
        map = MAP_FILES[map_index]
        map_name = replace(map, ".pddl" => "")

        VERBOSE && tprintln("Starting map $map_name (thread $(Threads.threadid()))")

        domain      = PlanningDomains.load_domain(DOMAIN_FILE)
        steps_to_go = build_steps_to_go_estimator()
        policies    = build_agent_policies(domain, steps_to_go)

        problem       = PlanningDomains.load_problem(joinpath(MAPS_DIR, map))
        initial_state = initstate(domain, problem)

        results = Vector{NamedTuple}(undef, RUNS)

        for run in 1:RUNS
            run_global = RUN_OFFSET + run
            VERBOSE && tprintln("[$(Threads.threadid())] map=$(map) run=$(run_global)")

            seed_this_run = seed_for(map, run_global)
            Random.seed!(seed_this_run)

            state           = initial_state
            state_t_minus_1 = state
            state_t_minus_2 = state
            agent_filled    = fill(0, N_AGENTS)
            active          = trues(N_AGENTS)
            last_zap_t      = 0   # deadlock check only compares states with the same agent set

            traj_path = joinpath(TRAJECTORY_DIR,
                                 "trajectory_$(map_name)_run$(run_global).log")

            elapsed_run = @elapsed begin
                io_handle = SAVE_TRAJ ? open(traj_path, "w") : nothing
                try
                    if io_handle !== nothing
                        @printf(io_handle, "# map=%s run=%d seed=%d temp=%.6g level=%d depth=%d order=%s zap_on_fill=true trajectory_level=%s\n",
                                map, run_global, seed_this_run, TEMPERATURE, LEVEL, DEPTH,
                                string(PLAY_ORDER), TRAJECTORY_LEVEL)
                        println(io_handle, "# t=0")
                        show(io_handle, state); println(io_handle)
                    end

                    for t in 1:TIME_MAX
                        round_order = [k for k in PLAY_ORDER if active[k]]
                        isempty(round_order) && break

                        state = simulation_step(state, policies, domain,
                                                round_order, steps_to_go;
                                                log_io=io_handle, t=t)

                        if io_handle !== nothing && FULL_TRAJ
                            println(io_handle, "# t=$t")
                            show(io_handle, state); println(io_handle)
                        end

                        # Detect fills, then ZAP: finished agents leave the world.
                        newly_filled = Int[]
                        for n in 1:N_AGENTS
                            if active[n] && agent_filled[n] == 0 && state[HAS_FILLED[n]]
                                agent_filled[n] = t
                                push!(newly_filled, n)
                            end
                        end
                        if !isempty(newly_filled)
                            state = remove_agents(state, domain, newly_filled)
                            for n in newly_filled
                                active[n] = false
                            end
                            last_zap_t = t
                            if io_handle !== nothing
                                @printf(io_handle, "[zap] t=%d removed=%s\n",
                                        t, string(newly_filled))
                            end
                        end

                        if all(>(0), agent_filled)
                            if io_handle !== nothing && !FULL_TRAJ
                                println(io_handle, "# t=$t (final)")
                                show(io_handle, state); println(io_handle)
                            end
                            break
                        end

                        # Deadlock detection: three identical consecutive states.
                        # Only compare once we're >= 3 steps past the last zap, so
                        # all compared states share the same agent set.
                        if t - last_zap_t >= 3 && state == state_t_minus_1 && state_t_minus_1 == state_t_minus_2
                            for n in 1:N_AGENTS
                                agent_filled[n] == 0 && (agent_filled[n] = -1)
                            end
                            if io_handle !== nothing && !FULL_TRAJ
                                println(io_handle, "# t=$t (stuck)")
                                show(io_handle, state); println(io_handle)
                            end
                            break
                        end

                        state_t_minus_2 = state_t_minus_1
                        state_t_minus_1 = state
                    end
                finally
                    io_handle !== nothing && close(io_handle)
                end
            end

            for n in 1:N_AGENTS
                agent_filled[n] == 0 && (agent_filled[n] = -1)
            end

            results[run] = (
                run                 = run_global,
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
                level               = LEVEL,
                depth               = DEPTH,
                order               = string(PLAY_ORDER),
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
