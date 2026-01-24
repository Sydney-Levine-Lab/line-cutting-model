using PDDL, PlanningDomains
using SymbolicPlanners
using CSV
using Random
using Dates

include(joinpath(@__DIR__, "water_collection_heuristic.jl"))

# Enable array-valued fluents (e.g. wall grids) for this PDDL domain
PDDL.Arrays.@register()

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

const N_AGENTS    = 8        # Must match map files
const TIME_MAX    = 1000     # Upper bound on timesteps for a single run
const TEMPERATURE = 0.0001   # Boltzmann temperature
const RUNS        = 5        # Number of runs per map

const SRC_DIR     = @__DIR__ 
const DOMAIN_FILE = joinpath(SRC_DIR, "domain.pddl") 
const MAPS_DIR    = joinpath(SRC_DIR, "maps")

# The experiment set: names correspond to files under maps/.
const MAP_FILES = [
    "no_line_1_test.pddl", "no_line_2_test.pddl", "no_line_3_test.pddl",
    "yes_line_7_test.pddl", "yes_line_8_test.pddl", "yes_line_9_test.pddl", "yes_line_10_test.pddl",
    "7esque_test.pddl", "9esque_test.pddl", "10esque_test.pddl",
    "maybe_4.pddl", "maybe_5.pddl", "maybe_6.pddl",
    "new_maybe_1.pddl", "new_maybe_2.pddl", "new_maybe_3.pddl",
    "new_maybe_4.pddl", "new_maybe_5.pddl", "new_maybe_6.pddl",
    "no_line_A.pddl", "no_line_B.pddl", "no_line_C.pddl", "no_line_D.pddl",
    "yes_line_B.pddl", "yes_line_C.pddl", "yes_line_D.pddl", "yes_line_E.pddl", "yes_line_F.pddl",
]
# Note: "yes_line_A.pddl" not used in stimuli

# Run is identified by label written at launch; if this label has been used, time stamp is used
const RUN_LABEL = get(ENV, "RUN_LABEL", "main2")
const RUN_ID = isempty(strip(RUN_LABEL)) ?
    Dates.format(now(), "yyyy-mm-dd_HHMMSS") :
    RUN_LABEL * "_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS")

const OUTPUT_DIR  = joinpath(SRC_DIR, "..", "data", "simulations", RUN_ID, "raw")
const FULL_HISTORY_DIR     = joinpath(OUTPUT_DIR, "state_histories")
const SAVE_HISTORIES = true   # Don't save full state histories by default

const SEED = 1234

# --------------------------------------------------------------------
# Agent policies
# --------------------------------------------------------------------
# Key assumptions:
# - state evaluation: A* + task-specific heuristic → steps-to-go estimate
# - policy: Boltzmann sampling from action values

"""
Build the steps-to-go estimator used for action selection.
"""
function build_steps_to_go_estimator()
    inner_heuristic   = WaterCollectionHeuristic()       # lower-bound estimate of steps-to-go (task-specific heuristic)
    planner           = AStarPlanner(inner_heuristic)    # search guided by heuristic
    planner_heuristic = PlannerHeuristic(planner)        # estimate steps-to-go from a state
    return memoized(planner_heuristic)                   # cache for speed
end

"""
Build per-agent Boltzmann policies from a steps-to-go estimator.

Each agent's goal is to reach a state where it has filled a tank (`has-filled(agent\$n)`).
"""
function build_agent_policies(domain::Domain, steps_to_go)
    policies = Vector{BoltzmannPolicy}(undef, N_AGENTS)
    for n in 1:N_AGENTS
        agent         = Const(Symbol("agent$n"))
        goal          = MinStepsGoal(Term[Compound(Symbol("has-filled"), Term[agent])])
        value_policy  = FunctionalVPolicy(steps_to_go, domain, goal)
        policies[n]   = BoltzmannPolicy(value_policy, TEMPERATURE)
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

# Note: the first two assumptions reduce coordination needs,
# so agents rarely collide (even with level-0 reasoning) 

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
        agent = Const(Symbol("agent$n"))
        # save the coordinates of the agent, in order to place a wall there
        agent_locs[agent_index] = (
            state[Compound(:xloc, Term[agent])],
            state[Compound(:yloc, Term[agent])]
        )
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

    agent = Const(Symbol("agent$k"))

    # Assign walls fluent
    fluents[pddl"(walls)"] = walls
    # Assign position of remaining agent
    fluents[Compound(:xloc, Term[agent])] = state[Compound(:xloc, Term[agent])]
    fluents[Compound(:yloc, Term[agent])] = state[Compound(:yloc, Term[agent])]

    # Task-status fluents for the remaining agent
    fluents[Compound(Symbol("has-filled"),  Term[agent])] = state[Compound(Symbol("has-filled"),  Term[agent])]
    fluents[Compound(Symbol("has-water1"), Term[agent])] = state[Compound(Symbol("has-water1"), Term[agent])]
    fluents[Compound(Symbol("has-water2"), Term[agent])] = state[Compound(Symbol("has-water2"), Term[agent])]
    fluents[Compound(Symbol("has-water3"), Term[agent])] = state[Compound(Symbol("has-water3"), Term[agent])]

    new_state = initstate(domain, objtypes, fluents)
    return new_state
end

"""
Run one simulation timestep: each agent takes one action.

Assumptions:
- fixed sequential order (1..N_AGENTS)
- later agents observe earlier moves (state is updated in sequence)
- agents plan treating others as static obstacles (level-0 projection)
"""
function simulation_step(state::State,
                         policies::AbstractVector{<:BoltzmannPolicy},
                         domain::Domain)
    interim_state = state
    for n in 1:N_AGENTS
        level_0_projection  = project_others_to_walls(interim_state, n, domain) # treat others as walls
        act                 = SymbolicPlanners.get_action(policies[n], level_0_projection) # select an action
        interim_state       = transition(domain, interim_state, act) # sequential update: later agents observe earlier moves
    end
    return interim_state
end

# --------------------------------------------------------------------
# Main simulation loop
# --------------------------------------------------------------------

function run_simulations()
    # Make sure output directories exist
    mkpath(OUTPUT_DIR)
    mkpath(FULL_HISTORY_DIR)

    # Load domain
    domain = PlanningDomains.load_domain(DOMAIN_FILE)

    # Build agent policies
    steps_to_go = build_steps_to_go_estimator()
    policies = build_agent_policies(domain, steps_to_go)

    for (map_index, map) in enumerate(MAP_FILES)
        println("Running simulations for ", map)

        # Load problem / initial state for this map
        problem       = PlanningDomains.load_problem(joinpath(MAPS_DIR, map))
        initial_state = initstate(domain, problem)

        # For each map, store per-run completion times
        #data[run] = Dict(agent_id => first_fill_timestep | -1)
        data = Dict{Int, Dict{Int, Int}}()

        output_rows = NamedTuple[] # one row per run

        for run in 1:RUNS
            println(run)

            seed_this_run = SEED + 10_000 * map_index + run
            Random.seed!(seed_this_run)

            # Reset state and history
            state = initial_state
            
            state_history = [state]

            # agent_filled[n] = timestep when agent n first fills a tank (0 if not yet, -1 if stuck)
            agent_filled = Dict{Int, Int}(n => 0 for n in 1:N_AGENTS)

            history_path = nothing
            history_io = nothing

                state_history_name = string(
                    "state_history_", map, "_T", TEMPERATURE,
                    "_runs", RUNS, "_run", run, ".txt",
                )
                history_path = joinpath(FULL_HISTORY_DIR, state_history_name)
                open(history_path, "w") do io
                    println(io, "# t=0")
                    show(io, state); println(io)

                for t in 1:TIME_MAX
                    state = simulation_step(state, policies, domain)

                    if SAVE_HISTORIES
                        open(history_path, "a") do file
                            write(file, string(state))
                        end
                    end

                    push!(state_history, state)

                    # If any agent has filled a tank, record the timestep (first time only)
                    for n in 1:N_AGENTS
                        agent = Const(Symbol("agent$n"))
                        if state[Compound(Symbol("has-filled"), Term[agent])] &&
                        agent_filled[n] == 0
                            agent_filled[n] = t
                        end
                    end

                    # If state stopped changing for 3 timesteps, set all to -1 and stop
                    if t >= 3 &&
                    state == state_history[t] &&
                    state == state_history[t-1] &&
                    state == state_history[t-2]
                        for n in 1:N_AGENTS
                            agent_filled[n] = -1
                        end
                        break
                    end

                    # all filled?
                    if all(state[Compound(Symbol("has-filled"), Term[Const(Symbol("agent$n"))])] for n in 1:N_AGENTS)
                        break
                    end

                    # Terminate if all agents have filled their tank
                    filled_all = all(
                        state[Compound(Symbol("has-filled"),
                                    Term[Const(Symbol("agent$n"))])] for n in 1:N_AGENTS
                    )
                    if filled_all
                        break
                    end
                end
            end

                for n in 1:N_AGENTS
                    if agent_filled[n] == 0 # agent did not complete the task
                        agent_filled[n] = -1   
                    end
                end
            
            push!(output_rows, (
                run = run, 
                agent1 = agent_filled[1],
                agent2 = agent_filled[2],
                agent3 = agent_filled[3],
                agent4 = agent_filled[4],
                agent5 = agent_filled[5],
                agent6 = agent_filled[6],
                agent7 = agent_filled[7],
                agent8 = agent_filled[8],
                temperature = TEMPERATURE,
                time_max = TIME_MAX,
                n_agents = N_AGENTS,
                map = replace(map, ".pddl" => ""),
                seed = seed_this_run,
            ))    

            data[run] = agent_filled
        end
        # Save per-(map, runs) results.
        csv_name = joinpath(OUTPUT_DIR, replace(map, ".pddl" => "") * ".csv")
        #print(csv_name)
        #csv_name = string(COMPLETION_TIMES_DIR, map, "_", TEMPERATURE, "_", RUNS, ".csv")
        CSV.write(csv_name, output_rows)
    end
end

# Run if called as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulations()
end
