using PDDL, PlanningDomains
using SymbolicPlanners
using CSV

include("water_collection_heuristic.jl")

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

const N_AGENTS    = 8
const TIME_MAX    = 1000
const TEMPERATURE = 0.3
const RUNS        = 5

const COMPLETION_TIMES_DIR = "../data/simulations/test2/"
const FULL_HISTORY_DIR     = joinpath(COMPLETION_TIMES_DIR, "state_histories")

const DOMAIN_PATH = "."
const MAPS_PATH   = "./maps"

const ALL_MAPS = [
    "no_line_1_test.pddl", "no_line_2_test.pddl", "no_line_3_test.pddl",
    "yes_line_7_test.pddl", "yes_line_8_test.pddl", "yes_line_9_test.pddl", "yes_line_10_test.pddl",
    "7esque_test.pddl", "9esque_test.pddl", "10esque_test.pddl",
    "maybe_4.pddl", "maybe_5.pddl", "maybe_6.pddl",
    "new_maybe_1.pddl", "new_maybe_2.pddl", "new_maybe_3.pddl",
    "new_maybe_4.pddl", "new_maybe_5.pddl", "new_maybe_6.pddl",
    "no_line_A.pddl", "no_line_B.pddl", "no_line_C.pddl", "no_line_D.pddl",
    "yes_line_B.pddl", "yes_line_C.pddl", "yes_line_D.pddl", "yes_line_E.pddl", "yes_line_F.pddl",
]

# --------------------------------------------------------------------
# Helper Functions (ALL DEFINED AT TOP LEVEL)
# --------------------------------------------------------------------

function project_others_to_walls(state::State, k::Int, N::Int, domain::Domain)
    objtypes = PDDL.get_objtypes(state)
    agent_locs = Array{Tuple{Int,Int}}(undef, N - 1)
    agent_index = 1

    for n in 1:N
        if n == k
            continue
        end
        agent = Const(Symbol("agent$n"))
        agent_locs[agent_index] = (
            state[Compound(:xloc, Term[agent])],
            state[Compound(:yloc, Term[agent])]
        )
        agent_index += 1
        delete!(objtypes, agent)
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

    agent = Const(Symbol("agent$k"))
    fluents[pddl"(walls)"] = walls
    fluents[Compound(:xloc, Term[agent])] = state[Compound(:xloc, Term[agent])]
    fluents[Compound(:yloc, Term[agent])] = state[Compound(:yloc, Term[agent])]
    fluents[Compound(Symbol("has-filled"),  Term[agent])] = state[Compound(Symbol("has-filled"),  Term[agent])]
    fluents[Compound(Symbol("has-water1"), Term[agent])] = state[Compound(Symbol("has-water1"), Term[agent])]
    fluents[Compound(Symbol("has-water2"), Term[agent])] = state[Compound(Symbol("has-water2"), Term[agent])]
    fluents[Compound(Symbol("has-water3"), Term[agent])] = state[Compound(Symbol("has-water3"), Term[agent])]

    new_state = initstate(domain, objtypes, fluents)
    return new_state
end

function build_boltzmann_policies(domain::Domain, memoized_h, temperature::Float64, N::Int)
    policies = Vector{BoltzmannPolicy}(undef, N)
    for n in 1:N
        agent = Const(Symbol("agent$n"))
        spec  = MinStepsGoal(Term[Compound(Symbol("has-filled"), Term[agent])])
        inner_policy  = FunctionalVPolicy(memoized_h, domain, spec)
        boltzmann_pol = BoltzmannPolicy(inner_policy, temperature)
        policies[n]   = boltzmann_pol
    end
    return policies
end

function level0_step!(state::State, policies::AbstractVector{BoltzmannPolicy}, domain::Domain, N::Int)
    interim_state = state
    for n in 1:N
        modified_state = project_others_to_walls(interim_state, n, N, domain)
        act = SymbolicPlanners.get_action(policies[n], modified_state)
        interim_state = transition(domain, interim_state, act)
    end
    return interim_state
end

# --------------------------------------------------------------------
# Main Simulation Function
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Main simulation loop
# --------------------------------------------------------------------

function run_simulations()
    # Make sure output directories exist
    mkpath(COMPLETION_TIMES_DIR)
    mkpath(FULL_HISTORY_DIR)

    # Load domain
    domain = PlanningDomains.load_domain(joinpath(DOMAIN_PATH, "domain.pddl"))

    # Register array theory for gridworld domains (required for wall grids)
    PDDL.Arrays.@register()

    # Set up heuristic and planner
    inner_heuristic   = WaterCollectionHeuristic()       # optimistic heuristic
    planner           = AStarPlanner(inner_heuristic)    # planner using heuristic
    planner_heuristic = PlannerHeuristic(planner)        # exact heuristic via planning
    memoized_h        = memoized(planner_heuristic)      # cached version

    for map in ALL_MAPS
        println(map)

        # Load problem / initial state for this map
        problem       = PlanningDomains.load_problem(joinpath(MAPS_PATH, map))
        initial_state = initstate(domain, problem)

        # For each map, store per-run completion times
        # data[run] = Dict(agent_id => first_fill_timestep | -1)
        data = Dict{Int, Dict{Int, Int}}()

        for run in 1:RUNS
            # Reset state and history
            state = initial_state
            state_history = [state]

            # Build per-agent Boltzmann policies
            boltzmann_policies = build_boltzmann_policies(domain, memoized_h, TEMPERATURE, N_AGENTS)

            # agent_filled[n] = timestep when agent n first fills a tank (0 if not yet, -1 if stuck)
            agent_filled = Dict{Int, Int}(n => 0 for n in 1:N_AGENTS)

            # Prepare state history file
            state_history_name = string(
                "state_history_", map, "_", TEMPERATURE,
                "_runs", RUNS, "_run", run, ".txt",
            )
            history_path = joinpath(FULL_HISTORY_DIR, state_history_name)
            open(history_path, "w") do file
                write(file, string(state))
            end

            for t in 1:TIME_MAX
                # One level-0 timestep (all agents move once)
                state = level0_step!(state, boltzmann_policies, domain, N_AGENTS)

                # Append state to history file
                open(history_path, "a") do file
                    write(file, string(state))
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

                # If all goals are met, terminate the loop
                filled_all = all(
                    state[Compound(Symbol("has-filled"),
                                   Term[Const(Symbol("agent$n"))])] for n in 1:N_AGENTS
                )
                if filled_all
                    break
                end
            end  # end for t in 1:TIME_MAX

            data[run] = agent_filled
        end  # end for run in 1:RUNS

        # Save per-(map, runs) results.
        csv_name = string(COMPLETION_TIMES_DIR, map, "_", TEMPERATURE, "_", RUNS, ".csv")
        CSV.write(csv_name, data, append=true)
    end  # end for map in ALL_MAPS
end  # end function run_simulations()

# --------------------------------------------------------------------
# Execute
# --------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    run_simulations()
end