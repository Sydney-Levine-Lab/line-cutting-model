using PDDL, PlanningDomains
using SymbolicPlanners
using CSV

include("water_collection_heuristic.jl")

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

const N_AGENTS     = 8                   # Must match problem files
const TIME_MAX     = 1000             # Upper bound on timesteps for a single run
const TEMPERATURE  = 0.3                 # Boltzmann temperature
const RUNS         = 5                   # Number of runs per map

const COMPLETION_TIMES_DIR  = "../data/simulations/test2/"
const FULL_HISTORY_DIR      = joinpath(COMPLETION_TIMES_DIR, "full_histories")

const DOMAIN_PATH = "."
const MAPS_PATH   = "./maps"

# The experiment set: names correspond to files under maps/.
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
# "yes_line_A.pddl" not used in stimuli



# --------------------------------------------------------------------
# Level-0 projection: others as walls
# --------------------------------------------------------------------

"""
    project_others_to_walls(state, k, N, domain)

Return a single-agent view for agent k by:
- deleting all other agent objects, and
- turning their last positions into walls.

This is the level-0 projection used for planning: the focal agent sees
others as static obstacles.
"""
function project_others_to_walls(state::State, k::Integer, N::Int, domain::Domain)
    # define objects and their types
    objtypes = PDDL.get_objtypes(state)
    # store the coordinates of all the agents that get deleted, to place walls there later
    agent_locs = Array{Tuple{Int,Int}}(undef, N-1)
    agent_index = 1
    # iterate over all agent objects. Unless it is the current agent, turn it into a wall
    for n in 1:N
        if n == k
            continue
        end
        agent = Const(Symbol("agent$n"))
        #save the coordinates of the agent, in order to place a wall there
        agent_locs[agent_index] = ((state[Compound(:xloc, Term[agent])], state[Compound(:yloc, Term[agent])]))
        agent_index += 1
        #then, delete the agent completely
        delete!(objtypes, agent)
    end
    # get the old wall matrix
    walls = copy(state[pddl"(walls)"])
    # it looks like this:
    # Bool[0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 1 1; 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 0 0 0 1 1 1 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 1 1 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    # modify wall matrix to include walls where agents were
    for loc in agent_locs
        walls[loc[2], loc[1]] = true
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

    # new state is missing this: Set(Term[not(has-water(agent2)), not(has-filled(agent1)), not(has-water(agent1)), not(has-filled(agent2))])
    #fluents is a dictionary of assignments to fluents (will have true/false values, array vals, ..)
    #bools default false
    fluents[Compound(Symbol("has-filled"), Term[agent])] = state[Compound(Symbol("has-filled"), Term[agent])]
    fluents[Compound(Symbol("has-water1"), Term[agent])] = state[Compound(Symbol("has-water1"), Term[agent])]
    fluents[Compound(Symbol("has-water2"), Term[agent])] = state[Compound(Symbol("has-water2"), Term[agent])]
    fluents[Compound(Symbol("has-water3"), Term[agent])] = state[Compound(Symbol("has-water3"), Term[agent])]

    new_state = initstate(domain, objtypes, fluents)
    #print(new_state)
    return new_state
end

inner_heuristic = WaterCollectionHeuristic() # Optimistic heuristic
planner = AStarPlanner(inner_heuristic) # Planner that uses optimistic heuristic
heuristic = PlannerHeuristic(planner) # Some exact heuristic
memoized_h = memoized(heuristic) # Memoized exact heuristic

# Number of agents (must match the problem files)
N=8 







multi_agent_lines = PlanningDomains.load_domain(joinpath(DOMAIN_PATH, "domain.pddl"))
domain = multi_agent_lines




for problem in ALL_MAPS
    print(problem)
    mal_problem = PlanningDomains.load_problem(joinpath(MAPS_PATH, problem))
    # Register array theory for gridworld domains (required for wall grids)
    PDDL.Arrays.@register()

    mal_state = initstate(multi_agent_lines, mal_problem)
    mal_spec = Specification(mal_problem)

    state = mal_state
    domain = multi_agent_lines

    T=1000 # Upper bound on timesteps for a single run
    #boltzmann_policy_parameters = [0.0001, 0.001, 0.01, 0.1, 1]
    boltzmann_policy_parameters = [0.3]

    runs=5

    # Nested: data[noise][iteration] = Dict(agent_id => first_fill_timestep | -1)
    data = Dict()
    for parameter in boltzmann_policy_parameters
        data[parameter] = Dict{Int64, Dict{Int64, Int64}}()
    end

    #iterating over all the boltzmann_policy_parameter values
    for noise in boltzmann_policy_parameters
        #iterate loop on each boltzmann_policy_parameter value 5 times each
        for iteration in 1:runs
            #print("noise: ", noise, " iteration: ", iteration, "\n")

            state = initstate(domain, mal_problem)
            state_history = [state]

            boltzmann_policies = Array{BoltzmannPolicy}(undef, N)
            for k in 1:N
                agent = Const(Symbol("agent$k"))
                mal_spec = MinStepsGoal(Term[Compound(Symbol("has-filled"), Term[agent])])
                inner_policy = FunctionalVPolicy(memoized_h, domain, mal_spec) # Policy that evaluates every state
                boltzmann_policy = BoltzmannPolicy(inner_policy, noise) # Boltzmann agent
                boltzmann_policies[k] = boltzmann_policy
            end

            #create a dictionary where the key is the agent number and the value is the time step that the agent has-filled
            agent_filled = Dict{Int64, Int64}(1 => 0, 2 => 0, 3 => 0, 4 => 0, 5 => 0, 6 => 0, 7 => 0, 8 => 0)

            #for saving state_history
            state_history_name = string("state_history", "_", problem, "_", noise, "_runs", runs, "_iteration", iteration, ".txt")
            file = open(joinpath(FULL_HISTORY_DIR, state_history_name), "w")
            write(file, string(state))
            close(file)

            for t in 1:T
                if t % 10 == 0
                    println("Timestep $t")
                end
                #actions = Term[]
                interim_state = state
                for k in 1:N
                    agent = Const(Symbol("agent$k"))
                    modified_state = project_others_to_walls(interim_state, k, N, domain) # Change other agents into walls                    
                    act = SymbolicPlanners.get_action(boltzmann_policies[k], modified_state)
                    interim_state = transition(domain, interim_state, act)
                end
            
                state = interim_state

                #save object state_history as text to a new file in current directory
                file = open(joinpath(FULL_HISTORY_DIR, state_history_name), "a")
                write(file, string(state))
                close(file)

                push!(state_history, state)
        
                # if any agent has filled the tank, then record the time step into the agent_filled dictionary
                for n in 1:N
                    agent = Const(Symbol("agent$n"))
                    if state[Compound(Symbol("has-filled"), Term[agent])]
                        #only change the value if it is equal to 0, otherwise constant overwriting.
                        if agent_filled[n] == 0
                            agent_filled[n] = t
                        end
                    end
                end
        
                # if state didn't change, then set all agent_filled values to -1
                if state == state_history[t] && state == state_history[t-1] && state == state_history[t-2] 
                    for n in 1:N
                        agent_filled[n] = -1
                    end
                    break
                end
        
                # if all the goals have been met, terminate the loop
                if all([state[Compound(Symbol("has-filled"), Term[Const(Symbol("agent$n"))])] for n in 1:N])
                    break
                end
            end

            #add the agent_filled dictionary to the data dictionary
            data[noise][iteration] = agent_filled
            #print(data[runs][iteration], "\n")
        end
        # Save per-(map, runs) results. CSV will be appended to if rerun.
        csv_name = string(COMPLETION_TIMES_DIR, problem, "_", noise, "_", runs, ".csv")
        CSV.write(csv_name, data, append=true)
    end
end



function run_simulations()
    # Load domain
    domain = PlanningDomains.load_domain(joinpath(DOMAIN_PATH, "domain.pddl"))

    mkpath(COMPLETION_TIMES_D
       
    # Register array theory for gridworld domains (required for wall grids)
    PDDL.Arrays.@register()

    # Set up heuristic and planner
    inner_heuristic = WaterCollectionHeuristic()         # optimistic heuristic
    planner         = AStarPlanner(inner_heuristic)      # planner using heuristic
    heuristic       = PlannerHeuristic(planner)          # exact heuristic via planning
    memoized_h      = memoized(heuristic)                # cached version

    for map in ALL_MAPS
        println(map)

        mal_map = PlanningDomains.load_problem(joinpath(MAPS_PATH, problem))
        mal_state   = initstate(multi_agent_lines, mal_map)
        mal_spec    = Specification(mal_map)

        state  = mal_state
        domain = multi_agent_lines

        # Use a single noise value for now, but keep structure for future sweep
        boltzmann_policy_parameters = [NOISE]

        # Nested: data[noise][iteration] = Dict(agent_id => first_fill_timestep | -1)
        data = Dict{Float64, Dict{Int, Dict{Int, Int}}}()
        for parameter in boltzmann_policy_parameters
            data[parameter] = Dict{Int, Dict{Int, Int}}()
        end

        for iteration in 1:RUNS
            # Reset initial state and history
            state = initstate(domain, mal_map)
            state_history = [state]

            # Build per-agent Boltzmann policies
            boltzmann_policies = build_boltzmann_policies(domain, memoized_h, noise, N_AGENTS)

            # Dictionary: agent_id => timestep of first fill (or -1 if never)
            agent_filled = Dict{Int, Int}()
            for n in 1:N_AGENTS
                agent_filled[n] = 0
            end

            # Prepare state history file
            state_history_name = string(
                "state_history_", map, "_", noise,
                "_runs", RUNS, "_iteration", iteration, ".txt",
            )
            file = open(joinpath(STATE_HISTORY_PATH, state_history_name), "w")
            write(file, string(state))
            close(file)

            for t in 1:TIME_MAX

                # One level-0 timestep (all agents move once)
                state = level0_step!(state, boltzmann_policies, domain, N_AGENTS)

                # Append state to history file
                file = open(joinpath(STATE_HISTORY_PATH, state_history_name), "a")
                write(file, string(state))
                close(file)

                push!(state_history, state)

                # If any agent has filled a tank, record the timestep
                for n in 1:N_AGENTS
                    agent = Const(Symbol("agent$n"))
                    if state[Compound(Symbol("has-filled"), Term[agent])]
                        if agent_filled[n] == 0
                            agent_filled[n] = t
                        end
                    end
                end

                # If state stopped changing for 3 timesteps, set all to -1 and stop
                if t ≥ 3 && state == state_history[t] &&
                            state == state_history[t-1] &&
                            state == state_history[t-2]
                    for n in 1:N_AGENTS
                        agent_filled[n] = -1
                    end
                    break
                end

                # If all goals are met, terminate the loop
                if all([state[Compound(Symbol("has-filled"),
                                        Term[Const(Symbol("agent$n"))])] for n in 1:N_AGENTS])
                    break
                end
            end

            data[noise][iteration] = agent_filled
        end

        # Save per-(map, runs) results. CSV will be appended to if rerun.
        csv_name = string(DATA_SAVE_PATH, map, "_", noise, "_", RUNS, ".csv")
        CSV.write(csv_name, data, append=true)
    end
end

