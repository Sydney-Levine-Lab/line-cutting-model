using PDDL, PlanningDomains
using SymbolicPlanners
import SymbolicPlanners: precompute!, is_precomputed, compute
using CSV

# -----------------------------------------------------------------------------
# WellTankHeuristic needs to here: see main_level_0_reasoning
# -----------------------------------------------------------------------------
export WellTankHeuristic
#=
    This is a custom data structure for the heuristic, which contains 
    a dictionary called min_well_tank_dists. The dictionary maps well objects (Const) to the minimum 
    distances (Int) between the well and any tank.
=#
struct WellTankHeuristic <: Heuristic
    min_well_tank_dists::Dict{Const, Int}
    min_tank_goal_dists::Dict{Const, Int}
end

#=
    This is a constructor for the WellTankHeuristic structure. 
    It initializes the min_well_tank_dists dictionary.
=#
function WellTankHeuristic()
    min_well_tank_dists = Dict{Const, Int}()
    min_tank_goal_dists = Dict{Const, Int}()
    return WellTankHeuristic(min_well_tank_dists, min_tank_goal_dists)
end

#=
    This function takes a state and an object 
    (a well, tank, or agent) and returns the object's (x, y) location in the state.
=#
function get_obj_loc(state::State, obj::Const)
    xloc = state[Compound(:xloc, Term[obj])]
    yloc = PDDL.get_fluent(state, Compound(:yloc, Term[obj]))
    return (xloc, yloc)
end

#=
    This function precomputes the minimum distances between each well and any tank in the state 
    and stores them in the min_well_tank_dists dictionary. It first clears the dictionary, 
    then calculates the distances and updates the dictionary accordingly.
=#
function precompute!(h::WellTankHeuristic, domain::Domain, state::State, spec::Specification)
    wells = PDDL.get_objects(state, :well)
    tanks = PDDL.get_objects(state, :tank)
    finishes = PDDL.get_objects(state, :finish)

    empty!(h.min_well_tank_dists)
    empty!(h.min_tank_goal_dists)

    well_locs = [get_obj_loc(state, w) for w in wells]
    tank_locs = [get_obj_loc(state, t) for t in tanks]
    finish_locs = [get_obj_loc(state, f) for f in finishes]

    for (w, well_loc) in zip(wells, well_locs)
        min_dist = typemax(Int)
        for tank_loc in tank_locs
            well_tank_dist = sum(abs.(well_loc .- tank_loc))
            min_dist = min(well_tank_dist, min_dist)
        end
        h.min_well_tank_dists[w] = min_dist
    end

    for (t, tank_loc) in zip(tanks, tank_locs)
        min_dist = typemax(Int)
        for finish_loc in finish_locs
            tank_goal_dist = sum(abs.(finish_loc .- tank_loc))
            min_dist = min(tank_goal_dist, min_dist)
        end
        h.min_tank_goal_dists[t] = min_dist
    end

    return h
end


#=
    This function checks if the heuristic has already precomputed the minimum distances 
    in the min_well_tank_dists dictionary.
=#
function is_precomputed(h::WellTankHeuristic,
                        domain::Domain, state::State, spec::Specification)
    return isdefined(h, :min_well_tank_dists) && isdefined(h, :min_tank_goal_dists)
end

function compute(h::WellTankHeuristic, domain::Domain, state::State, spec::Specification)
    # Precompute if necessary
    if !is_precomputed(h, domain, state, spec)
        precompute!(h, domain, state, spec)
    end

    # Find max heuristic distance to goal across agents
    max_dist = 0

    agents = PDDL.get_objects(state, :agent)
    wells = PDDL.get_objects(state, :well)
    tanks = PDDL.get_objects(state, :tank)
    finishes = PDDL.get_objects(state, :finish)

    for a in agents 
        entire_completed = Compound(Symbol("has-completed"), Term[a])
        task_completed = Compound(Symbol("has-filled"), Term[a])
        water_collected = state[Compound(Symbol("has-water1"), Term[a])] || state[Compound(Symbol("has-water2"), Term[a])] || state[Compound(Symbol("has-water3"), Term[a])]
            
        #condition, agent has completed the tanks
        if state[entire_completed]
            continue
        end

        agent_loc = get_obj_loc(state, a)
        agent_dist = Inf

        #condition, agent hasn't gotten the water yet
        if !water_collected
            min_well_dist, min_well = find_nearest_obj(agent_loc, wells, state)
            min_tank_dist = h.min_well_tank_dists[min_well]
            min_goal_dist = minimum([h.min_tank_goal_dists[t] for t in keys(h.min_tank_goal_dists)])
            agent_dist = min_well_dist + min_tank_dist + min_goal_dist
        elseif !state[task_completed] # agent has collected water, but hasn't stored it in a tank yet
            min_tank_dist, min_tank = find_nearest_obj(agent_loc, tanks, state)
            min_goal_dist = h.min_tank_goal_dists[min_tank]
            agent_dist = min_tank_dist + min_goal_dist
        else # agent has filled the tank with water
            min_goal_dist, _ = find_nearest_obj(agent_loc, finishes, state)
            agent_dist = min_goal_dist
        end

        max_dist = max(max_dist, agent_dist)
    end 
    return max_dist
end

function find_nearest_obj(agent_loc, objs, state)
    min_dist = Inf
    min_obj = nothing
    for obj in objs
        obj_loc = get_obj_loc(state, obj)
        dist = sum(abs.(obj_loc .- agent_loc))
        if dist < min_dist
            min_dist = dist
            min_obj = obj
        end
    end
    return min_dist, min_obj
end




# -----------------------------------------------------------------------------
# Script purpose
# -----------------------------------------------------------------------------
# Test Joe's script (used for 2023 CSS paper)

# Load domain and problem - (local file names)
domain_path = "."
state_history_path = "./state_histories"
data_save_path = "./maps/data/"
multi_agent_lines = load_domain(joinpath(domain_path, "domain.pddl"))
problem_path = "./maps"
#mal_problem = load_problem(joinpath(problem_path, "new_maybe_3.pddl"))
domain = multi_agent_lines

# The experiment set: names correspond to files under maps/.
all_problems = ["no_line_1_test.pddl", "yes_line_F.pddl"]
#ll_problems = ["no_line_1_test.pddl", "no_line_2_test.pddl", "no_line_3_test.pddl", "yes_line_7_test.pddl", "yes_line_8_test.pddl", "yes_line_9_test.pddl", "yes_line_10_test.pddl", "7esque_test.pddl", "9esque_test.pddl", "10esque_test.pddl", "maybe_4.pddl", "maybe_5.pddl", "maybe_6.pddl", "new_maybe_1.pddl", "new_maybe_2.pddl", "new_maybe_3.pddl", "new_maybe_4.pddl", "new_maybe_5.pddl", "new_maybe_6.pddl", "no_line_A.pddl", "no_line_B.pddl", "no_line_C.pddl", "no_line_D.pddl", "yes_line_A.pddl", "yes_line_B.pddl", "yes_line_C.pddl", "yes_line_D.pddl", "yes_line_E.pddl", "yes_line_F.pddl"]

# Helper: project to a single-agent view for agent k by deleting other agents
# and turning their last positions into walls.
# define function to modify state to be a single agent state (turning all other agents into walls)
# CHANGED name: this is level 0 reasoning
function modify_state_level_0(state::State, k::Integer)
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

# ------------------------
# CHANGED: adding level 1 reasoning function
#
function modify_state_level_1(state::State, k::Integer, boltzmann_policies, domain::Domain)
    # Step 1: Predict what each other agent will do using level 0 reasoning
    predicted_state = state
    
    for n in 1:N
        if n == k
            continue  # Skip current agent
        end
        
        try
            # What would agent n do if they used level 0 reasoning?
            level0_state = modify_state_level_0(state, n)
            predicted_action = SymbolicPlanners.get_action(boltzmann_policies[n], level0_state)
            
            # Apply that action to our predicted world
            predicted_state = transition(domain, predicted_state, predicted_action)
        catch e
            # If prediction fails (invalid action), assume agent stays in place
            println("    Warning: Could not predict action for agent $n from point of view of agent $k, assuming stays put")
            # predicted_state remains unchanged for this agent
        end
    end
    
    # Step 2: Now create single-agent view of this predicted world for agent k
    return modify_state_level_0(predicted_state, k)
end

inner_heuristic = WellTankHeuristic() # Optimistic heuristic
planner = AStarPlanner(inner_heuristic) # Planner that uses optimistic heuristic
heuristic = PlannerHeuristic(planner) # Some exact heuristic
memoized_h = memoized(heuristic) # Memoized exact heuristic

# Number of agents (must match the problem files)
N=8 

#define multi agent domain
struct MultiAgentDomain{D <: Domain} <: Domain
    domain::D
end #TODO: modify transition dynamic. allow multiple actions and handle collisions. was used in multi rtdp

#transition to roll out action and check for collisions
# Note: a simultaneous multi-agent transition exists below but is not used in
# this script's main loop (we update sequentially instead). Kept for reference.
# function PDDL.transition(domain::MultiAgentDomain, state::State, actions)
#     agent_locs = []
#     #basically, right now we iterate through the list of agents in the same order each time setup
#     for (i, act) in enumerate(actions)
# 	    agent = Const(Symbol("agent$i"))
#         next_state = transition(domain.domain, state, act)
#         next_agent_loc = (next_state[Compound(:xloc, Term[agent])], next_state[Compound(:yloc, Term[agent])])

#         #if an agent tries to move into a square that a preceding agent has moved into, then this agent stays still
# 	    if next_agent_loc in agent_locs
# 	        agent_loc = (state[Compound(:xloc, Term[agent])], state[Compound(:yloc, Term[agent])]) 
#             push!(agent_locs, agent_loc)
#         #no collision, so move the agent into the new location
#         else
#             state = next_state
#             push!(agent_locs, next_agent_loc)
#         end
#     end
#     return state
# end

# Not used below (sequential rollout). Retained for potential future use.
multi_domain = MultiAgentDomain(domain)


for problem in all_problems
    mal_problem = load_problem(joinpath(problem_path, problem))
    # Register array theory for gridworld domains (required for wall grids)
    PDDL.Arrays.@register()

    mal_state = initstate(multi_agent_lines, mal_problem)
    mal_spec = Specification(mal_problem)

    state = mal_state
    domain = multi_agent_lines

    T=1000 # Upper bound on timesteps for a single run
    boltzmann_policy_parameters = [0.0001]

    # Nested: data[noise][iteration] = Dict(agent_id => first_fill_timestep | -1)
    data = Dict()
    collision_data = Dict() # collecting data about collisions too
    for parameter in boltzmann_policy_parameters
        data[parameter] = Dict{Int64, Dict{Int64, Int64}}()
    end

    #iterating over all the boltzmann_policy_parameter values
    for noise in boltzmann_policy_parameters
        #iterate loop on each boltzmann_policy_parameter value 5 times each
        for iteration in 1:5
            #print("noise: ", noise, " iteration: ", iteration, "\n")

            state = initstate(domain, mal_problem)
            state_history = [state]

            boltzmann_policies = Array{BoltzmannPolicy}(undef, N)
            for n in 1:N
                agent = Const(Symbol("agent$n"))
                mal_spec = MinStepsGoal(Term[Compound(Symbol("has-filled"), Term[agent])])
                inner_policy = FunctionalVPolicy(memoized_h, domain, mal_spec) # Policy that evaluates every state
                boltzmann_policy = BoltzmannPolicy(inner_policy, noise) # Boltzmann agent
                boltzmann_policies[n] = boltzmann_policy
            end

            #create a dictionary where the key is the agent number and the value is the time step that the agent has-filled
            agent_filled = Dict{Int64, Int64}(1 => 0, 2 => 0, 3 => 0, 4 => 0, 5 => 0, 6 => 0, 7 => 0, 8 => 0)

            #for saving state_history
            state_history_name = string("state_history", "_", problem, "_", noise, "_iteration", iteration, ".txt")
            file = open(joinpath(state_history_path, state_history_name), "w")
            write(file, string(state))
            close(file)

            collision_count = 0

            for t in 1:T
                println("Timestep $t")
                # Print all agent positions at start of timestep
                print("Collisions:")
                println(collision_count)
               print("Positions: ")
                for n in 1:N
                    agent = Const(Symbol("agent$n"))
                    pos = (state[Compound(:xloc, Term[agent])], state[Compound(:yloc, Term[agent])])
                    print("A$n:$pos ")
                end
                println()
                #actions = Term[]
             
                # CHANGED: NEW SIMULTANEOUS CODE:
                actions = []
                for n in 1:N
                    agent = Const(Symbol("agent$n"))
                    modified_state = modify_state_level_1(state, n, boltzmann_policies, domain)
                    act = SymbolicPlanners.get_action(boltzmann_policies[n], modified_state)
                    push!(actions, (n, act))
                    println("  Agent $n chose: $act")
                end
                
                # Calculate all intended moves simultaneously
                intended_moves = Dict{Int, Tuple{Int,Int}}()
                valid_actions = Dict{Int, Any}()
                
                for (n, act) in actions
                    agent = Const(Symbol("agent$n"))
                    current_pos = (state[Compound(:xloc, Term[agent])], state[Compound(:yloc, Term[agent])])
                    
                    # Calculate intended position based on action
                    if occursin("up", string(act))
                        intended_pos = (current_pos[1], current_pos[2] - 1)
                    elseif occursin("down", string(act))
                        intended_pos = (current_pos[1], current_pos[2] + 1)
                    elseif occursin("left", string(act))
                        intended_pos = (current_pos[1] - 1, current_pos[2])
                    elseif occursin("right", string(act))
                        intended_pos = (current_pos[1] + 1, current_pos[2])
                    else
                        intended_pos = current_pos  # Non-movement actions
                    end
                    
                    intended_moves[n] = intended_pos
                    valid_actions[n] = act
                end
                
                # Check for conflicts (multiple agents trying to move to same position)
                position_conflicts = Dict{Tuple{Int,Int}, Vector{Int}}()
                for (n, pos) in intended_moves
                    if !haskey(position_conflicts, pos)
                        position_conflicts[pos] = []
                    end
                    push!(position_conflicts[pos], n)
                end
                
                # Resolve conflicts: if multiple agents want same position, none get it
                conflicted_agents = Set{Int}()
                for (pos, agents) in position_conflicts
                    if length(agents) > 1
                        println("  Conflict at $pos between agents: $agents")
                        for agent_id in agents
                            push!(conflicted_agents, agent_id)
                        end
                    end
                end
                
                # Apply all non-conflicted actions simultaneously
                interim_state = state
                for (n, act) in actions
                    if n in conflicted_agents
                        println("  Agent $n: COLLISION (stayed put)")
                        collision_count += 1
                    else
                        try
                            interim_state = transition(domain, interim_state, act)
                            println("  Agent $n: SUCCESS")
                        catch e
                            println("  Agent $n: COLLISION (invalid move)")
                            collision_count += 1
                        end
                    end
                end
                
                state = interim_state

                #save object state_history as text to a new file in current directory
                file = open(joinpath(state_history_path, state_history_name), "a")
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
            collision_data[noise][iteration] = collision_count  # Track collisions separately
            #print(data[noise][iteration], "\n")
        end
        # Save per-(map, noise) results. CSV will be appended to if rerun.
        csv_name = string(data_save_path, problem, "_", noise, ".csv")
        CSV.write(csv_name, data, append=true)

        collision_csv_name = string(data_save_path, "collisions_", problem, "_", noise, ".csv")
        CSV.write(collision_csv_name, collision_data[noise], append=true)   
    end
end