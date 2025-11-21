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
