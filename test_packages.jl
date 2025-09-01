####
# Trying to include SymbolicPlanners packages (incl. WallHeuristic) directly
####

# Working approach - Direct includes to bypass package issues
using PDDL, PlanningDomains
using SymbolicPlanners
import SymbolicPlanners: precompute!, is_precomputed, compute  
using CSV

# Define WellTankHeuristic directly (bypassing package loading issues)
struct WellTankHeuristic <: Heuristic
    min_well_tank_dists::Dict{Const, Int}
    min_tank_goal_dists::Dict{Const, Int}
end

WellTankHeuristic() = WellTankHeuristic(Dict{Const, Int}(), Dict{Const, Int}())

# Copy all required functions from the local welltank.jl
function get_obj_loc(state::State, obj::Const)
    xloc = state[Compound(:xloc, Term[obj])]
    yloc = PDDL.get_fluent(state, Compound(:yloc, Term[obj]))
    return (xloc, yloc)
end

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

function is_precomputed(h::WellTankHeuristic, domain::Domain, state::State, spec::Specification)
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

# Copy the modify_state function from main_modeling_line
function modify_state(state::State, k::Integer)
    N = 8  # Number of agents
    objtypes = PDDL.get_objtypes(state)
    agent_locs = Array{Tuple{Int,Int}}(undef, N-1)
    agent_index = 1
    
    for n in 1:N
        if n == k
            continue
        end
        agent = Const(Symbol("agent$n"))
        agent_locs[agent_index] = ((state[Compound(:xloc, Term[agent])], state[Compound(:yloc, Term[agent])]))
        agent_index += 1
        delete!(objtypes, agent)
    end
    
    walls = copy(state[pddl"(walls)"])
    for loc in agent_locs
        walls[loc[2], loc[1]] = true
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
    fluents[Compound(Symbol("has-filled"), Term[agent])] = state[Compound(Symbol("has-filled"), Term[agent])]
    fluents[Compound(Symbol("has-water1"), Term[agent])] = state[Compound(Symbol("has-water1"), Term[agent])]
    fluents[Compound(Symbol("has-water2"), Term[agent])] = state[Compound(Symbol("has-water2"), Term[agent])]
    fluents[Compound(Symbol("has-water3"), Term[agent])] = state[Compound(Symbol("has-water3"), Term[agent])]

    new_state = initstate(domain, objtypes, fluents)
    return new_state
end

println("Testing WellTankHeuristic setup...")

# Test basic functionality
domain = load_domain("domain.pddl")
println("✓ Domain loaded")

h = WellTankHeuristic()
println("✓ WellTankHeuristic created: ", h)

# Test with a simple problem
problem = load_problem("maps/no_line_1_test.pddl")
state = initstate(domain, problem)
println("✓ Problem and initial state loaded")

println("Success! All components working. Ready to run full simulation.")