# Goal-matched distance guide for collecting and depositing water.

using PDDL
using SymbolicPlanners
import SymbolicPlanners: precompute!, is_precomputed, compute

"""
    WaterCollectionHeuristic()

Distance to `has-filled`, including the three collection actions and the
deposit action. Finish squares and post-deposit travel are not part of this
goal. Already deposited agents therefore have distance zero.

A reverse breadth-first search precomputes distances over position and four
water phases: empty, collect1, collect2, and full. Movement is allowed only
when empty or full, as in `domain.pddl`. Wells and tanks are service objects,
not walkable cells. All legal adjacent service positions are considered.

Other agent objects are ignored in this distance calculation. In the actual
single-agent planning views their positions are already represented as walls,
so the distances are exact for that fixed view. In a multi-agent state they
are a relaxation and their maximum remains an admissible lower bound.

Only the most recent geometry is cached. Wall contents and all well/tank
coordinates are checked before reuse, including direct calls to `compute`.
Unsupported goals use zero, preserving uninformed shortest-path search.
"""
mutable struct WaterCollectionHeuristic <: Heuristic
    walls::BitMatrix
    well_locs::Vector{Tuple{Int,Int}}
    tank_locs::Vector{Tuple{Int,Int}}
    distances::Array{Float64,3}
    prepared::Bool
end

WaterCollectionHeuristic() = WaterCollectionHeuristic(
    falses(0, 0), Tuple{Int,Int}[], Tuple{Int,Int}[],
    Array{Float64,3}(undef, 0, 0, 4), false,
)

function get_obj_loc(state::State, obj::Const)
    return (state[Compound(:xloc, Term[obj])],
            state[Compound(:yloc, Term[obj])])
end

function _water_object_locations(state::State, kind::Symbol)
    return sort!(Tuple{Int,Int}[get_obj_loc(state, obj)
                               for obj in PDDL.get_objects(state, kind)])
end

# This guide only claims a lower bound for positive deposit goals with unit
# action costs. Coordinate, has-completed, disjunctive, and weighted goals
# deliberately fall back to h=0 rather than reusing the water-task estimate.
function _water_goal_agents(spec::Specification)
    spec isa SymbolicPlanners.MinStepsGoal || return nothing
    agents = Const[]
    for term in SymbolicPlanners.get_goal_terms(spec)
        term isa Compound || return nothing
        term.name == Symbol("has-filled") || return nothing
        length(term.args) == 1 && term.args[1] isa Const || return nothing
        push!(agents, term.args[1])
    end
    return agents
end

function is_precomputed(h::WaterCollectionHeuristic,
                        domain::Domain, state::State, spec::Specification)
    h.prepared || return false
    h.walls == state[pddl"(walls)"] || return false
    h.well_locs == _water_object_locations(state, :well) || return false
    return h.tank_locs == _water_object_locations(state, :tank)
end

function _water_service_cells(passable::BitMatrix, locations)
    height, width = size(passable)
    cells = falses(height, width)
    for (x, y) in locations
        for (dx, dy) in ((0, -1), (0, 1), (-1, 0), (1, 0))
            nx, ny = x + dx, y + dy
            if 1 <= nx <= width && 1 <= ny <= height && passable[ny, nx]
                cells[ny, nx] = true
            end
        end
    end
    return cells
end

function precompute!(h::WaterCollectionHeuristic,
                     domain::Domain, state::State, spec::Specification)
    # A* invokes precompute! before checking its initial goal. Finished ghosts
    # and unsupported goals need no distance table at all.
    goal_agents = _water_goal_agents(spec)
    goal_agents === nothing && return h
    isempty(goal_agents) && return h
    present_agents = PDDL.get_objects(state, :agent)
    if all(a -> a in present_agents, goal_agents) &&
       all(a -> state[Compound(Symbol("has-filled"), Term[a])], goal_agents)
        return h
    end
    is_precomputed(h, domain, state, spec) && return h
    h.walls = copy(BitMatrix(state[pddl"(walls)"]))
    h.well_locs = _water_object_locations(state, :well)
    h.tank_locs = _water_object_locations(state, :tank)
    height, width = size(h.walls)

    passable = .!h.walls
    for (x, y) in Iterators.flatten((h.well_locs, h.tank_locs))
        1 <= x <= width && 1 <= y <= height && (passable[y, x] = false)
    end
    well_service = _water_service_cells(passable, h.well_locs)
    tank_service = _water_service_cells(passable, h.tank_locs)
    distances = fill(Inf, height, width, 4)
    queue = Tuple{Int,Int,Int}[]
    sizehint!(queue, 4 * height * width)

    # Any full agent next to any tank can deposit in one action. Starting all
    # reverse searches at distance one makes this an ordinary unit-edge BFS.
    for cell in findall(tank_service)
        y, x = Tuple(cell)
        distances[y, x, 4] = 1.0
        push!(queue, (x, y, 4))
    end

    head = 1
    while head <= length(queue)
        x, y, phase = queue[head]
        head += 1
        predecessor_distance = distances[y, x, phase] + 1.0

        if phase == 1 || phase == 4
            for (dx, dy) in ((0, -1), (0, 1), (-1, 0), (1, 0))
                nx, ny = x + dx, y + dy
                if 1 <= nx <= width && 1 <= ny <= height &&
                   passable[ny, nx] && isinf(distances[ny, nx, phase])
                    distances[ny, nx, phase] = predecessor_distance
                    push!(queue, (nx, ny, phase))
                end
            end
        end

        # Reversing collect3, collect2, or collect1 changes the phase at the
        # same adjacent well cell. Partially collected agents cannot move.
        if phase > 1 && well_service[y, x] &&
           isinf(distances[y, x, phase - 1])
            distances[y, x, phase - 1] = predecessor_distance
            push!(queue, (x, y, phase - 1))
        end
    end

    h.distances = distances
    h.prepared = true
    return h
end

function compute(h::WaterCollectionHeuristic,
                 domain::Domain, state::State, spec::Specification)
    goal_agents = _water_goal_agents(spec)
    goal_agents === nothing && return 0.0
    isempty(goal_agents) && return 0.0
    present_agents = PDDL.get_objects(state, :agent)
    any(a -> !(a in present_agents), goal_agents) && return 0.0

    # Deposit clears water flags. Check the requested completion predicate
    # before interpreting those flags, including for wandering ghosts.
    unfinished = Const[a for a in goal_agents
                       if !state[Compound(Symbol("has-filled"), Term[a])]]
    isempty(unfinished) && return 0.0
    is_precomputed(h, domain, state, spec) || precompute!(h, domain, state, spec)

    max_distance = 0.0
    height, width, _ = size(h.distances)
    for agent in unfinished
        x, y = get_obj_loc(state, agent)
        # Out-of-grid/occupied initial locations are outside the gridworld's
        # legal-state invariant; h=0 is conservative for such direct queries.
        1 <= x <= width && 1 <= y <= height || return 0.0
        (h.walls[y, x] || (x, y) in h.well_locs ||
         (x, y) in h.tank_locs) && return 0.0
        water1 = state[Compound(Symbol("has-water1"), Term[agent])]
        water2 = state[Compound(Symbol("has-water2"), Term[agent])]
        water3 = state[Compound(Symbol("has-water3"), Term[agent])]
        phase = if water3 && !water1 && !water2
            4
        elseif !water3 && water1 && water2
            3
        elseif !water3 && water1 && !water2
            2
        elseif !water3 && !water1 && !water2
            1
        else
            return 0.0 # A noncanonical flag combination has no claimed bound.
        end
        max_distance = max(max_distance, h.distances[y, x, phase])
    end
    return max_distance
end
