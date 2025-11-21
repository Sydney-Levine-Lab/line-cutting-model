"Converts ASCII gridworlds to PDDL problem."
function ascii_to_pddl_problem(str::String, name="lines-problem")
    rows = split(str, "\n", keepempty=false)
    width, height = maximum(length.(strip.(rows))), length(rows)
    agents, wells, tanks = Const[], Const[], Const[]
    walls = parse_pddl("(= walls (new-bit-matrix false $width $height))")
    init = Term[walls]
    goal = pddl"(forall (?a - agent) (has-filled ?a))"
    for (y, row) in enumerate(rows)
        for (x, char) in enumerate(strip(row))
            if char == '.' # Unoccupied
                continue
            elseif char == 'W' # Wall
                wall = parse_pddl("(= walls (set-index walls true $y $x))")
                push!(init, wall)
            elseif char == 'w' # Well / water
                w = Const(Symbol("well$(length(wells)+1)"))
                push!(wells, w)
                append!(init, parse_pddl("(= (xloc $w) $x)", "(= (yloc $w) $y)"))
            elseif char == 't' # Tank
                t = Const(Symbol("tank$(length(tanks)+1)"))
                push!(tanks, t)
                append!(init, parse_pddl("(= (xloc $t) $x)", "(= (yloc $t) $y)"))
            elseif char in "0123456789" # Agent
                a = Const(Symbol("agent$char"))
                push!(agents, a)
                append!(init, parse_pddl("(= (xloc $a) $x)", "(= (yloc $a) $y)"))
            end
        end
    end
    objtypes = merge(Dict(x => :agent for x in agents),
                     Dict(x => :well for x in wells),
                     Dict(x => :tank for x in tanks))
    problem = GenericProblem(Symbol(name), Symbol("lines-gridworld"),
                             [agents; wells; tanks],
                             objtypes, init, goal, nothing)
    return problem
end

"Load ASCII from path, and convert to PDDL problem."
function load_ascii_problem(path::String)
    str = open(f->read(f, String), path)
    return ascii_to_pddl_problem(str)
end

"Print PDDL state as ASCII grid."
function print_ascii(io::IO, state::State, pretty::Bool=true)
    walls = state[pddl"walls"]
    height, width = size(walls)
    agents = PDDL.get_objects(state, :agent)
    agent_locs = get_object_locs(state, :agent)
    well_locs = get_object_locs(state, :well)
    tank_locs = get_object_locs(state, :tank)
    for y in 1:height
        for x in 1:width
            if (x, y) in agent_locs # Agents
                idx = findfirst(loc -> loc == (x, y), agent_locs)
                char = string(agents[idx].name)[end]
                print(io, char)
            elseif (x, y) in well_locs # Wells
                print(io, 'o')
            elseif (x, y) in tank_locs # Tanks
                print(io, 't')
            elseif walls[y, x] # Walls
                print(io, 'W')
            else # Empty cells
                print(io, '.')
            end
            # Print additional space for readibility
            if pretty print(io, ' ') end
        end
        # Go to next row
        println(io)
    end
end

print_ascii(state::State, pretty::Bool=true) =
    print_ascii(stdout, state, pretty)

"Return list of object locations."
function get_object_locs(state::State, type::Symbol=:object)
    return [(state[Compound(:xloc, Term[o])], state[Compound(:yloc, Term[o])])
             for o in PDDL.get_objects(state, type)]
end
