# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

const N_AGENTS     = 8                   # Must match problem files
const TIME_MAX        = 1000             # Upper bound on timesteps for a single run
const TEMPERATURE  = 0.3                 # Boltzmann temperature
const RUNS         = 5                   # Number of runs per map

const COMPLETION_TIMES_DIR  = "../data/simulations/test2/"
const FULL_HISTORY_DIR      = joinpath(COMPLETION_TIMES_DIR, "full_histories")

const DOMAIN_PATH = "."
const MAPS_PATH   = "./maps"

mkpath(COMPLETION_TIMES_DIR)
mkpath(FULL_HISTORY_DIR)

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

if abspath(PROGRAM_FILE) == @__FILE__
    project_others_to_walls(state::State, k::Integer, N::Int, domain::Domain)
end
