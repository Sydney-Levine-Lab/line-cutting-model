using PDDL, PlanningDomains
using SymbolicPlanners
using Base.Threads
using Random
using CSV, Printf
using Dates
using Pkg

include(joinpath(@__DIR__, "water_collection_heuristic.jl"))

# Enable array-valued fluents (e.g. wall grids) for this PDDL domain
PDDL.Arrays.@register()

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

const N_AGENTS    = 8        # Must match map files
const TIME_MAX    = 1000     # Upper bound on timesteps for a single run
const TEMPERATURE = 0.0001   # Boltzmann temperature
const RUNS        = 50       # Number of runs per map

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
const RUN_LABEL = get(ENV, "RUN_LABEL", "main")
const RUN_ID    = isempty(strip(RUN_LABEL)) ?
    Dates.format(now(), "yyyy-mm-dd_HHMMSS") :
    RUN_LABEL * "_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS")

# Random seed = BASE_SEED + MAP_SEED_OFFSET * map_number + run_number
const BASE_SEED         = 1234
const MAP_SEED_OFFSET   = 10_000

const OUTPUT_DIR        = joinpath(SRC_DIR, "..", "data", "simulations", RUN_ID, "raw")
const TRAJECTORY_DIR    = joinpath(OUTPUT_DIR, "trajectories")

const SAVE_TRAJECTORIES = true   # For now: save trajectories by default (TBD time it takes)
const VERBOSE = true
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
        agent         = AGENTS[n]
        goal          = MinStepsGoal(Term[HAS_FILLED[n]])
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
        agent = AGENTS[n]
        # save the coordinates of the agent, in order to place a wall there
        agent_locs[agent_index] = (state[XLOC[n]], state[YLOC[n]])
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

    # Assign walls fluent
    fluents[pddl"(walls)"] = walls
    # Assign position of remaining agent
    fluents[XLOC[k]] = state[XLOC[k]]
    fluents[YLOC[k]] = state[YLOC[k]]
    
    # Task-status fluents for the remaining agent
    fluents[HAS_FILLED[k]] = state[HAS_FILLED[k]]
    fluents[HAS_WATER1[k]] = state[HAS_WATER1[k]]
    fluents[HAS_WATER2[k]] = state[HAS_WATER2[k]]
    fluents[HAS_WATER3[k]] = state[HAS_WATER3[k]]
    
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
    mkpath(TRAJECTORY_DIR)

    println("Starting simulations:")
    println("  maps = $(length(MAP_FILES))")
    println("  runs per map = $(RUNS)")
    println("  threads = $(Threads.nthreads())")
    println("  output dir = $(OUTPUT_DIR)")
    println("  save trajectories = $(SAVE_TRAJECTORIES)")
    println()

    # Reproducibility snapshot (files, once per run)
    #open(joinpath(OUTPUT_DIR, "versions.txt"), "w") do io
    #    println(io, "Julia ", VERSION)
    #    println(io)
    #    versioninfo(io)
    #end

    #open(joinpath(OUTPUT_DIR, "pkg_status.txt"), "w") do io
    #    println(io, "Project dependencies:")
    #    Pkg.status(io)
    #    println(io)
    #    println("Full manifest:")
    #    Pkg.status(io; mode=Pkg.PKGMODE_MANIFEST)
    #end

   # for f in ("Project.toml", "Manifest.toml")
    #    src = joinpath(SRC_DIR, f)
    #    if isfile(src)
    #        cp(src, joinpath(OUTPUT_DIR, f); force=true)
    #    end
    #end

    # Parallelize over maps
    Threads.@threads for map_index in eachindex(MAP_FILES)
        map = MAP_FILES[map_index]
        map_name = replace(map, ".pddl" => "")
        if VERBOSE
            #lock(print_lock) do
            println("Starting map $map_name (thread $(threadid()))")
            #end
        end

        # Load domain and build policies
        domain = PlanningDomains.load_domain(DOMAIN_FILE)
        steps_to_go = build_steps_to_go_estimator()
        policies = build_agent_policies(domain, steps_to_go)

        # Load problem / initial state for this map
        problem = PlanningDomains.load_problem(joinpath(MAPS_DIR, map))
        initial_state = initstate(domain, problem)

        # Results container (one row per run) for this map
        results = Vector{NamedTuple}(undef, RUNS)

        for run in 1:RUNS
            println("[$(threadid())] map=$(map) run=$(run)")

            # Seed per (map_index, run) so results are reproducible independent of thread scheduling.
            seed_this_run = BASE_SEED + MAP_SEED_OFFSET * map_index + run
            Random.seed!(seed_this_run)

            # Reset current state
            state = initial_state
            # Keep previous two states for detecting if an agent is stuck
            state_t_minus_1 = state
            state_t_minus_2 = state

            # Record timestep when agent n first fills a tank (0 if not yet, -1 if stuck)
            agent_filled = fill(0, N_AGENTS)

            traj_name = "trajectory_$(map_name)_run$(run).log"
            traj_path = joinpath(TRAJECTORY_DIR, traj_name)
            
            elapsed_run = @elapsed begin
                if SAVE_TRAJECTORIES
                    open(traj_path, "w") do io
                    # write header + t=0 in trajectory file
                    @printf(io, "# map=%s run=%d seed=%d temp=%.6g\n", map, run, seed_this_run, TEMPERATURE)
                    println(io, "# t=0")
                    show(io, state); println(io)
            
                    for t in 1:TIME_MAX
                        state = simulation_step(state, policies, domain)

                        println(io, "# t=$t")
                        show(io, state); println(io)
                
                        # Record first time any agent fills tank
                        for n in 1:N_AGENTS
                            if agent_filled[n] == 0 && state[HAS_FILLED[n]]
                                agent_filled[n] = t
                            end
                        end

                        # Break if stuck for 3 consecutive steps
                        if t>=3 && state == state_t_minus_1 && state_t_minus_1 == state_t_minus_2
                            fill!(agent_filled, -1)
                            break
                        end
                
                        # Break if all agents filled tank
                        if all(>(0), agent_filled)
                            break
                        end

                        # Shift states
                        state_t_minus_2 = state_t_minus_1
                        state_t_minus_1 = state
                    end
                end
                else
                    # Run the simulation without saving trajectory log
                    for t in 1:TIME_MAX
                        state = simulation_step(state, policies, domain)

                        # Record first time any agent fills tank
                        for n in 1:N_AGENTS
                            if agent_filled[n] == 0 && state[HAS_FILLED[n]]
                                agent_filled[n] = t
                            end
                        end

                        # Break if stuck for 3 consecutive steps
                        if t>=3 && state == state_t_minus_1 && state_t_minus_1 == state_t_minus_2
                            fill!(agent_filled, -1)
                            break
                        end
                
                        # Break if all agents filled tank
                        if all(>(0), agent_filled)
                            break
                        end

                        # Shift states
                        state_t_minus_2 = state_t_minus_1
                        state_t_minus_1 = state
                    end
                end
            end

            # mark unfinished agents as -1
            for n in 1:N_AGENTS
                if agent_filled[n] == 0
                    agent_filled[n] = -1
                end
            end
            
            results[run] = (
                run = run, 
                agent1 = agent_filled[1],
                agent2 = agent_filled[2],
                agent3 = agent_filled[3],
                agent4 = agent_filled[4],
                agent5 = agent_filled[5],
                agent6 = agent_filled[6],
                agent7 = agent_filled[7],
                agent8 = agent_filled[8],
                map = map_name,
                seed = seed_this_run,
                temperature = TEMPERATURE,
                time_max = TIME_MAX,
                n_agents = N_AGENTS,
                run_elapsed_seconds = elapsed_run,
            )    
        end

        # Save per-(map, runs) results.
        csv_name = joinpath(OUTPUT_DIR, replace(map, ".pddl" => "") * ".csv")
        CSV.write(csv_name, results)

        println("Finished map $map_name → wrote $(basename(csv_name))")
    end
end

# Run if called as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulations()
end
