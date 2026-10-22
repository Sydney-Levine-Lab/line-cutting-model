using Test, Random, PDDL, PlanningDomains, SymbolicPlanners, CSV

# Load the real simulator definitions without invoking its run loop. Only the
# heuristic include is redirected, so projections, goals and PDDL actions are
# the production implementation. All writes remain in this review workspace.
const REPO_SRC = get(ENV, "LINES_TARGET_SRC", normpath(joinpath(@__DIR__, "..", "src")))
const HEURISTIC_FILE = get(ENV, "LINES_HEURISTIC_SRC", normpath(joinpath(@__DIR__, "..", "src", "water_collection_heuristic.jl")))
const BASELINE_SRC = get(ENV, "LINES_BASELINE_SRC", normpath(joinpath(REPO_SRC, "..", "..", "cogsci-2026", "src")))
const BASELINE_HEURISTIC = get(ENV, "LINES_BASELINE_HEURISTIC", joinpath(BASELINE_SRC, "water_collection_heuristic.jl"))
const OUTPUT_DIR = get(ENV, "LINES_TEST_OUTPUT") do
    mktempdir(; prefix="lines-heuristic-routes-", cleanup=false)
end

ENV["DEPTH"] = "0"
ENV["LEVEL"] = "0"
ENV["ZAP"] = "false"
ENV["TEMPERATURE"] = "0.0001"
ENV["EVAL_STEPCOST"] = "true"
ENV["TIEBREAK"] = "random"
ENV["WRITE_SNAPSHOT"] = "false"
ENV["TRAJECTORY_LEVEL"] = "none"

module Corrected
    main_file = joinpath(Main.REPO_SRC, "main.jl")
    source = read(main_file, String)
    old_include = "include(joinpath(@__DIR__, \"water_collection_heuristic.jl\"))"
    count(old_include, source) == 1 || error("Expected one heuristic include")
    source = replace(source, old_include => "include(" * repr(Main.HEURISTIC_FILE) * ")")
    Base.include_string(@__MODULE__, source, main_file)
end

module LegacyHeuristic
    include(Main.BASELINE_HEURISTIC)
end

module LegacySimulator
    main_file = joinpath(Main.REPO_SRC, "main.jl")
    source = read(main_file, String)
    old_include = "include(joinpath(@__DIR__, \"water_collection_heuristic.jl\"))"
    source = replace(source, old_include => "include(" * repr(Main.BASELINE_HEURISTIC) * ")")
    Base.include_string(@__MODULE__, source, main_file)
end

const ROUTE_ROWS = NamedTuple[]
const ACTION_ROWS = NamedTuple[]
const ORACLE_BUDGET = 100_000
const DOMAIN = PlanningDomains.load_domain(Corrected.DOMAIN_FILE)
const AGENT1 = Corrected.AGENTS[1]
const GOAL1 = MinStepsGoal(Term[Corrected.HAS_FILLED[1]])

oracle_planner() = AStarPlanner(NullHeuristic(); save_search=true, max_nodes=ORACLE_BUDGET)
function solution_distance(solution)
    solution.status == :success && return Float64(length(solution))
    solution.status == :failure && return Inf
    error("Search stopped with $(solution.status); this is not a shortest-path result")
end

function initial_world(map)
    problem = PlanningDomains.load_problem(joinpath(Corrected.MAPS_DIR, map))
    return initstate(DOMAIN, problem)
end

function lone_world(map; k=1)
    state = initial_world(map)
    return Corrected.remove_agents(state, DOMAIN, [n for n in 1:8 if n != k])
end

function stage(state, k=1)
    state[Corrected.HAS_FILLED[k]] && return :deposited
    state[Corrected.HAS_WATER3[k]] && return :loaded
    state[Corrected.HAS_WATER2[k]] && return :collect2
    state[Corrected.HAS_WATER1[k]] && return :collect1
    return :empty
end

function set_stage!(state, phase; k=1)
    state[Corrected.HAS_FILLED[k]] = phase == :deposited
    state[Corrected.HAS_WATER1[k]] = phase in (:collect1, :collect2)
    state[Corrected.HAS_WATER2[k]] = phase == :collect2
    state[Corrected.HAS_WATER3[k]] = phase == :loaded
    return state
end

function check_plan(solution, state, goal)
    solution.status == :success || return
    cursor = state
    for action in solution
        @test action in collect(PDDL.available(DOMAIN, cursor))
        cursor = transition(DOMAIN, cursor, action)
    end
    @test SymbolicPlanners.is_goal(goal, DOMAIN, cursor)
end

function validate_case(map, label, state; k=1, heuristic=Corrected.WaterCollectionHeuristic(), check_actions=true)
    goal = MinStepsGoal(Term[Corrected.HAS_FILLED[k]])
    reference = oracle_planner()(DOMAIN, state, goal)
    planner = AStarPlanner(heuristic; save_search=true, max_nodes=ORACLE_BUDGET)
    guided = planner(DOMAIN, state, goal)
    exact_distance = solution_distance(reference)
    guided_distance = solution_distance(guided)
    estimate = SymbolicPlanners.compute(heuristic, DOMAIN, state, goal)

    @test guided_distance == exact_distance
    @test estimate >= 0
    @test estimate <= exact_distance
    # Single-agent views reduce exactly to the occupancy/service graph. This
    # independent PDDL oracle tests the stronger exact-table claim as well.
    @test estimate == exact_distance
    check_plan(guided, state, goal)

    actions = collect(PDDL.available(DOMAIN, state))
    exact_values = Dict{Term,Float64}()
    guided_values = Dict{Term,Float64}()
    if check_actions
        exact_policy = FunctionalVPolicy(PlannerHeuristic(oracle_planner()), DOMAIN, goal)
        guided_policy = FunctionalVPolicy(PlannerHeuristic(planner), DOMAIN, goal)
        for action in actions
            next_state = transition(DOMAIN, state, action)
            next_estimate = SymbolicPlanners.compute(heuristic, DOMAIN, next_state, goal)
            @test estimate <= 1 + next_estimate
            exact_value = SymbolicPlanners.get_value(exact_policy, state, action)
            guided_value = SymbolicPlanners.get_value(guided_policy, state, action)
            @test exact_value == guided_value
            exact_values[action] = exact_value
            guided_values[action] = guided_value
            push!(ACTION_ROWS, (; map, label, agent=k, action=string(action), exact_value, guided_value))
        end
        @test Set(keys(exact_values)) == Set(keys(guided_values))
        if !isempty(exact_values)
            exact_best = maximum(values(exact_values))
            guided_best = maximum(values(guided_values))
            @test Set(a for (a, v) in exact_values if v == exact_best) ==
                  Set(a for (a, v) in guided_values if v == guided_best)
        end
    end
    push!(ROUTE_ROWS, (; map, label, agent=k, exact_distance, guided_distance,
                      heuristic=Float64(estimate), n_actions=length(actions),
                      exact_expanded=reference.expanded, guided_expanded=guided.expanded))
    return (; reference, guided, exact_values, guided_values)
end

@testset "goal-matched water collection routes" begin
    @test Corrected.ZAP == false
    @testset "fresh helper and deposited state" begin
        state = lone_world("yes_line_7.pddl")
        set_stage!(state, :deposited)
        h = Corrected.WaterCollectionHeuristic()
        @test SymbolicPlanners.compute(h, DOMAIN, state, GOAL1) == 0
        @test all(!state[p] for p in (Corrected.HAS_WATER1[1], Corrected.HAS_WATER2[1], Corrected.HAS_WATER3[1]))
        validate_case("yes_line_7.pddl", "fresh_deposited", state; heuristic=h)
    end

    @testset "reproduced finish-map regressions" begin
        for (map, x, y, expected, legacy_expected) in
            [("no_line_1.pddl", 1, 3, 2, 4), ("yes_line_7.pddl", 2, 10, 1, 6)]
            state = lone_world(map)
            state[Corrected.XLOC[1]], state[Corrected.YLOC[1]] = x, y
            set_stage!(state, :loaded)
            result = validate_case(map, "loaded_regression", state)
            @test solution_distance(result.reference) == expected
            legacy = AStarPlanner(LegacyHeuristic.WaterCollectionHeuristic())(DOMAIN, state, GOAL1)
            @test solution_distance(legacy) == legacy_expected
            if map == "no_line_1.pddl"
                @test result.guided_values[pddl"(down agent1)"] == -2
                @test result.guided_values[pddl"(right agent1)"] == -4
            end
        end
    end

    @testset "all canonical maps and legal water phases" begin
        seen_phases = Set{Symbol}()
        shared_heuristic = Corrected.WaterCollectionHeuristic()
        for map in Corrected.CANONICAL_MAP_FILES
            state = lone_world(map)
            initial = validate_case(map, "lone_initial", state; heuristic=shared_heuristic)
            @test initial.reference.status == :success
            states = initial.reference.trajectory
            for phase in (:collect1, :collect2, :loaded, :deposited)
                idx = findfirst(s -> stage(s) == phase, states)
                @test idx !== nothing
                if idx !== nothing
                    push!(seen_phases, phase)
                    validate_case(map, string(phase), states[idx]; heuristic=shared_heuristic)
                end
            end
            loaded_states = filter(s -> stage(s) == :loaded, states)
            if length(loaded_states) > 2
                validate_case(map, "loaded_midroute", loaded_states[cld(length(loaded_states), 2)]; heuristic=shared_heuristic)
            end
            # Real initial planning snapshots include the other seven agents
            # as walls. No actual population behavior is changed by fixtures.
            world = initial_world(map)
            for k in (1, 8)
                projected = Corrected.project_others_to_walls(world, k, DOMAIN)
                validate_case(map, "occupied_initial", projected; k, heuristic=shared_heuristic)
            end
            println("validated ", map, "; route cases so far = ", length(ROUTE_ROWS))
            flush(stdout)
        end
        @test seen_phases == Set((:collect1, :collect2, :loaded, :deposited))
    end

    @testset "geometry changes and unreachable states" begin
        state = lone_world("yes_line_8.pddl")
        state[Corrected.XLOC[1]], state[Corrected.YLOC[1]] = 2, 10
        set_stage!(state, :loaded)
        h = Corrected.WaterCollectionHeuristic()
        @test solution_distance(validate_case("yes_line_8.pddl", "geometry_before", state; heuristic=h).guided) == 1
        trapped = copy(state)
        walls = copy(trapped[pddl"(walls)"])
        services = [(trapped[Compound(:xloc, Term[o])], trapped[Compound(:yloc, Term[o])])
                    for o in vcat(PDDL.get_objects(trapped, :tank), PDDL.get_objects(trapped, :well))]
        found = false
        for y in 2:(size(walls, 1)-1), x in 2:(size(walls, 2)-2)
            any(abs(x-sx) + abs(y-sy) <= 2 for (sx, sy) in services) && continue
            walls .= true
            walls[y, x] = walls[y, x+1] = false
            trapped[Corrected.XLOC[1]], trapped[Corrected.YLOC[1]] = x, y
            found = true
            break
        end
        @test found
        trapped[pddl"(walls)"] = walls
        @test SymbolicPlanners.compute(h, DOMAIN, trapped, GOAL1) == Inf
        result = validate_case("yes_line_8.pddl", "unreachable_changed_geometry", trapped; heuristic=h)
        @test result.reference.status == :failure
        @test SymbolicPlanners.compute(h, DOMAIN, state, GOAL1) == 1
        validate_case("yes_line_8.pddl", "geometry_restored", state; heuristic=h)
        deposited = set_stage!(copy(trapped), :deposited)
        @test SymbolicPlanners.compute(h, DOMAIN, deposited, GOAL1) == 0
    end

    @testset "unsupported goal fallback and non-goal agents" begin
        state = lone_world("no_line_1.pddl")
        h = Corrected.WaterCollectionHeuristic()
        finish_goal = MinStepsGoal(pddl"(has-completed agent1)")
        coord_goal = MinStepsGoal(pddl"(= (xloc agent1) 2)")
        @test SymbolicPlanners.compute(h, DOMAIN, state, finish_goal) == 0
        @test SymbolicPlanners.compute(h, DOMAIN, state, coord_goal) == 0
        @test solution_distance(AStarPlanner(h)(DOMAIN, state, coord_goal)) ==
              solution_distance(oracle_planner()(DOMAIN, state, coord_goal))
        world = initial_world("no_line_1.pddl")
        world[Corrected.HAS_FILLED[1]] = true
        @test SymbolicPlanners.compute(h, DOMAIN, world, GOAL1) == 0
        agent8 = lone_world("no_line_1.pddl"; k=8)
        validate_case("no_line_1.pddl", "goal_agent8", agent8; k=8, heuristic=h)
    end

    @testset "completed ghost behavior remains identical" begin
        @test LegacySimulator.ZAP == Corrected.ZAP == false
        old_policies = LegacySimulator.build_agent_policies(DOMAIN, LegacySimulator.build_steps_to_go_estimator())
        new_policies = Corrected.build_agent_policies(DOMAIN, Corrected.build_steps_to_go_estimator())
        for map in Corrected.CANONICAL_MAP_FILES
            # A completed ghost at its former position remains present and
            # faces exactly the same action set/other-agent obstacles.
            world = initial_world(map)
            set_stage!(world, :deposited)
            old_view = LegacySimulator.project_others_to_walls(world, 1, DOMAIN)
            new_view = Corrected.project_others_to_walls(world, 1, DOMAIN)
            @test old_view == new_view
            @test Set(PDDL.available(DOMAIN, old_view)) == Set(PDDL.available(DOMAIN, new_view))
            old_q = Dict(SymbolicPlanners.get_action_values(old_policies[1], old_view))
            new_q = Dict(SymbolicPlanners.get_action_values(new_policies[1], new_view))
            @test old_q == new_q
            @test all(==(-1.0), values(new_q))
            for seed in 1:5
                Random.seed!(seed)
                old_action = SymbolicPlanners.get_action(old_policies[1], old_view)
                Random.seed!(seed)
                new_action = SymbolicPlanners.get_action(new_policies[1], new_view)
                @test old_action == new_action
            end
        end
    end
end

mkpath(OUTPUT_DIR)
CSV.write(joinpath(OUTPUT_DIR, "route_validation.csv"), ROUTE_ROWS)
CSV.write(joinpath(OUTPUT_DIR, "action_value_validation.csv"), ACTION_ROWS)
println("Saved ", length(ROUTE_ROWS), " route cases and ", length(ACTION_ROWS), " exact action-value comparisons to ", OUTPUT_DIR)
println("Only heuristic definitions were substituted. Ghosts, timing, choice noise and simulator source were unchanged.")
